import { useCallback, useEffect, useMemo, useState } from "react";
import axios from "axios";
import { ArrowRight, ChevronLeft, ChevronRight, Compass, Link2, MapPinned, Plus, Wrench } from "lucide-react";
import { API_URL, getSessionHeaders, getSessionId } from "./config";

function App() {
  const [repoUrl, setRepoUrl] = useState("");
  const [repos, setRepos] = useState([]);
  const [selectedRepoId, setSelectedRepoId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [activeAnswerId, setActiveAnswerId] = useState(null);
  const [indexing, setIndexing] = useState(false);
  const [loadingRepos, setLoadingRepos] = useState(true);
  const [asking, setAsking] = useState(false);
  const [formError, setFormError] = useState("");
  const [stage, setStage] = useState("landing");
  const [citationsOpen, setCitationsOpen] = useState(true);
  const sessionHeaders = useMemo(() => getSessionHeaders(), []);

  const selectedRepo = useMemo(
    () => repos.find((repo) => repo.id === selectedRepoId) || null,
    [repos, selectedRepoId],
  );

  const answerMessages = useMemo(
    () => messages.filter((message) => message.answerData),
    [messages],
  );

  const activeAnswerMessage = useMemo(() => {
    if (activeAnswerId) {
      return answerMessages.find((message) => message.id === activeAnswerId) || null;
    }
    return answerMessages.at(-1) || null;
  }, [activeAnswerId, answerMessages]);

  const refreshRepos = useCallback(async (preserveSelection = true) => {
    setLoadingRepos(true);
    try {
      const { data } = await axios.get(`${API_URL}/api/repos`, {
        headers: sessionHeaders,
      });
      setRepos(data);

      if (data.length === 0) {
        setSelectedRepoId(null);
      } else if (!preserveSelection || !data.some((repo) => repo.id === selectedRepoId)) {
        setSelectedRepoId(data[0].id);
      }
    } catch {
      setFormError("Unable to load repositories right now.");
    } finally {
      setLoadingRepos(false);
    }
  }, [selectedRepoId, sessionHeaders]);

  useEffect(() => {
    refreshRepos(false);
  }, [refreshRepos]);

  useEffect(() => {
    if (selectedRepo?.status === "indexed") {
      setIndexing(false);
      setFormError("");
      setStage("workspace");
      return;
    }

    if (selectedRepo?.status === "failed") {
      setIndexing(false);
      setStage("landing");
      setFormError(formatRepoError(selectedRepo.error_message));
      return;
    }

    if (!selectedRepo || !["queued", "indexing"].includes(selectedRepo.status)) {
      setIndexing(false);
      return;
    }

    setIndexing(true);
    const timer = setInterval(() => {
      refreshRepos(true);
    }, 3000);

    return () => clearInterval(timer);
  }, [refreshRepos, selectedRepo]);

  const startIndexing = async () => {
    const value = repoUrl.trim();
    if (!value) {
      setFormError("Enter a GitHub repository URL.");
      return;
    }

    setIndexing(true);
    setFormError("");
    setMessages([]);
    setActiveAnswerId(null);

    try {
      const { data } = await axios.post(
        `${API_URL}/api/repos/index`,
        { github_url: value },
        { headers: sessionHeaders },
      );

      const nextRepo = data.repo;
      setSelectedRepoId(nextRepo.id);
      setMessages([
        {
          id: `system-${Date.now()}`,
          role: "system",
          content: "Hey, what question do you have for me today?",
        },
      ]);
      await refreshRepos(true);
    } catch (error) {
      setIndexing(false);
      setFormError(error?.response?.data?.detail || "Failed to start");
    }
  };

  const sendQuestion = async () => {
    if (!selectedRepo) {
      setFormError("Index a repository first.");
      return;
    }

    if (selectedRepo.status !== "indexed") {
      setFormError("Wait for indexing to finish before asking questions.");
      return;
    }

    const value = question.trim();
    if (!value) return;

    const userMessage = {
      id: `user-${Date.now()}`,
      role: "user",
      content: value,
    };

    setMessages((current) => [...current, userMessage]);
    setQuestion("");
    setFormError("");
    setAsking(true);

    try {
      const { data } = await axios.post(
        `${API_URL}/api/query`,
        {
          repo_id: selectedRepo.id,
          question: value,
          top_k: 8,
          history: messages
            .filter(
              (message) =>
                (message.role === "user" || message.role === "assistant")
                && typeof message.content === "string"
                && message.content.trim()
                && (message.role !== "assistant" || message.answerData),
            )
            .slice(-6)
            .map((message) => ({
              role: message.role,
              content: message.content,
            })),
        },
        {
          headers: sessionHeaders,
        },
      );

      const assistantMessage = {
        id: `answer-${Date.now()}`,
        role: "assistant",
        content: data.answer,
        answerData: data,
      };

      setMessages((current) => [...current, assistantMessage]);
      setActiveAnswerId(assistantMessage.id);
    } catch (error) {
      setFormError(error?.response?.data?.detail || "Query failed.");
    } finally {
      setAsking(false);
    }
  };

  const endSession = async () => {
    try {
      await axios.post(`${API_URL}/api/session/end?session_id=${encodeURIComponent(getSessionId())}`);
    } catch {
      // Best effort cleanup.
    } finally {
      window.sessionStorage.clear();
      setRepos([]);
      setSelectedRepoId(null);
      setMessages([]);
      setActiveAnswerId(null);
      setQuestion("");
      setRepoUrl("");
      setFormError("");
      setIndexing(false);
      setStage("landing");
    }
  };

  if (stage === "landing") {
    return (
      <LandingScreen
        formError={formError}
        indexing={indexing}
        loadingRepos={loadingRepos}
        repoUrl={repoUrl}
        selectedRepo={selectedRepo}
        setRepoUrl={setRepoUrl}
        startIndexing={startIndexing}
      />
    );
  }

  return (
    <WorkspaceScreen
      activeAnswerMessage={activeAnswerMessage}
      asking={asking}
      citationsOpen={citationsOpen}
      endSession={endSession}
      formError={formError}
      messages={messages}
      question={question}
      selectedRepo={selectedRepo}
      sendQuestion={sendQuestion}
      setActiveAnswerId={setActiveAnswerId}
      setCitationsOpen={setCitationsOpen}
      setQuestion={setQuestion}
    />
  );
}

function LandingScreen({
  formError,
  indexing,
  loadingRepos,
  repoUrl,
  selectedRepo,
  setRepoUrl,
  startIndexing,
}) {
  return (
    <div
      className="h-screen overflow-hidden text-white"
      style={APP_BACKGROUND_STYLE}
    >
      <div className="flex h-screen items-center justify-center px-6">
        <div className="w-full max-w-3xl">
          <div className="mb-10 flex items-center justify-center gap-3 text-zinc-300">
            <Compass className="h-8 w-8" strokeWidth={1.8} />
            <span className="font-display text-lg uppercase tracking-[0.36em] text-zinc-300">
              Code Compass
            </span>
          </div>
          <div className="text-center">
            <h1 className="mt-4 font-display text-5xl tracking-[-0.06em] text-white md:text-7xl">
              Ask a GitHub repo anything.
            </h1>
          </div>

          <div className="mt-12 rounded-[32px] border border-white/10 bg-white/[0.04] p-4 shadow-[0_40px_120px_rgba(0,0,0,0.45)] backdrop-blur-xl">
            <div className="flex flex-col gap-3 md:flex-row">
              <input
                className="h-16 flex-1 rounded-[24px] border border-white/20 bg-white/10 px-5 text-base text-white outline-none backdrop-blur-xl transition placeholder:text-zinc-300/70 focus:border-white/35"
                placeholder="Paste GitHub URL"
                value={repoUrl}
                onChange={(event) => setRepoUrl(event.target.value)}
              />
              <button
                className="flex h-16 min-w-[88px] items-center justify-center rounded-[24px] bg-white px-4 text-sm font-semibold text-black transition hover:bg-zinc-200 disabled:cursor-wait disabled:bg-zinc-500"
                onClick={startIndexing}
                disabled={indexing}
              >
                {indexing ? <SpinnerOnly /> : <ArrowRight className="h-6 w-6" strokeWidth={2.2} />}
              </button>
            </div>

            {formError && <p className="mt-4 text-sm text-rose-400">{formError}</p>}
            {(selectedRepo?.progress?.message || (selectedRepo?.name && indexing)) && (
              <div className="mt-5 flex flex-wrap items-center gap-3 text-sm text-zinc-500">
                {selectedRepo?.progress?.message && (
                  <span className="rounded-full border border-white/10 bg-white/[0.025] px-3 py-1.5 text-zinc-400/85">
                    {selectedRepo.progress.message}
                  </span>
                )}
                {selectedRepo?.name && indexing && (
                  <span className="rounded-full border border-white/10 bg-white/[0.025] px-3 py-1.5 text-zinc-400/85">
                    preparing {selectedRepo.owner}/{selectedRepo.name}
                  </span>
                )}
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}

function WorkspaceScreen({
  activeAnswerMessage,
  asking,
  citationsOpen,
  endSession,
  formError,
  messages,
  question,
  selectedRepo,
  sendQuestion,
  setActiveAnswerId,
  setCitationsOpen,
  setQuestion,
}) {
  const hasCitations = Boolean(activeAnswerMessage?.answerData?.sources?.length);

  return (
    <div
      className="h-screen overflow-hidden text-white"
      style={APP_BACKGROUND_STYLE}
    >
      <div
        className={[
          "grid h-screen grid-cols-1 gap-px overflow-hidden bg-transparent",
          hasCitations && citationsOpen ? "xl:grid-cols-[minmax(0,1fr)_400px]" : "",
        ].join(" ")}
      >
        <section className="flex min-h-0 h-screen flex-col overflow-hidden bg-black/45 backdrop-blur-[3px]">
          <div className="flex items-center justify-between border-b border-white/10 px-5 py-4 md:px-8">
            <div>
              <h2 className="font-display text-2xl tracking-[-0.05em] text-white md:text-3xl">
                {selectedRepo ? `${selectedRepo.owner}/${selectedRepo.name}` : "Repository"}
              </h2>
            </div>
            <div className="flex items-center gap-4">
              <button
                className="inline-flex items-center gap-2 rounded-full bg-white px-4 py-2 text-sm font-medium text-black transition hover:bg-zinc-200"
                onClick={endSession}
                type="button"
                aria-label="Start new session"
              >
                <Plus className="h-4 w-4" strokeWidth={2.1} />
                New
              </button>
              {hasCitations && !citationsOpen ? (
                <button
                  className="text-zinc-500 transition hover:text-white"
                  onClick={() => setCitationsOpen(true)}
                  type="button"
                  aria-label="Expand citations"
                >
                  <ChevronLeft className="h-5 w-5" strokeWidth={2.1} />
                </button>
              ) : null}
            </div>
          </div>

          <div className="scrollbar-thin flex flex-1 flex-col gap-4 overflow-auto px-5 py-6 md:px-8">
            {messages.length === 0 ? (
              <div className="flex flex-1 items-center justify-center rounded-[28px] border border-dashed border-white/10 bg-white/[0.02] p-8 text-center text-zinc-500">
                Ask a question
              </div>
            ) : (
              messages.map((message) => (
                <article
                  key={message.id}
                  className={[
                    "max-w-4xl rounded-[28px] border px-5 py-4",
                    message.role === "user"
                      ? "ml-auto border-white/25 bg-white/[0.16] text-white backdrop-blur-xl"
                      : "border-white/10 bg-white/[0.03] text-white",
                  ].join(" ")}
                  onClick={() => {
                    if (message.answerData) {
                      setActiveAnswerId(message.id);
                    }
                  }}
                >
                  {message.role !== "user" ? (
                    <p className="mb-2 text-[11px] uppercase tracking-[0.24em] text-zinc-500">
                      {selectedRepo ? `${selectedRepo.owner}/${selectedRepo.name}` : "Repository"}
                    </p>
                  ) : null}
                  {message.answerData ? (
                    <AnswerBlock answer={message.answerData} />
                  ) : (
                    <p className="text-sm leading-7">{message.content}</p>
                  )}
                </article>
              ))
            )}
          </div>

          <div className="border-t border-white/10 bg-transparent px-5 py-4 md:px-8">
            <div className="rounded-[32px] border border-white/10 bg-white/[0.04] p-4 shadow-[0_40px_120px_rgba(0,0,0,0.45)] backdrop-blur-xl">
              <div className="flex flex-col gap-3">
                <div className="relative">
                  <textarea
                    className="min-h-[72px] w-full resize-none rounded-[24px] border border-white/20 bg-transparent px-5 py-4 pr-16 text-sm text-white outline-none transition placeholder:text-zinc-300/70 focus:border-white/35"
                    placeholder="Ask a question"
                    value={question}
                    onChange={(event) => setQuestion(event.target.value)}
                  />
                  <button
                    className="absolute bottom-4 right-4 flex h-9 w-9 items-center justify-center rounded-full bg-white/10 text-white backdrop-blur-xl transition hover:bg-white/10 disabled:cursor-wait disabled:text-zinc-500"
                    onClick={sendQuestion}
                    disabled={asking}
                  >
                    {asking ? <SpinnerOnly /> : <ArrowRight className="h-5 w-5" strokeWidth={2.2} />}
                  </button>
                </div>
                <div className="text-sm text-rose-400">{formError || ""}</div>
              </div>
            </div>
          </div>
        </section>

        {hasCitations && citationsOpen ? (
          <aside className="relative flex min-h-0 h-screen flex-col overflow-hidden border-l border-white/10 bg-black/50 backdrop-blur-[3px]">
            <div className="flex min-h-[69px] items-center border-b border-white/10 px-5 py-4">
              <div className="flex w-full items-start justify-between gap-3">
                <div>
                  <p className="text-sm font-medium uppercase tracking-[0.24em] text-zinc-400">
                    Citations
                  </p>
                </div>
                <button
                  className="text-zinc-500 transition hover:text-white"
                  onClick={() => setCitationsOpen(false)}
                  type="button"
                  aria-label="Collapse citations"
                >
                  <ChevronRight className="h-5 w-5" strokeWidth={2.1} />
                </button>
              </div>
            </div>

            <div className="scrollbar-thin flex-1 overflow-auto px-5 py-5">
              <CitationRail answer={activeAnswerMessage.answerData} repo={selectedRepo} />
            </div>
          </aside>
        ) : null}
      </div>
    </div>
  );
}

function AnswerBlock({ answer }) {
  return (
    <div className="space-y-4 text-sm leading-7 text-zinc-100">
      <MarkdownAnswer value={answer.answer} />
    </div>
  );
}

function MarkdownAnswer({ value }) {
  const lines = String(value || "").replace(/\r\n/g, "\n").split("\n");
  const elements = [];
  let paragraph = [];
  let listItems = [];
  let codeLines = [];
  let inCodeBlock = false;

  const flushParagraph = () => {
    if (!paragraph.length) return;
    elements.push(
      <p key={`p-${elements.length}`} className="whitespace-pre-wrap text-sm leading-7 text-zinc-100">
        {renderInlineMarkdown(paragraph.join(" "))}
      </p>,
    );
    paragraph = [];
  };

  const flushList = () => {
    if (!listItems.length) return;
    elements.push(
      <ul key={`ul-${elements.length}`} className="list-disc space-y-2 pl-5 text-sm leading-7 text-zinc-100">
        {listItems.map((item, index) => (
          <li key={`li-${index}`}>{renderInlineMarkdown(item)}</li>
        ))}
      </ul>,
    );
    listItems = [];
  };

  const flushCodeBlock = () => {
    if (!codeLines.length) return;
    elements.push(
      <pre
        key={`code-${elements.length}`}
        className="overflow-x-auto rounded-2xl border border-white/10 bg-black/30 p-4 text-xs leading-6 text-zinc-100"
    >
        <code>{codeLines.join("\n")}</code>
      </pre>,
    );
    codeLines = [];
  };

  lines.forEach((line) => {
    const trimmed = line.trim();

    if (trimmed.startsWith("```")) {
      flushParagraph();
      flushList();
      if (inCodeBlock) {
        flushCodeBlock();
        inCodeBlock = false;
      } else {
        inCodeBlock = true;
      }
      return;
    }

    if (inCodeBlock) {
      codeLines.push(line);
      return;
    }

    const headingMatch = trimmed.match(/^(#{1,3})\s+(.*)$/);
    if (headingMatch) {
      flushParagraph();
      flushList();
      const level = headingMatch[1].length;
      const text = headingMatch[2];
      const className =
        level === 1
          ? "text-xl font-semibold text-white"
          : level === 2
            ? "text-lg font-semibold text-white"
            : "text-base font-semibold text-zinc-100";
      elements.push(
        <div key={`h-${elements.length}`} className={className}>
          {renderInlineMarkdown(text)}
        </div>,
      );
      return;
    }

    const listMatch = trimmed.match(/^[-*]\s+(.*)$/);
    if (listMatch) {
      flushParagraph();
      listItems.push(listMatch[1]);
      return;
    }

    if (!trimmed) {
      flushParagraph();
      flushList();
      return;
    }

    paragraph.push(trimmed);
  });

  flushParagraph();
  flushList();
  if (inCodeBlock || codeLines.length) {
    flushCodeBlock();
  }

  return <div className="space-y-4">{elements}</div>;
}

function renderInlineMarkdown(text) {
  const nodes = [];
  const pattern = /(`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\]\([^)]+\))/g;
  let lastIndex = 0;
  let match;

  while ((match = pattern.exec(text)) !== null) {
    if (match.index > lastIndex) {
      nodes.push(text.slice(lastIndex, match.index));
    }

    const token = match[0];
    if (token.startsWith("`") && token.endsWith("`")) {
      nodes.push(
        <code
          key={`code-${match.index}`}
          className="rounded bg-white/10 px-1.5 py-0.5 text-[0.95em] text-zinc-100"
        >
          {token.slice(1, -1)}
        </code>,
      );
    } else if (token.startsWith("**") && token.endsWith("**")) {
      nodes.push(
        <strong key={`strong-${match.index}`} className="font-semibold text-white">
          {token.slice(2, -2)}
        </strong>,
      );
    } else {
      const linkMatch = token.match(/^\[([^\]]+)\]\(([^)]+)\)$/);
      if (linkMatch) {
        nodes.push(
          <a
            key={`link-${match.index}`}
            className="text-zinc-100 underline decoration-white/30 underline-offset-4 hover:decoration-white"
            href={linkMatch[2]}
            target="_blank"
            rel="noreferrer"
          >
            {linkMatch[1]}
          </a>,
        );
      } else {
        nodes.push(token);
      }
    }

    lastIndex = match.index + token.length;
  }

  if (lastIndex < text.length) {
    nodes.push(text.slice(lastIndex));
  }

  return nodes;
}

function CitationRail({ answer, repo }) {
  const sources = [...(answer.sources || [])].sort(
    (left, right) => getConfidencePercentValue(right) - getConfidencePercentValue(left),
  );

  return (
    <div className="space-y-4">
      {sources.map((source, index) => (
        <article key={`${source.file_path}-${index}`} className="flex min-h-[180px] flex-col rounded-[28px] border border-white/10 bg-white/[0.03] p-5">
          <div className="flex items-start justify-between gap-3">
            <a
              className="flex items-center gap-2 text-sm font-semibold leading-6 text-white underline-offset-4 hover:underline"
              href={buildGitHubSourceUrl(repo, source)}
              target="_blank"
              rel="noreferrer"
            >
              <Link2 className="h-4 w-4 shrink-0" strokeWidth={1.9} />
              <span>{source.file_path}</span>
            </a>
          </div>
          <div className="mt-4 flex items-start gap-2 text-sm text-zinc-300">
            <Wrench className="mt-0.5 h-4 w-4 shrink-0" strokeWidth={1.9} />
            <span>Function: {source.symbol_name || "Unknown symbol"}</span>
          </div>
          <div className="mt-2 flex items-start gap-2 text-sm text-zinc-300">
            <MapPinned className="mt-0.5 h-4 w-4 shrink-0" strokeWidth={1.9} />
            <span>
              Lines: {source.line_start}–{source.line_end}
            </span>
          </div>
          {/* {source.signature && <p className="mt-4 text-xs leading-6 text-zinc-500">{source.signature}</p>} */}
          <p className="mt-auto pt-4 text-right text-[11px] leading-5 text-zinc-400">
            Confidence: {formatConfidencePercent(source)}
          </p>
        </article>
      ))}
    </div>
  );
}

function SpinnerOnly() {
  return <span className="h-4 w-4 animate-spin rounded-full border-2 border-current border-t-transparent" />;
}

function formatRepoError(errorMessage) {
  if (!errorMessage) {
    return "Indexing failed. Try another repository URL.";
  }

  if (errorMessage.includes("repository") && errorMessage.includes("not found")) {
    return "That GitHub repository URL could not be cloned. Check the repo URL and branch, then try again.";
  }

  return errorMessage;
}

function formatConfidencePercent(source) {
  const percent = getConfidencePercentValue(source);
  return `${percent}%`;
}

function getConfidencePercentValue(source) {
  const semantic = source?.semantic_score;
  if (typeof semantic === "number" && semantic > 0) {
    return Math.max(8, Math.min(99, Math.round(semantic * 100)));
  }

  const rerank = source?.rerank_score;
  if (typeof rerank === "number") {
    // Soften raw cross-encoder scores into a demo-friendly percentage range.
    const normalized = 1 / (1 + Math.exp(-(rerank / 4)));
    return Math.max(8, Math.min(99, Math.round(normalized * 100)));
  }

  return 0;
}

const APP_BACKGROUND_STYLE = {
  background:
    "radial-gradient(circle at top left, rgba(255,255,255,0.08), transparent 22%), radial-gradient(circle at bottom right, rgba(0,78,146,0.2), transparent 28%), linear-gradient(to right, #004e92, #000428)",
};

function buildGitHubSourceUrl(repo, source) {
  const baseUrl = repo?.github_url;
  if (!baseUrl) return "#";
  const branch = repo?.branch || "main";
  return `${baseUrl.replace(/\/$/, "")}/blob/${branch}/${source.file_path}#L${source.line_start}-L${source.line_end}`;
}

export default App;
