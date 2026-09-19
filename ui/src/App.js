import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import axios from "axios";
import { PrismLight as SyntaxHighlighter } from "react-syntax-highlighter";
import bash from "react-syntax-highlighter/dist/esm/languages/prism/bash";
import c from "react-syntax-highlighter/dist/esm/languages/prism/c";
import cpp from "react-syntax-highlighter/dist/esm/languages/prism/cpp";
import go from "react-syntax-highlighter/dist/esm/languages/prism/go";
import java from "react-syntax-highlighter/dist/esm/languages/prism/java";
import javascript from "react-syntax-highlighter/dist/esm/languages/prism/javascript";
import jsx from "react-syntax-highlighter/dist/esm/languages/prism/jsx";
import json from "react-syntax-highlighter/dist/esm/languages/prism/json";
import python from "react-syntax-highlighter/dist/esm/languages/prism/python";
import rust from "react-syntax-highlighter/dist/esm/languages/prism/rust";
import tsx from "react-syntax-highlighter/dist/esm/languages/prism/tsx";
import typescript from "react-syntax-highlighter/dist/esm/languages/prism/typescript";
import yaml from "react-syntax-highlighter/dist/esm/languages/prism/yaml";
import { vscDarkPlus } from "react-syntax-highlighter/dist/esm/styles/prism";
import { ArrowRight, ChevronDown, ChevronUp, Code2, Compass, Files, Plus } from "lucide-react";
import { API_URL, getSessionHeaders, getSessionId, resetSessionId } from "./config";

[
  ["bash", bash],
  ["c", c],
  ["cpp", cpp],
  ["go", go],
  ["java", java],
  ["javascript", javascript],
  ["jsx", jsx],
  ["json", json],
  ["python", python],
  ["rust", rust],
  ["tsx", tsx],
  ["typescript", typescript],
  ["yaml", yaml],
].forEach(([name, language]) => SyntaxHighlighter.registerLanguage(name, language));

function App() {
  const [sessionId, setSessionId] = useState(() => getSessionId());
  const activeSessionIdRef = useRef(sessionId);
  const [repoUrl, setRepoUrl] = useState("");
  const [reindex, setReindex] = useState(false);
  const [repos, setRepos] = useState([]);
  const [selectedRepoId, setSelectedRepoId] = useState(null);
  const [messages, setMessages] = useState([]);
  const [question, setQuestion] = useState("");
  const [indexing, setIndexing] = useState(false);
  const [loadingRepos, setLoadingRepos] = useState(true);
  const [asking, setAsking] = useState(false);
  const [formError, setFormError] = useState("");
  const [stage, setStage] = useState("landing");
  const sessionHeaders = useMemo(() => getSessionHeaders(sessionId), [sessionId]);

  const selectedRepo = useMemo(
    () => repos.find((repo) => repo.id === selectedRepoId) || null,
    [repos, selectedRepoId],
  );

  const refreshRepos = useCallback(async (preserveSelection = true) => {
    const requestSessionId = sessionId;
    setLoadingRepos(true);
    try {
      const { data } = await axios.get(`${API_URL}/api/repos`, {
        headers: sessionHeaders,
      });
      if (activeSessionIdRef.current !== requestSessionId) return;
      setRepos(data);

      setSelectedRepoId((currentRepoId) => {
        if (data.length === 0) return null;
        if (!preserveSelection || !data.some((repo) => repo.id === currentRepoId)) {
          return data[0].id;
        }
        return currentRepoId;
      });
    } catch {
      if (activeSessionIdRef.current !== requestSessionId) return;
      setFormError("Unable to load repositories right now.");
    } finally {
      if (activeSessionIdRef.current === requestSessionId) {
        setLoadingRepos(false);
      }
    }
  }, [sessionHeaders, sessionId]);

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

    try {
      const { data } = await axios.post(
        `${API_URL}/api/repos/index`,
        { github_url: value, reindex },
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
    } catch (error) {
      setFormError(error?.response?.data?.detail || "Query failed.");
    } finally {
      setAsking(false);
    }
  };

  const endSession = () => {
    const endingSessionId = sessionId;

    setRepos([]);
    setSelectedRepoId(null);
    setMessages([]);
    setQuestion("");
    setRepoUrl("");
    setReindex(false);
    setFormError("");
    setIndexing(false);
    setStage("landing");
    const nextSessionId = resetSessionId();
    activeSessionIdRef.current = nextSessionId;
    setSessionId(nextSessionId);

    axios
      .post(`${API_URL}/api/session/end?session_id=${encodeURIComponent(endingSessionId)}`)
      .catch(() => null);
  };

  if (stage === "landing") {
    return (
      <LandingScreen
        formError={formError}
        indexing={indexing}
        loadingRepos={loadingRepos}
        repoUrl={repoUrl}
        reindex={reindex}
        selectedRepo={selectedRepo}
        setReindex={setReindex}
        setRepoUrl={setRepoUrl}
        startIndexing={startIndexing}
      />
    );
  }

  return (
    <WorkspaceScreen
      asking={asking}
      endSession={endSession}
      formError={formError}
      messages={messages}
      question={question}
      selectedRepo={selectedRepo}
      sendQuestion={sendQuestion}
      setQuestion={setQuestion}
    />
  );
}

function LandingScreen({
  formError,
  indexing,
  loadingRepos,
  repoUrl,
  reindex,
  selectedRepo,
  setReindex,
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

            <div className="mt-3 flex items-center gap-3 px-1">
              <div className="min-w-0 text-left">
                <p className="text-xs font-medium text-zinc-400">Always re-embed</p>
              </div>
              <button
                type="button"
                role="switch"
                aria-checked={reindex}
                aria-label="Always re-embed"
                onClick={() => setReindex((current) => !current)}
                disabled={indexing}
                className={[
                  "relative h-5 w-9 shrink-0 rounded-full border transition focus:outline-none focus-visible:ring-2 focus-visible:ring-white/60 disabled:cursor-wait disabled:opacity-50",
                  reindex
                    ? "border-white bg-white"
                    : "border-white/20 bg-white/10 hover:bg-white/15",
                ].join(" ")}
              >
                <span
                  className={[
                    "absolute left-0 top-0.5 h-3.5 w-3.5 rounded-full transition-transform",
                    reindex
                      ? "translate-x-[18px] bg-black"
                      : "translate-x-0.5 bg-zinc-400",
                  ].join(" ")}
                />
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
  asking,
  endSession,
  formError,
  messages,
  question,
  selectedRepo,
  sendQuestion,
  setQuestion,
}) {
  const messagesContainerRef = useRef(null);

  useEffect(() => {
    const container = messagesContainerRef.current;
    if (!container) return undefined;

    const frame = window.requestAnimationFrame(() => {
      container.scrollTo({ top: container.scrollHeight, behavior: "smooth" });
    });
    return () => window.cancelAnimationFrame(frame);
  }, [messages]);

  return (
    <div
      className="h-screen overflow-hidden text-white"
      style={APP_BACKGROUND_STYLE}
    >
      <div className="grid h-screen grid-cols-1 overflow-hidden bg-transparent">
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
            </div>
          </div>

          <div
            ref={messagesContainerRef}
            className="scrollbar-thin flex flex-1 flex-col gap-4 overflow-auto px-5 py-6 md:px-8"
          >
            {messages.length === 0 ? (
              <div className="flex flex-1 items-center justify-center rounded-[28px] border border-dashed border-white/10 bg-white/[0.02] p-8 text-center text-zinc-500">
                Ask a question
              </div>
            ) : (
              messages.map((message) => (
                <article
                  key={message.id}
                  className={[
                    "rounded-[28px] border px-5 py-4",
                    message.role === "user"
                      ? "ml-auto max-w-[60%] border-white/25 bg-white/[0.16] text-white backdrop-blur-xl"
                      : "w-full border-white/10 bg-white/[0.03] text-white",
                  ].join(" ")}
                >
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

      </div>
    </div>
  );
}

function AnswerBlock({ answer }) {
  const snippets = answer.implementation_snippets || [];
  const relatedFiles = answer.related_files || [];
  const isArchitecture = answer.answer_mode === "architecture";
  const answerParts = splitArchitectureAnswer(answer.direct_answer || answer.answer || "");

  return (
    <div className="space-y-7 text-sm leading-7 text-zinc-100">
      <AnswerSection eyebrow={isArchitecture ? "Architecture walkthrough" : "Implementation"}>
        <MarkdownAnswer
          value={isArchitecture ? answerParts.walkthrough : answer.direct_answer || answer.answer}
        />
      </AnswerSection>

      {snippets.length > 0 ? (
        <AnswerSection eyebrow="Relevant implementation">
          <div className="space-y-4">
            {snippets.map((snippet, index) => (
              <ImplementationSnippet
                key={`${snippet.file_path}-${snippet.line_start}-${index}`}
                snippet={snippet}
              />
            ))}
          </div>
        </AnswerSection>
      ) : null}

      {isArchitecture && answerParts.rationale ? (
        <AnswerSection eyebrow="Why this design">
          <MarkdownAnswer value={answerParts.rationale} />
        </AnswerSection>
      ) : null}

      {relatedFiles.length > 0 ? (
        <AnswerSection eyebrow="Files involved">
          <div className="overflow-hidden rounded-2xl border border-white/10 bg-black/20">
            {relatedFiles.map((file, index) => (
              <div
                key={`${file.file_path}-${index}`}
                className="flex gap-3 border-b border-white/10 px-4 py-3 last:border-b-0"
              >
                <Files
                  className="mt-1 h-4 w-4 shrink-0 text-zinc-500"
                  strokeWidth={1.8}
                />
                <div className="min-w-0">
                  <div className="flex items-start gap-2">
                    {file.source ? (
                      <span className="shrink-0 font-mono text-[11px] font-semibold text-sky-300">
                        [{file.source}]
                      </span>
                    ) : null}
                    <p className="break-all font-mono text-xs font-medium leading-6 text-zinc-100">
                      {file.file_path}
                    </p>
                  </div>
                  <p className="text-xs leading-5 text-zinc-400">{file.description}</p>
                </div>
              </div>
            ))}
          </div>
        </AnswerSection>
      ) : null}
    </div>
  );
}

function splitArchitectureAnswer(value) {
  const match = /^###\s+Why this design\s*$/im.exec(value);
  if (!match) {
    return { walkthrough: value, rationale: "" };
  }

  return {
    walkthrough: value.slice(0, match.index).trim(),
    rationale: value.slice(match.index + match[0].length).trim(),
  };
}

function AnswerSection({ eyebrow, children }) {
  return (
    <section>
      <h3 className="mb-3 text-[11px] font-semibold uppercase tracking-[0.22em] text-zinc-500">
        {eyebrow}
      </h3>
      {children}
    </section>
  );
}

function ImplementationSnippet({ snippet }) {
  const [expanded, setExpanded] = useState(false);
  const code = expanded ? snippet.expanded_code : snippet.code;
  const lineStart = expanded ? snippet.expanded_line_start : snippet.line_start;
  const lineEnd = expanded ? snippet.expanded_line_end : snippet.line_end;
  const annotations = snippet.annotations || [];
  const annotatedLines = new Set(annotations.map((annotation) => annotation.line));

  return (
    <div className="overflow-hidden rounded-2xl border border-white/10 bg-[#09090b]">
      <div className="flex flex-wrap items-center justify-between gap-3 border-b border-white/10 px-4 py-3">
        <div className="flex min-w-0 items-center gap-2.5">
          <Code2 className="h-4 w-4 shrink-0 text-zinc-500" strokeWidth={1.9} />
          {snippet.source ? (
            <span className="shrink-0 font-mono text-[11px] font-semibold text-sky-300">
              [{snippet.source}]
            </span>
          ) : null}
          <div className="min-w-0">
            <p className="truncate font-mono text-xs font-medium text-zinc-100">
              {snippet.file_path}
            </p>
            <p className="mt-0.5 truncate text-[11px] text-zinc-500">
              {snippet.symbol_name || "Relevant code"} · lines {lineStart}–{lineEnd}
            </p>
          </div>
        </div>
        {snippet.expandable ? (
          <button
            className="inline-flex items-center gap-1.5 rounded-full border border-white/10 px-2.5 py-1 text-[11px] font-medium text-zinc-400 transition hover:border-white/20 hover:text-white"
            onClick={() => setExpanded((value) => !value)}
            type="button"
          >
            {expanded ? (
              <ChevronUp className="h-3.5 w-3.5" strokeWidth={2} />
            ) : (
              <ChevronDown className="h-3.5 w-3.5" strokeWidth={2} />
            )}
            {expanded ? "Collapse" : "Expand"}
          </button>
        ) : null}
      </div>
      <div className="scrollbar-thin overflow-x-auto">
        <SyntaxHighlighter
          language={normalizeSyntaxLanguage(snippet.language)}
          style={vscDarkPlus}
          showLineNumbers
          startingLineNumber={lineStart}
          wrapLines
          wrapLongLines={false}
          customStyle={{
            background: "transparent",
            margin: 0,
            padding: "1rem",
            fontSize: "12px",
            lineHeight: "1.65",
          }}
          lineNumberStyle={{ color: "#52525b", minWidth: "2.75em" }}
          lineProps={(lineNumber) => ({
            style: (
              annotatedLines.has(lineNumber)
              || annotatedLines.has(lineStart + lineNumber - 1)
            )
              ? {
                  background: "rgba(56, 189, 248, 0.08)",
                  borderLeft: "2px solid rgba(56, 189, 248, 0.55)",
                  display: "block",
                }
              : { display: "block" },
          })}
        >
          {code || ""}
        </SyntaxHighlighter>
      </div>
      {annotations.length > 0 ? (
        <div className="space-y-2 border-t border-white/10 bg-sky-400/[0.03] px-4 py-3">
          {annotations.map((annotation) => (
            <div
              className="flex items-start gap-2 text-xs leading-5 text-zinc-300"
              key={`${annotation.line}-${annotation.label}`}
            >
              <span className="mt-0.5 text-sky-300">▲</span>
              <span>
                <span className="mr-2 font-mono text-sky-300">L{annotation.line}</span>
                {annotation.label}
              </span>
            </div>
          ))}
        </div>
      ) : null}
    </div>
  );
}

function normalizeSyntaxLanguage(language) {
  const aliases = {
    js: "javascript",
    jsx: "jsx",
    py: "python",
    rs: "rust",
    ts: "typescript",
    tsx: "tsx",
  };
  return aliases[language] || language || "text";
}

function MarkdownAnswer({ value }) {
  const lines = String(value || "").replace(/\r\n/g, "\n").split("\n");
  const elements = [];
  let paragraph = [];
  let listItems = [];
  let listType = null;
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
    const ListTag = listType === "ordered" ? "ol" : "ul";
    elements.push(
      <ListTag
        key={`list-${elements.length}`}
        className={`${listType === "ordered" ? "list-decimal" : "list-disc"} space-y-2 pl-5 text-sm leading-7 text-zinc-100`}
      >
        {listItems.map((item, index) => (
          <li key={`li-${index}`}>{renderInlineMarkdown(item)}</li>
        ))}
      </ListTag>,
    );
    listItems = [];
    listType = null;
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

    const flowSteps = parseFlowSteps(trimmed);
    if (flowSteps) {
      flushParagraph();
      flushList();
      elements.push(
        <ExecutionFlow key={`flow-${elements.length}`} steps={flowSteps} />,
      );
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

    const listMatch = trimmed.match(/^([-*]|\d+\.)\s+(.*)$/);
    if (listMatch) {
      flushParagraph();
      const nextListType = /\d+\./.test(listMatch[1]) ? "ordered" : "unordered";
      if (listType && listType !== nextListType) {
        flushList();
      }
      listType = nextListType;
      listItems.push(listMatch[2]);
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

function parseFlowSteps(value) {
  if (!value.includes("→") && !value.includes("->")) return null;
  const steps = value
    .split(/\s*(?:→|->)\s*/)
    .map((step) => step.replace(/`|\*\*|\[\d+\]/g, "").trim())
    .filter(Boolean);
  return steps.length >= 3 ? steps : null;
}

function ExecutionFlow({ steps }) {
  return (
    <div className="rounded-2xl border border-sky-300/15 bg-sky-400/[0.035] px-4 py-5">
      <div className="flex flex-col items-center" role="list" aria-label="Execution flow">
        {steps.map((step, index) => (
          <div className="flex w-full flex-col items-center" key={`${step}-${index}`} role="listitem">
            <div className="w-full max-w-xl rounded-xl border border-white/10 bg-black/20 px-4 py-2.5 text-center font-mono text-xs font-medium text-zinc-100">
              {step}
            </div>
            {index < steps.length - 1 ? (
              <div className="flex h-10 flex-col items-center justify-center text-sky-300/70">
                <span className="h-5 w-px bg-sky-300/30" />
                <ChevronDown className="-mt-0.5 h-4 w-4" strokeWidth={2} />
              </div>
            ) : null}
          </div>
        ))}
      </div>
    </div>
  );
}

function renderInlineMarkdown(text) {
  const nodes = [];
  const pattern = /(`[^`]+`|\*\*[^*]+\*\*|\[[^\]]+\]\([^)]+\)|\[\d+\])/g;
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
    } else if (/^\[\d+\]$/.test(token)) {
      nodes.push(
        <span
          key={`citation-${match.index}`}
          className="ml-0.5 font-mono text-[0.85em] font-semibold text-sky-300"
        >
          {token}
        </span>,
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

const APP_BACKGROUND_STYLE = {
  background:
    "radial-gradient(circle at top left, rgba(255,255,255,0.08), transparent 22%), radial-gradient(circle at bottom right, rgba(0,78,146,0.2), transparent 28%), linear-gradient(to right, #004e92, #000428)",
};

export default App;
