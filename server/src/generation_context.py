import ast
import re
import textwrap
from dataclasses import dataclass
from typing import List, Optional


OMISSION_TEMPLATE = "... [omitted lines {start}-{end}] ..."
CODE_TERMINAL_PATTERN = re.compile(
    r"^\s*(?:return|raise|yield|throw|except|catch)\b"
)
QUERY_STOPWORDS = {
    "about",
    "after",
    "against",
    "also",
    "and",
    "are",
    "before",
    "does",
    "from",
    "how",
    "into",
    "its",
    "that",
    "the",
    "their",
    "this",
    "through",
    "what",
    "when",
    "where",
    "which",
    "with",
}


@dataclass(frozen=True)
class ContextSpan:
    start: int
    end: int
    reason: str
    score: float
    required: bool = False
    anchor: Optional[int] = None


@dataclass
class SelectedSourceContext:
    source_index: int
    text: str
    line_ranges: List[tuple[int, int]]
    budget: int
    used_chars: int
    complete: bool
    reasons: List[str]


@dataclass
class GenerationContext:
    blocks: List[str]
    joined_context: str
    selected_sources: List[SelectedSourceContext]
    total_budget: int
    used_chars: int

    def diagnostics(self) -> dict:
        return {
            "total_content_budget": self.total_budget,
            "used_content_chars": self.used_chars,
            "sources": [
                {
                    "source": selected.source_index,
                    "budget": selected.budget,
                    "used_chars": selected.used_chars,
                    "complete": selected.complete,
                    "line_ranges": selected.line_ranges,
                    "reasons": selected.reasons,
                }
                for selected in self.selected_sources
            ],
        }


class GenerationContextBuilder:
    """Build bounded, source-aware context after retrieval has finished."""

    def __init__(
        self,
        per_source_budget: int = 1500,
        max_source_budget: int = 3600,
    ):
        self.per_source_budget = max(1, int(per_source_budget))
        self.max_source_budget = max(self.per_source_budget, int(max_source_budget))

    def build(
        self,
        sources: List[dict],
        question: str,
        answer_mode: str,
    ) -> GenerationContext:
        if not sources:
            return GenerationContext([], "", [], 0, 0)

        total_budget = self.per_source_budget * len(sources)
        budgets = self._allocate_budgets(sources, total_budget, answer_mode)
        selected_sources = [
            self._select_source(
                source,
                source_index=index,
                question=question,
                budget=budgets[index - 1],
            )
            for index, source in enumerate(sources, start=1)
        ]
        blocks = [
            self._render_source_block(source, selected)
            for source, selected in zip(sources, selected_sources)
        ]
        return GenerationContext(
            blocks=blocks,
            joined_context="\n\n".join(blocks),
            selected_sources=selected_sources,
            total_budget=total_budget,
            used_chars=sum(selected.used_chars for selected in selected_sources),
        )

    def build_display_snippet(
        self,
        source: dict,
        question: str = "",
        max_lines: int = 28,
        expanded_max: int = 70,
    ) -> dict:
        lines = str(source.get("content") or "").splitlines()
        if not lines:
            lines = [str(source.get("signature") or source.get("symbol_name") or "")]

        if len(lines) <= max_lines:
            visible_start, visible_end = 0, len(lines)
        else:
            candidates = self._candidate_spans(source, lines, question)
            anchor = self._display_anchor(candidates, source, lines, question)
            visible_start, visible_end = self._leading_context_window(
                anchor,
                len(lines),
                max_lines,
            )

        if len(lines) <= expanded_max:
            expanded_start, expanded_end = 0, len(lines)
        else:
            anchor = visible_start + (visible_end - visible_start) // 2
            expanded_start, expanded_end = self._centered_window(
                anchor,
                len(lines),
                expanded_max,
            )

        source_line_start = int(source.get("line_start") or 1)
        annotations = self._build_annotations(
            lines[visible_start:visible_end],
            source_line_start + visible_start,
        )
        return {
            "file_path": source.get("file_path"),
            "language": source.get("language") or "text",
            "symbol_name": source.get("symbol_name"),
            "line_start": source_line_start + visible_start,
            "line_end": source_line_start + visible_end - 1,
            "code": "\n".join(lines[visible_start:visible_end]),
            "expanded_line_start": source_line_start + expanded_start,
            "expanded_line_end": source_line_start + expanded_end - 1,
            "expanded_code": "\n".join(lines[expanded_start:expanded_end]),
            "expandable": expanded_start < visible_start or expanded_end > visible_end,
            "annotations": annotations,
        }

    @staticmethod
    def _build_annotations(lines: List[str], line_start: int) -> List[dict]:
        candidates = []
        for offset, line in enumerate(lines):
            stripped = line.strip()
            lowered = stripped.lower()
            label = None
            priority = 0

            if re.search(r"\bif\b.*\berrors?\b", lowered):
                label = "The failure path begins when earlier processing has collected errors."
                priority = 100
            elif re.search(r"\braise\b", lowered):
                label = "Control leaves this function here and transfers to the exception-handling path."
                priority = 95
            elif "depend" in lowered and any(
                token in lowered for token in {"solve", "resolve"}
            ):
                label = "Dependencies are resolved here before endpoint execution continues."
                priority = 90
            elif "response" in lowered and re.search(r"\breturn\b", lowered):
                label = "The response object leaves this handler here."
                priority = 85

            if label:
                candidates.append(
                    {
                        "line": line_start + offset,
                        "label": label,
                        "priority": priority,
                    }
                )

        selected = sorted(candidates, key=lambda item: item["priority"], reverse=True)[:3]
        selected.sort(key=lambda item: item["line"])
        return [
            {"line": item["line"], "label": item["label"]}
            for item in selected
        ]

    def _allocate_budgets(
        self,
        sources: List[dict],
        total_budget: int,
        answer_mode: str,
    ) -> List[int]:
        lengths = [len(str(source.get("content") or "").rstrip()) for source in sources]
        budgets = [0] * len(sources)
        long_indexes = []

        base_for_long = {
            "architecture": 950,
            "configuration": 850,
            "implementation": 700,
            "debugging": 700,
        }.get(answer_mode, 700)

        for index, length in enumerate(lengths):
            if length <= self.per_source_budget:
                budgets[index] = length
            else:
                budgets[index] = min(base_for_long, length)
                long_indexes.append(index)

        remaining = max(0, total_budget - sum(budgets))
        caps = [
            min(
                length,
                self.max_source_budget
                if answer_mode != "architecture"
                else min(self.max_source_budget, 2600),
            )
            for length in lengths
        ]

        while remaining > 0:
            eligible = [index for index in long_indexes if budgets[index] < caps[index]]
            if not eligible:
                break
            weights = {
                index: self._source_weight(index, answer_mode) for index in eligible
            }
            weight_total = sum(weights.values())
            progressed = 0
            for index in eligible:
                share = max(1, int(remaining * weights[index] / weight_total))
                addition = min(share, caps[index] - budgets[index], remaining - progressed)
                if addition <= 0:
                    continue
                budgets[index] += addition
                progressed += addition
                if progressed >= remaining:
                    break
            if progressed == 0:
                break
            remaining -= progressed

        return budgets

    @staticmethod
    def _source_weight(index: int, answer_mode: str) -> float:
        if answer_mode == "architecture":
            return 1.0
        if answer_mode == "configuration":
            return max(0.8, 2.0 / (1.0 + 0.35 * index))
        return max(0.65, 3.0 / (1.0 + 0.55 * index))

    def _select_source(
        self,
        source: dict,
        source_index: int,
        question: str,
        budget: int,
    ) -> SelectedSourceContext:
        content = str(source.get("content") or "").rstrip()
        lines = content.splitlines()
        source_line_start = int(source.get("line_start") or 1)
        if not lines:
            lines = [str(source.get("signature") or source.get("symbol_name") or "")]
            content = lines[0]

        if len(content) <= budget:
            line_end = source_line_start + len(lines) - 1
            return SelectedSourceContext(
                source_index=source_index,
                text=content,
                line_ranges=[(source_line_start, line_end)],
                budget=budget,
                used_chars=len(content),
                complete=True,
                reasons=["complete source"],
            )

        candidates = self._candidate_spans(source, lines, question)
        signature = self._compact_signature(source, content)
        selected = self._choose_spans(
            lines,
            candidates,
            source_line_start,
            signature,
            budget,
        )
        rendered, ranges = self._render_ranges(
            lines,
            selected,
            source_line_start,
            signature,
        )
        reasons = list(dict.fromkeys(span.reason for span in selected))
        if signature:
            reasons.insert(0, "signature")
        return SelectedSourceContext(
            source_index=source_index,
            text=rendered,
            line_ranges=ranges,
            budget=budget,
            used_chars=len(rendered),
            complete=False,
            reasons=reasons,
        )

    def _candidate_spans(
        self,
        source: dict,
        lines: List[str],
        question: str,
    ) -> List[ContextSpan]:
        content = "\n".join(lines)
        terms = self._query_terms(
            " ".join(
                [
                    question,
                    str(source.get("symbol_name") or ""),
                    str(source.get("file_path") or ""),
                ]
            )
        )
        candidates = []
        language = str(source.get("language") or "text").lower()
        is_code = language != "text" and not str(source.get("file_path") or "").lower().endswith(
            (".md", ".mdx", ".rst")
        )
        excluded_query_lines = set()

        if is_code and language == "python":
            candidates.extend(self._python_spans(content, lines))
            excluded_query_lines = self._python_non_runtime_prefix(content)
        elif is_code:
            candidates.extend(self._generic_code_spans(lines))
        else:
            candidates.extend(self._document_spans(lines, terms))

        candidates.extend(
            self._query_spans(
                lines,
                terms,
                is_code=is_code,
                excluded_lines=excluded_query_lines,
            )
        )
        if not candidates:
            candidates.append(
                ContextSpan(0, min(len(lines), 8), "source opening", 20, True, 0)
            )
        return self._deduplicate_spans(candidates, len(lines))

    def _python_spans(self, content: str, lines: List[str]) -> List[ContextSpan]:
        try:
            tree = ast.parse(textwrap.dedent(content))
        except SyntaxError:
            return self._generic_code_spans(lines)

        function = next(
            (
                node
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ),
            None,
        )
        if function is None:
            return self._generic_code_spans(lines)

        body = list(function.body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]

        spans = []
        if body:
            first = body[0]
            start = max(0, first.lineno - 1)
            end = min(len(lines), min(first.end_lineno or first.lineno, first.lineno + 9))
            spans.append(
                ContextSpan(start, end, "first executable block", 120, True, start)
            )

        parent = {}
        for node in ast.walk(function):
            for child in ast.iter_child_nodes(node):
                parent[child] = node

        terminal_nodes = [
            node
            for node in ast.walk(function)
            if isinstance(node, (ast.Return, ast.Raise, ast.Yield, ast.YieldFrom))
        ]
        for terminal in terminal_nodes:
            enclosing = terminal
            current = parent.get(terminal)
            while current is not None and current is not function:
                if isinstance(current, (ast.If, ast.Try, ast.ExceptHandler, ast.Match)):
                    size = (current.end_lineno or current.lineno) - current.lineno + 1
                    if size <= 12:
                        enclosing = current
                current = parent.get(current)
            start = max(0, enclosing.lineno - 1)
            end = min(len(lines), enclosing.end_lineno or enclosing.lineno)
            if end - start > 12:
                start = max(0, terminal.lineno - 2)
                end = min(len(lines), (terminal.end_lineno or terminal.lineno) + 2)
            spans.append(
                ContextSpan(
                    start,
                    end,
                    "terminal control flow",
                    105 + terminal.lineno / max(1, len(lines)),
                    False,
                    terminal.lineno - 1,
                )
            )
        return spans

    @staticmethod
    def _python_non_runtime_prefix(content: str) -> set[int]:
        try:
            tree = ast.parse(textwrap.dedent(content))
        except SyntaxError:
            return set()
        function = next(
            (
                node
                for node in ast.walk(tree)
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
            ),
            None,
        )
        if function is None:
            return set()
        body = list(function.body)
        if (
            body
            and isinstance(body[0], ast.Expr)
            and isinstance(getattr(body[0], "value", None), ast.Constant)
            and isinstance(body[0].value.value, str)
        ):
            body = body[1:]
        if not body:
            return set()
        return set(range(max(0, body[0].lineno - 1)))

    @staticmethod
    def _generic_code_spans(lines: List[str]) -> List[ContextSpan]:
        spans = []
        first_executable = None
        in_block_comment = False
        for index, line in enumerate(lines):
            stripped = line.strip()
            if "/*" in stripped:
                in_block_comment = True
            if in_block_comment:
                if "*/" in stripped:
                    in_block_comment = False
                continue
            if not stripped or stripped.startswith(("//", "#", "*", "import ", "from ")):
                continue
            if index == 0 or re.match(
                r"^(?:export\s+)?(?:async\s+)?(?:def|function|class|const|let|var|fn)\b",
                stripped,
            ):
                continue
            first_executable = index
            break
        if first_executable is not None:
            spans.append(
                ContextSpan(
                    first_executable,
                    min(len(lines), first_executable + 8),
                    "first executable block",
                    120,
                    True,
                    first_executable,
                )
            )

        for index, line in enumerate(lines):
            if CODE_TERMINAL_PATTERN.search(line):
                spans.append(
                    ContextSpan(
                        max(0, index - 2),
                        min(len(lines), index + 3),
                        "terminal control flow",
                        105 + index / max(1, len(lines)),
                        False,
                        index,
                    )
                )
        return spans

    @staticmethod
    def _document_spans(lines: List[str], terms: set[str]) -> List[ContextSpan]:
        spans = [ContextSpan(0, min(len(lines), 8), "document opening", 25, True, 0)]
        headings = [
            index
            for index, line in enumerate(lines)
            if re.match(r"^\s*(?:#{1,6}\s+|[A-Z][A-Za-z0-9 _-]+:\s*$)", line)
        ]
        for position, start in enumerate(headings):
            end = headings[position + 1] if position + 1 < len(headings) else len(lines)
            end = min(end, start + 16)
            text = " ".join(lines[start:end]).lower()
            hits = sum(term in text for term in terms)
            if hits:
                spans.append(
                    ContextSpan(
                        start,
                        end,
                        "matching document section",
                        60 + 10 * hits,
                        False,
                        start,
                    )
                )
        return spans

    @staticmethod
    def _query_spans(
        lines: List[str],
        terms: set[str],
        is_code: bool,
        excluded_lines: Optional[set[int]] = None,
    ) -> List[ContextSpan]:
        spans = []
        excluded_lines = excluded_lines or set()
        for index, line in enumerate(lines):
            if index in excluded_lines:
                continue
            lowered = line.lower()
            hits = sum(term in lowered for term in terms)
            if not hits:
                continue
            stripped = line.strip()
            if is_code and stripped.startswith(("#", "//", "*")):
                score = 35 + hits
            else:
                score = 80 + 8 * hits
            start = max(0, index - 3)
            while start < index and start in excluded_lines:
                start += 1
            spans.append(
                ContextSpan(
                    start,
                    min(len(lines), index + 4),
                    "query-relevant code" if is_code else "query-relevant text",
                    score,
                    False,
                    index,
                )
            )
        return spans

    @staticmethod
    def _deduplicate_spans(
        spans: List[ContextSpan],
        line_count: int,
    ) -> List[ContextSpan]:
        deduplicated = {}
        for span in spans:
            start = max(0, min(span.start, line_count - 1))
            end = max(start + 1, min(span.end, line_count))
            normalized = ContextSpan(
                start,
                end,
                span.reason,
                span.score,
                span.required,
                span.anchor if span.anchor is not None else start,
            )
            key = (start, end, span.reason)
            existing = deduplicated.get(key)
            if existing is None or normalized.score > existing.score:
                deduplicated[key] = normalized
        return list(deduplicated.values())

    def _choose_spans(
        self,
        lines: List[str],
        candidates: List[ContextSpan],
        source_line_start: int,
        signature: str,
        budget: int,
    ) -> List[ContextSpan]:
        required = [span for span in candidates if span.required]
        query = sorted(
            (span for span in candidates if span.reason.startswith("query-relevant")),
            key=lambda span: span.score,
            reverse=True,
        )[:1]
        terminal = sorted(
            (span for span in candidates if span.reason == "terminal control flow"),
            key=lambda span: (span.anchor or 0, span.score),
            reverse=True,
        )[:2]
        optional = sorted(candidates, key=lambda span: span.score, reverse=True)
        ordered = required + query + terminal + optional

        selected = []
        seen = set()
        for candidate in ordered:
            key = (candidate.start, candidate.end, candidate.reason)
            if key in seen:
                continue
            seen.add(key)
            proposal = self._merge_spans(selected + [candidate])
            rendered, _ = self._render_ranges(
                lines,
                proposal,
                source_line_start,
                signature,
            )
            if len(rendered) <= budget:
                selected = proposal
                continue

            anchor = candidate.anchor if candidate.anchor is not None else candidate.start
            compact = ContextSpan(
                max(candidate.start, anchor - 1),
                min(candidate.end, anchor + 2),
                candidate.reason,
                candidate.score,
                candidate.required,
                anchor,
            )
            proposal = self._merge_spans(selected + [compact])
            rendered, _ = self._render_ranges(
                lines,
                proposal,
                source_line_start,
                signature,
            )
            if len(rendered) <= budget:
                selected = proposal

        if not selected:
            fallback = ContextSpan(0, 1, "source opening", 1, True, 0)
            rendered, _ = self._render_ranges(
                lines,
                [fallback],
                source_line_start,
                signature,
            )
            if len(rendered) <= budget:
                selected = [fallback]
        return selected

    @staticmethod
    def _merge_spans(spans: List[ContextSpan]) -> List[ContextSpan]:
        if not spans:
            return []
        ordered = sorted(spans, key=lambda span: (span.start, span.end))
        merged = [ordered[0]]
        for span in ordered[1:]:
            previous = merged[-1]
            if span.start <= previous.end + 1:
                reasons = list(dict.fromkeys([previous.reason, span.reason]))
                merged[-1] = ContextSpan(
                    previous.start,
                    max(previous.end, span.end),
                    "+".join(reasons),
                    max(previous.score, span.score),
                    previous.required or span.required,
                    previous.anchor if previous.score >= span.score else span.anchor,
                )
            else:
                merged.append(span)
        return merged

    @staticmethod
    def _render_ranges(
        lines: List[str],
        spans: List[ContextSpan],
        source_line_start: int,
        signature: str,
    ) -> tuple[str, List[tuple[int, int]]]:
        parts = []
        ranges = []
        cursor = 0
        if signature:
            parts.append(f"Signature (line {source_line_start}): {signature}")
            ranges.append((source_line_start, source_line_start))
            cursor = 1

        for span in sorted(spans, key=lambda item: item.start):
            if span.start > cursor:
                parts.append(
                    OMISSION_TEMPLATE.format(
                        start=source_line_start + cursor,
                        end=source_line_start + span.start - 1,
                    )
                )
            parts.append("\n".join(lines[span.start : span.end]))
            ranges.append(
                (
                    source_line_start + span.start,
                    source_line_start + span.end - 1,
                )
            )
            cursor = max(cursor, span.end)

        if cursor < len(lines):
            parts.append(
                OMISSION_TEMPLATE.format(
                    start=source_line_start + cursor,
                    end=source_line_start + len(lines) - 1,
                )
            )
        return (
            "\n".join(part for part in parts if part),
            GenerationContextBuilder._merge_line_ranges(ranges),
        )

    @staticmethod
    def _merge_line_ranges(ranges: List[tuple[int, int]]) -> List[tuple[int, int]]:
        if not ranges:
            return []
        merged = [ranges[0]]
        for start, end in sorted(ranges[1:]):
            previous_start, previous_end = merged[-1]
            if start <= previous_end + 1:
                merged[-1] = (previous_start, max(previous_end, end))
            else:
                merged.append((start, end))
        return merged

    @staticmethod
    def _compact_signature(source: dict, content: str) -> str:
        language = str(source.get("language") or "").lower()
        if language == "python":
            try:
                tree = ast.parse(textwrap.dedent(content))
                function = next(
                    (
                        node
                        for node in ast.walk(tree)
                        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef))
                    ),
                    None,
                )
                if function is not None:
                    args = []
                    positional = list(function.args.posonlyargs) + list(function.args.args)
                    positional_defaults = [None] * (
                        len(positional) - len(function.args.defaults)
                    ) + list(function.args.defaults)
                    for argument, default in zip(positional, positional_defaults):
                        rendered = argument.arg
                        if default is not None:
                            default_text = ast.unparse(default)
                            rendered += f"={default_text}" if len(default_text) <= 30 else "=..."
                        args.append(rendered)
                    if function.args.vararg:
                        args.append(f"*{function.args.vararg.arg}")
                    elif function.args.kwonlyargs:
                        args.append("*")
                    for argument, default in zip(
                        function.args.kwonlyargs,
                        function.args.kw_defaults,
                    ):
                        rendered = argument.arg
                        if default is not None:
                            default_text = ast.unparse(default)
                            rendered += f"={default_text}" if len(default_text) <= 30 else "=..."
                        args.append(rendered)
                    if function.args.kwarg:
                        args.append(f"**{function.args.kwarg.arg}")
                    prefix = "async def" if isinstance(function, ast.AsyncFunctionDef) else "def"
                    returns = ""
                    if function.returns is not None:
                        return_text = ast.unparse(function.returns)
                        if len(return_text) <= 80:
                            returns = f" -> {return_text}"
                    return f"{prefix} {function.name}({', '.join(args)}){returns}:"
            except (SyntaxError, ValueError):
                pass

        signature = str(source.get("signature") or "").strip()
        if signature and len(signature) <= 240:
            return signature
        symbol = str(source.get("symbol_name") or "symbol")
        return f"{symbol} (…)"

    @staticmethod
    def _query_terms(text: str) -> set[str]:
        return {
            term
            for term in re.findall(r"[a-zA-Z_][a-zA-Z0-9_]{2,}", (text or "").lower())
            if term not in QUERY_STOPWORDS
        }

    @staticmethod
    def _display_anchor(
        candidates: List[ContextSpan],
        source: dict,
        lines: List[str],
        question: str,
    ) -> int:
        terms = GenerationContextBuilder._query_terms(
            f"{question} {source.get('symbol_name', '')}"
        )
        scored = []
        for span in candidates:
            anchor = span.anchor if span.anchor is not None else span.start
            line = lines[min(max(anchor, 0), len(lines) - 1)].lower()
            term_hits = sum(term in line for term in terms)
            reason_bonus = {
                "query-relevant code": 60,
                "first executable block": 30,
                "terminal control flow": 25,
            }.get(span.reason, 0)
            scored.append((term_hits * 20 + reason_bonus + span.score, anchor))
        if not scored:
            return 0
        return max(scored)[1]

    @staticmethod
    def _centered_window(anchor: int, line_count: int, window_size: int) -> tuple[int, int]:
        count = min(max(1, window_size), line_count)
        start = max(0, anchor - count // 2)
        start = min(start, max(0, line_count - count))
        return start, start + count

    @staticmethod
    def _leading_context_window(
        anchor: int,
        line_count: int,
        window_size: int,
    ) -> tuple[int, int]:
        count = min(max(1, window_size), line_count)
        start = max(0, anchor - min(10, count - 1))
        end = min(line_count, start + count)
        start = max(0, end - count)
        return start, end

    @staticmethod
    def _render_source_block(source: dict, selected: SelectedSourceContext) -> str:
        shown = ", ".join(
            f"{start}-{end}" if start != end else str(start)
            for start, end in selected.line_ranges
        ) or "none"
        return "\n".join(
            [
                f"[Source {selected.source_index}]",
                f"File: {source.get('file_path', '')}",
                f"Symbol: {source.get('symbol_name', '')}",
                f"Lines: {source.get('line_start', '')}-{source.get('line_end', '')}",
                f"Lines shown: {shown}",
                selected.text,
            ]
        )
