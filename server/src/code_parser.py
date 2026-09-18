import re
import time
from pathlib import Path
from typing import Dict, List, Optional

from tree_sitter_languages import get_parser


LANGUAGE_BY_EXTENSION = {
    ".py": "python",
    ".js": "javascript",
    ".jsx": "javascript",
    ".ts": "typescript",
    ".tsx": "tsx",
    ".java": "java",
    ".go": "go",
    ".rs": "rust",
    ".c": "c",
    ".h": "c",
    ".cc": "cpp",
    ".cpp": "cpp",
    ".cxx": "cpp",
    ".hpp": "cpp",
    ".hh": "cpp",
}

# Node types that make a good standalone retrieval unit (function/method-sized).
SYMBOL_NODE_TYPES = {
    "python": {"function_definition", "class_definition"},
    "javascript": {
        "function_declaration",
        "class_declaration",
        "method_definition",
        "generator_function_declaration",
        "lexical_declaration",
        "variable_declaration",
    },
    "typescript": {
        "function_declaration",
        "class_declaration",
        "method_definition",
        "interface_declaration",
        "type_alias_declaration",
        "lexical_declaration",
        "variable_statement",
    },
    "tsx": {
        "function_declaration",
        "class_declaration",
        "method_definition",
        "interface_declaration",
        "type_alias_declaration",
        "lexical_declaration",
        "variable_statement",
    },
    "java": {
        "class_declaration",
        "method_declaration",
        "interface_declaration",
        "enum_declaration",
    },
    "go": {
        "function_declaration",
        "method_declaration",
        "type_declaration",
    },
    "rust": {
        "function_item",
        "impl_item",
        "struct_item",
        "enum_item",
        "trait_item",
    },
    "c": {
        "function_definition",
        "struct_specifier",
        "enum_specifier",
        "type_definition",
    },
    "cpp": {
        "function_definition",
        "class_specifier",
        "struct_specifier",
        "enum_specifier",
        "namespace_definition",
        "type_definition",
    },
}

# Node types that are "containers": they hold nested members (methods, fields)
# that are themselves worth indexing as their own chunks. For these, we emit a
# compact *overview* chunk (signature + docstring + head-of-body + member
# list) instead of dumping the entire body into one chunk. Without this, a
# large class becomes a single multi-hundred-line chunk whose embedding is a
# blurry average of everything inside it (hurts hit rate for method-specific
# queries) and whose content is silently cut off by the ~1500 char preview
# used when the chunk is fed to the LLM (hurts faithfulness, since the method
# the user actually asked about may fall outside the truncated window).
CONTAINER_NODE_TYPES = {
    "python": {"class_definition"},
    "javascript": {"class_declaration"},
    "typescript": {"class_declaration"},
    "tsx": {"class_declaration"},
    "java": {"class_declaration", "interface_declaration", "enum_declaration"},
    "rust": {"impl_item", "trait_item"},
    "cpp": {"class_specifier", "namespace_definition"},
}

IDENTIFIER_TYPES = {
    "identifier",
    "property_identifier",
    "type_identifier",
    "field_identifier",
}

MAX_OVERVIEW_BODY_LINES = 40
MAX_OVERVIEW_CHARS = 1200
MAX_MEMBERS_LISTED = 25


class CodeParser:
    def __init__(self):
        self.parsers = {}

    def detect_language(self, file_path: str) -> str:
        return LANGUAGE_BY_EXTENSION.get(Path(file_path).suffix.lower(), "text")

    def _get_parser(self, language: str):
        if language == "text":
            return None
        if language not in self.parsers:
            self.parsers[language] = get_parser(language)
        return self.parsers[language]

    def chunk_file(
        self,
        file_path: str,
        repo_root: str,
        profile: Optional[dict] = None,
    ) -> List[Dict]:
        read_started_at = time.perf_counter()
        language = self.detect_language(file_path)
        source = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        source_bytes = source.encode("utf-8")
        relative_path = str(Path(file_path).resolve().relative_to(Path(repo_root).resolve()))
        read_seconds = time.perf_counter() - read_started_at

        if not source.strip():
            self._update_profile(profile, read_seconds, 0.0, 0.0, 0)
            return []

        parser = self._get_parser(language)
        if parser is None:
            chunk_started_at = time.perf_counter()
            chunks = self._fallback_chunks(source, relative_path, language)
            self._update_profile(
                profile,
                read_seconds,
                0.0,
                time.perf_counter() - chunk_started_at,
                len(chunks),
            )
            return chunks

        parse_started_at = time.perf_counter()
        tree = parser.parse(source_bytes)
        parse_seconds = time.perf_counter() - parse_started_at
        chunk_started_at = time.perf_counter()
        lines = source.splitlines()
        chunks = []
        capture_types = SYMBOL_NODE_TYPES.get(language, set())
        container_types = CONTAINER_NODE_TYPES.get(language, set())

        def visit(node, context_name: Optional[str] = None):
            if node.type in capture_types:
                if node.type in container_types:
                    overview = self._build_container_overview(
                        node, source_bytes, lines, relative_path, language, context_name, capture_types
                    )
                    if overview:
                        chunks.append(overview)

                    own_name = self._extract_own_name(node, source_bytes)
                    nested_context = own_name
                    if context_name and own_name:
                        nested_context = f"{context_name}.{own_name}"
                    elif context_name and not own_name:
                        nested_context = context_name

                    for child in node.children:
                        visit(child, nested_context)
                    return

                chunk = self._build_chunk(node, source_bytes, lines, relative_path, language, context_name)
                if chunk:
                    chunks.append(chunk)
                    return
            for child in node.children:
                visit(child, context_name)

        visit(tree.root_node)

        if not chunks:
            chunks = self._fallback_chunks(source, relative_path, language)

        file_overview = self._build_file_overview(
            tree.root_node, source_bytes, lines, relative_path, language, chunks
        )
        if file_overview:
            chunks.insert(0, file_overview)

        self._update_profile(
            profile,
            read_seconds,
            parse_seconds,
            time.perf_counter() - chunk_started_at,
            len(chunks),
        )
        return chunks

    @staticmethod
    def _update_profile(
        profile: Optional[dict],
        read_seconds: float,
        parse_seconds: float,
        chunk_seconds: float,
        chunk_count: int,
    ) -> None:
        if profile is None:
            return
        profile.update(
            read_seconds=read_seconds,
            parse_seconds=parse_seconds,
            chunk_seconds=chunk_seconds,
            chunk_count=chunk_count,
        )

    def _build_chunk(
        self,
        node,
        source_bytes: bytes,
        lines: List[str],
        relative_path: str,
        language: str,
        context_name: Optional[str] = None,
    ) -> Optional[Dict]:
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        snippet = "\n".join(lines[start_line - 1 : end_line]).strip()
        if len(snippet.splitlines()) < 2:
            return None

        own_name = self._extract_own_name(node, source_bytes)
        qualified_name = own_name
        if context_name and own_name:
            qualified_name = f"{context_name}.{own_name}"
        elif context_name and not own_name:
            qualified_name = context_name

        signature = lines[start_line - 1].strip() if start_line - 1 < len(lines) else ""
        searchable_text = "\n".join(
            part
            for part in [relative_path, context_name or "", qualified_name or "", signature, snippet]
            if part
        )

        metadata = {"parser": "tree-sitter"}
        if context_name:
            metadata["parent"] = context_name

        return {
            "file_path": relative_path,
            "language": language,
            "symbol_name": qualified_name or relative_path.split("/")[-1],
            "symbol_type": node.type,
            "line_start": start_line,
            "line_end": end_line,
            "signature": signature,
            "content": snippet,
            "searchable_text": searchable_text,
            "metadata_json": metadata,
        }

    def _build_container_overview(
        self,
        node,
        source_bytes: bytes,
        lines: List[str],
        relative_path: str,
        language: str,
        context_name: Optional[str],
        capture_types: set,
    ) -> Optional[Dict]:
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        body_lines = lines[start_line - 1 : end_line]
        if not body_lines:
            return None

        own_name = self._extract_own_name(node, source_bytes)
        qualified_name = own_name
        if context_name and own_name:
            qualified_name = f"{context_name}.{own_name}"
        elif context_name and not own_name:
            qualified_name = context_name
        if not qualified_name:
            qualified_name = relative_path.split("/")[-1]

        # Head-of-body preview: naturally captures the signature, docstring,
        # and field declarations that come before the first nested method,
        # so simple data classes / structs keep their field list even though
        # we no longer store the entire body verbatim.
        preview_lines = body_lines[:MAX_OVERVIEW_BODY_LINES]
        content = "\n".join(preview_lines).strip()
        truncated_body = len(body_lines) > MAX_OVERVIEW_BODY_LINES
        if len(content) > MAX_OVERVIEW_CHARS:
            content = content[:MAX_OVERVIEW_CHARS].rstrip()
            truncated_body = True

        member_names = self._collect_member_names(node, source_bytes, capture_types)
        members_line = ""
        if member_names:
            shown = member_names[:MAX_MEMBERS_LISTED]
            members_line = f"Members: {', '.join(shown)}"
            remaining = len(member_names) - len(shown)
            if remaining > 0:
                members_line += f" (+{remaining} more, indexed separately)"
        elif truncated_body:
            members_line = "(body truncated; see file for full contents)"

        content_parts = [part for part in [content, members_line] if part]
        full_content = "\n\n".join(content_parts)

        signature = body_lines[0].strip() if body_lines else ""
        searchable_text = "\n".join(
            part
            for part in [relative_path, context_name or "", qualified_name, signature, full_content]
            if part
        )

        return {
            "file_path": relative_path,
            "language": language,
            "symbol_name": qualified_name,
            "symbol_type": f"{node.type}_overview",
            "line_start": start_line,
            "line_end": end_line,
            "signature": signature,
            "content": full_content,
            "searchable_text": searchable_text,
            "metadata_json": {
                "parser": "tree-sitter",
                "kind": "container_overview",
                **({"parent": context_name} if context_name else {}),
            },
        }

    def _build_file_overview(self, root_node, source_bytes: bytes, lines: List[str], relative_path: str, language: str, symbol_chunks: List[Dict]) -> Optional[Dict]:
        if language == "text":
            return None
        path = Path(relative_path)
        parts = list(path.parts)
        directory = str(path.parent) if str(path.parent) != "." else "repository root"
        filename = path.name
        component = parts[1] if len(parts) >= 2 and parts[0] in {"packages", "apps"} else (parts[-2] if len(parts) >= 2 else "")
        role = self._infer_file_role(relative_path)
        imports, exports, export_targets = self._extract_module_edges(root_node, source_bytes)
        symbols, seen = [], set()
        for chunk in symbol_chunks:
            name = chunk.get("symbol_name")
            if name and name not in seen and chunk.get("symbol_type") != "fallback_chunk":
                seen.add(name); symbols.append(name)
            if len(symbols) >= 40:
                break
        overview = [f"File: {relative_path}", f"Filename: {filename}", f"Directory: {directory}", f"Language: {language}"]
        if component: overview.append(f"Package/component: {component}")
        overview.append(f"Role: {role}")
        if imports: overview.append("Imports: " + ", ".join(imports[:30]))
        if exports: overview.append("Exports: " + ", ".join(exports[:40]))
        if export_targets: overview.append("Export targets: " + ", ".join(export_targets[:30]))
        if symbols: overview.append("Symbols: " + ", ".join(symbols))
        content = "\n".join(overview)
        path_terms = " ".join(t for t in re.split(r"[\/._:-]+", relative_path) if t)
        return {
            "file_path": relative_path, "language": language,
            "symbol_name": f"{filename}:module-overview", "symbol_type": "file_overview",
            "line_start": 1, "line_end": 1, "signature": f"module {relative_path}",
            "content": content,
            "searchable_text": content + f"\nPath terms: {path_terms}\nModule: {path.stem}",
            "metadata_json": {"parser": "tree-sitter", "kind": "file_overview", "directory": directory, "component": component, "role": role, "imports": imports[:30], "exports": exports[:40], "export_targets": export_targets[:30]},
        }

    @staticmethod
    def _infer_file_role(relative_path: str) -> str:
        path = relative_path.lower().replace("\\", "/")
        name = Path(path).name
        if name in {"index.ts", "index.tsx", "index.js", "index.jsx"}:
            return "module/package entry point or barrel export"
        for needle, role in [("router", "router/API composition"), ("route", "request route/endpoint"), ("handler", "handler/job execution"), ("controller", "request controller"), ("service", "service/business logic"), ("transport", "transport/integration adapter"), ("adapter", "integration adapter"), ("repository", "data repository"), ("config", "configuration"), ("schema", "schema/type definition"), ("test", "test"), ("spec", "test/specification")]:
            if needle in name or f"/{needle}" in path:
                return role
        return "source module"

    def _extract_module_edges(self, root_node, source_bytes: bytes):
        imports, exports, targets = [], [], []
        for child in root_node.children:
            text = source_bytes[child.start_byte:child.end_byte].decode("utf-8", errors="ignore").strip()
            if not text: continue
            if child.type == "import_statement" or text.startswith("import "):
                target = self._module_target(text)
                if target and target not in imports: imports.append(target)
            if "export" in child.type or text.startswith("export ") or text.startswith("module.exports"):
                target = self._module_target(text)
                if target and target not in targets: targets.append(target)
                for name in self._exported_names(text):
                    if name not in exports: exports.append(name)
        return imports, exports, targets

    @staticmethod
    def _module_target(statement: str) -> Optional[str]:
        for pattern in [r"from\s+[\"']([^\"']+)[\"']", r"(?:import|export)\s+[\"']([^\"']+)[\"']", r"require\(\s*[\"']([^\"']+)[\"']\s*\)"]:
            match = re.search(pattern, statement)
            if match: return match.group(1)
        return None

    @staticmethod
    def _exported_names(statement: str) -> List[str]:
        names = []
        match = re.search(r"export\s*\{([^}]+)\}", statement, re.S)
        if match:
            for part in match.group(1).split(","):
                item = part.strip()
                if item: names.append(re.split(r"\s+as\s+", item)[-1].strip())
        match = re.search(r"export\s+(?:default\s+)?(?:async\s+)?(?:const|let|var|function|class|interface|type|enum)\s+([A-Za-z_$][A-Za-z0-9_$]*)", statement)
        if match: names.append(match.group(1))
        if statement.startswith("export default") and not match: names.append("default")
        if statement.startswith("export *"): names.append("*")
        return names

    def _collect_member_names(self, node, source_bytes: bytes, capture_types: set) -> List[str]:
        """Collect names of direct member symbols (methods/fields) inside a
        container without descending into nested containers, so a class's
        member list doesn't pick up grandchildren from an inner class."""
        names = []
        seen = set()

        def walk(current):
            for child in current.children:
                if child.type in capture_types:
                    name = self._extract_own_name(child, source_bytes)
                    if name and name not in seen:
                        seen.add(name)
                        names.append(name)
                    # Don't descend further into this member's own body.
                    continue
                walk(child)

        walk(node)
        return names

    @staticmethod
    def _extract_own_name(node, source_bytes: bytes) -> Optional[str]:
        name_node = node.child_by_field_name("name")
        if name_node is not None:
            candidate = source_bytes[name_node.start_byte : name_node.end_byte].decode(
                "utf-8", errors="ignore"
            ).strip()
            if candidate:
                return candidate
        return CodeParser._find_identifier(node, source_bytes)

    @staticmethod
    def _find_identifier(node, source_bytes: bytes) -> Optional[str]:
        stack = list(node.children)
        while stack:
            current = stack.pop(0)
            if current.type in IDENTIFIER_TYPES:
                return source_bytes[current.start_byte : current.end_byte].decode(
                    "utf-8", errors="ignore"
                ).strip()
            stack.extend(current.children)
        return None

    def _fallback_chunks(self, source: str, relative_path: str, language: str) -> List[Dict]:
        blocks = []
        lines = source.splitlines()
        buffer = []
        start_line = 1
        for index, line in enumerate(lines, start=1):
            if not buffer:
                start_line = index
            buffer.append(line)
            trigger = False
            if language == "text":
                trigger = len(buffer) >= 60 or (line.startswith("#") and len(buffer) > 8)
            else:
                trigger = (
                    re.match(r"^\s*(def |class |function |const |export |interface |type )", line)
                    and len(buffer) > 8
                ) or len(buffer) >= 80

            if trigger:
                chunk_text = "\n".join(buffer).strip()
                if chunk_text:
                    blocks.append(
                        {
                            "file_path": relative_path,
                            "language": language,
                            "symbol_name": f"{Path(relative_path).name}:{start_line}",
                            "symbol_type": "fallback_chunk",
                            "line_start": start_line,
                            "line_end": index,
                            "signature": buffer[0].strip(),
                            "content": chunk_text,
                            "searchable_text": f"{relative_path}\n{chunk_text}",
                            "metadata_json": {
                                "parser": "fallback",
                            },
                        }
                    )
                buffer = []

        if buffer:
            chunk_text = "\n".join(buffer).strip()
            if chunk_text:
                blocks.append(
                    {
                        "file_path": relative_path,
                        "language": language,
                        "symbol_name": f"{Path(relative_path).name}:{start_line}",
                        "symbol_type": "fallback_chunk",
                        "line_start": start_line,
                        "line_end": len(lines),
                        "signature": buffer[0].strip(),
                        "content": chunk_text,
                        "searchable_text": f"{relative_path}\n{chunk_text}",
                        "metadata_json": {
                            "parser": "fallback",
                        },
                    }
                )
        return blocks
