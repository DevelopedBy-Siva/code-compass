import re
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
}

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
}

IDENTIFIER_TYPES = {
    "identifier",
    "property_identifier",
    "type_identifier",
    "field_identifier",
}


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

    def chunk_file(self, file_path: str, repo_root: str) -> List[Dict]:
        language = self.detect_language(file_path)
        source = Path(file_path).read_text(encoding="utf-8", errors="ignore")
        relative_path = str(Path(file_path).resolve().relative_to(Path(repo_root).resolve()))

        if not source.strip():
            return []

        parser = self._get_parser(language)
        if parser is None:
            return self._fallback_chunks(source, relative_path, language)

        tree = parser.parse(bytes(source, "utf-8"))
        lines = source.splitlines()
        chunks = []
        capture_types = SYMBOL_NODE_TYPES.get(language, set())

        def visit(node):
            if node.type in capture_types:
                chunk = self._build_chunk(node, source, lines, relative_path, language)
                if chunk:
                    chunks.append(chunk)
                    return
            for child in node.children:
                visit(child)

        visit(tree.root_node)

        if not chunks:
            return self._fallback_chunks(source, relative_path, language)

        return chunks

    def _build_chunk(self, node, source: str, lines: List[str], relative_path: str, language: str) -> Optional[Dict]:
        start_line = node.start_point[0] + 1
        end_line = node.end_point[0] + 1
        snippet = "\n".join(lines[start_line - 1 : end_line]).strip()
        if len(snippet.splitlines()) < 2:
            return None

        name_node = node.child_by_field_name("name")
        symbol_name = None
        if name_node is not None:
            symbol_name = source[name_node.start_byte : name_node.end_byte].strip()
        if not symbol_name:
            symbol_name = self._find_identifier(node, source)

        signature = lines[start_line - 1].strip() if start_line - 1 < len(lines) else ""
        searchable_text = "\n".join(
            part for part in [relative_path, symbol_name or "", signature, snippet] if part
        )

        return {
            "file_path": relative_path,
            "language": language,
            "symbol_name": symbol_name or relative_path.split("/")[-1],
            "symbol_type": node.type,
            "line_start": start_line,
            "line_end": end_line,
            "signature": signature,
            "content": snippet,
            "searchable_text": searchable_text,
            "metadata_json": {
                "parser": "tree-sitter",
            },
        }

    def _find_identifier(self, node, source: str) -> Optional[str]:
        stack = list(node.children)
        while stack:
            current = stack.pop(0)
            if current.type in IDENTIFIER_TYPES:
                return source[current.start_byte : current.end_byte].strip()
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
