import hashlib
from typing import List, Tuple
from pathlib import Path
import pypdf


class DocumentProcessor:
    # FIX: 512 chars was too small — caused mid-thought splits that hurt faithfulness.
    # 1200 chars with 150 overlap keeps full function signatures + docstrings intact.
    def __init__(self, chunk_size: int = 1200, chunk_overlap: int = 150):
        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap

    def extract_text_from_pdf(self, file_path: str) -> str:
        text = ""
        try:
            with open(file_path, "rb") as file:
                pdf_reader = pypdf.PdfReader(file)
                for page in pdf_reader.pages:
                    text += page.extract_text() + "\n"
        except Exception as e:
            raise ValueError(f"Error reading PDF: {str(e)}")
        return text.strip()

    def chunk_text(self, text: str) -> List[str]:
        if not text:
            return []

        chunks = []
        start = 0
        text_length = len(text)

        while start < text_length:
            end = start + self.chunk_size
            chunk = text[start:end]

            if end < text_length:
                # FIX: prefer paragraph breaks > sentence breaks > newlines.
                # Previously only checked period + newline, missing paragraph boundaries.
                last_double_newline = chunk.rfind("\n\n")
                last_period = chunk.rfind(".")
                last_newline = chunk.rfind("\n")

                if last_double_newline > self.chunk_size * 0.4:
                    break_point = last_double_newline + 1
                elif last_period > self.chunk_size * 0.5:
                    break_point = last_period + 1
                elif last_newline > self.chunk_size * 0.4:
                    break_point = last_newline + 1
                else:
                    break_point = self.chunk_size

                chunk = chunk[:break_point]
                end = start + break_point

            stripped = chunk.strip()
            if stripped:
                chunks.append(stripped)

            start = end - self.chunk_overlap

        return [c for c in chunks if c]

    def process_document(self, file_path: str) -> Tuple[str, List[str]]:
        file_ext = Path(file_path).suffix.lower()
        if file_ext == ".pdf":
            text = self.extract_text_from_pdf(file_path)
        elif file_ext == ".txt":
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
        else:
            raise ValueError(f"Unsupported file type: {file_ext}")

        chunks = self.chunk_text(text)
        return text, chunks

    @staticmethod
    def compute_file_hash(file_path: str) -> str:
        hash_md5 = hashlib.md5()
        with open(file_path, "rb") as f:
            for chunk in iter(lambda: f.read(4096), b""):
                hash_md5.update(chunk)
        return hash_md5.hexdigest()