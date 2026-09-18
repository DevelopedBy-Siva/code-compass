import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from qdrant_client import QdrantClient


SERVER_ROOT = Path(__file__).resolve().parents[1]
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from evals.run_eval import compute_debug_stage_metrics
from src.code_parser import CodeParser
from src.embeddings import EmbeddingGenerator, RETRIEVAL_INSTRUCTION
from src.hybrid_search import HybridSearchEngine, RERANK_SUFFIX
from src.rag_system import CodebaseRAGSystem
from src.vector_store import QdrantVectorStore


class _FakeReranker:
    def __call__(self, signal):
        batch_size = signal.shape[0]
        logits = torch.zeros((batch_size, 1, 4), dtype=torch.float32)
        logits[:, -1, 1] = -signal
        logits[:, -1, 2] = signal
        return SimpleNamespace(logits=logits)


class RetrievalModelTests(unittest.TestCase):
    def test_qwen_embedding_uses_last_non_padding_token(self):
        states = torch.tensor(
            [
                [[1.0, 0.0], [2.0, 0.0], [9.0, 0.0]],
                [[3.0, 0.0], [8.0, 0.0], [7.0, 0.0]],
            ]
        )
        mask = torch.tensor([[1, 1, 0], [1, 0, 0]])
        pooled = EmbeddingGenerator._last_token_pool(states, mask)
        self.assertTrue(torch.equal(pooled, torch.tensor([[2.0, 0.0], [3.0, 0.0]])))
        self.assertIn("codebase question", RETRIEVAL_INSTRUCTION)

    def test_reranker_scores_yes_probability_and_varies_by_document(self):
        engine = HybridSearchEngine.__new__(HybridSearchEngine)
        engine.rerank_batch_size = 2
        engine.device = "cpu"
        engine._no_token_id = 1
        engine._yes_token_id = 2
        engine.reranker = _FakeReranker()
        engine._prepare_rerank_inputs = lambda query, docs: {
            "signal": torch.tensor(
                [2.0 if document.startswith("relevant") else -2.0 for document in docs]
            )
        }

        scores = engine._score_relevance_batch(
            "query",
            ["relevant implementation", "unrelated material"],
        )
        self.assertGreater(scores[0], scores[1])
        self.assertTrue(RERANK_SUFFIX.endswith("</think>\n\n"))


class RetrievalPipelineTests(unittest.TestCase):
    def test_translated_document_families_do_not_flood_candidates(self):
        candidates = [
            {"id": "ja", "file_path": "docs/ja/docs/how-to/openapi.md"},
            {"id": "en", "file_path": "docs/en/docs/how-to/openapi.md"},
            {"id": "code", "file_path": "fastapi/openapi/utils.py"},
        ]
        selected = CodebaseRAGSystem._deduplicate_candidates(candidates, top_k=3)
        self.assertEqual([item["id"] for item in selected], ["en", "code"])

    def test_request_to_response_question_is_cross_file(self):
        intent = CodebaseRAGSystem._question_intent(
            "How does an incoming HTTP request reach a view and become a response?"
        )
        self.assertEqual(intent, "cross_file")

    def test_stage_metric_preserves_original_rank(self):
        debug = [
            {"file_path": "wrong.py", "semantic_rank": 1},
            {"file_path": "expected.py", "semantic_rank": 2},
        ]
        metrics = compute_debug_stage_metrics(["expected.py"], debug, "semantic_rank")
        self.assertEqual(metrics["retrieval_hit"], 1)
        self.assertEqual(metrics["reciprocal_rank"], 0.5)

    def test_path_only_match_cannot_overpower_dual_retriever_match(self):
        system = CodebaseRAGSystem.__new__(CodebaseRAGSystem)
        fused = [
            {
                "id": "dual",
                "file_path": "core/query.py",
                "rrf_score": 2.0 / 61.0,
            }
        ]
        path = [
            {
                "id": "path-only",
                "file_path": "docs/api/example.md",
                "path_score": 20.0,
            }
        ]
        merged = system._merge_ranked_candidates(fused, path, top_k=2)
        self.assertEqual(merged[0]["id"], "dual")

    def test_unicode_before_symbol_does_not_corrupt_symbol_name(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "sample.py"
            path.write_text(
                '# café introduces multibyte text\n\nclass QuerySet:\n    def filter(self):\n        return self\n',
                encoding="utf-8",
            )
            chunks = CodeParser().chunk_file(str(path), directory)

        names = {chunk["symbol_name"] for chunk in chunks}
        self.assertIn("QuerySet", names)
        self.assertIn("QuerySet.filter", names)


class QdrantVectorStoreTests(unittest.TestCase):
    def test_repository_cache_generation_is_replaced_only_after_activation(self):
        client = QdrantClient(location=":memory:")
        store = QdrantVectorStore(
            embedding_dim=3,
            client=client,
            collection_name="test_code_compass",
        )
        store.upsert_batch_size = 1
        metadata = [
            {
                "repository_key": "github:owner/repo@main",
                "cache_generation": "generation-1",
                "cache_ready": False,
                "file_path": "src/relevant.py",
                "symbol_name": "relevant",
                "signature": "def relevant():",
                "content": "relevant implementation",
                "searchable_text": "src relevant implementation",
            },
            {
                "repository_key": "github:owner/repo@main",
                "cache_generation": "generation-1",
                "cache_ready": False,
                "file_path": "src/secondary.py",
                "symbol_name": "secondary",
                "signature": "def secondary():",
                "content": "secondary implementation",
                "searchable_text": "src secondary implementation",
            },
            {
                "repository_key": "github:owner/other@main",
                "cache_generation": "other-generation",
                "cache_ready": True,
                "file_path": "src/other.py",
                "symbol_name": "other",
                "signature": "def other():",
                "content": "other implementation",
                "searchable_text": "src other implementation",
            },
        ]
        ids = store.add_embeddings(
            np.array(
                [
                    [1.0, 0.0, 0.0],
                    [0.0, 0.0, 1.0],
                    [0.0, 1.0, 0.0],
                ],
                dtype="float32",
            ),
            metadata,
        )

        self.assertEqual(len(ids), 3)
        self.assertEqual(store.get_stats()["total_vectors"], 3)
        self.assertEqual(store.get_repository_chunks("github:owner/repo@main"), [])

        store.activate_repository_generation(
            "github:owner/repo@main",
            "generation-1",
        )

        hits = store.search(
            np.array([1.0, 0.0, 0.0], dtype="float32"),
            k=5,
            repository_key="github:owner/repo@main",
        )
        self.assertEqual(len(hits), 2)
        self.assertEqual(hits[0][1]["file_path"], "src/relevant.py")

        restored = store.get_repository_chunks("github:owner/repo@main")
        self.assertEqual(len(restored), 2)
        self.assertEqual(
            {chunk["content"] for chunk in restored},
            {"relevant implementation", "secondary implementation"},
        )

        store.add_embeddings(
            np.array([[0.0, 1.0, 0.0]], dtype="float32"),
            [
                {
                    "repository_key": "github:owner/repo@main",
                    "cache_generation": "generation-2",
                    "cache_ready": False,
                    "file_path": "src/replacement.py",
                    "symbol_name": "replacement",
                    "signature": "def replacement():",
                    "content": "replacement implementation",
                    "searchable_text": "src replacement implementation",
                }
            ],
        )
        self.assertEqual(
            {chunk["file_path"] for chunk in store.get_repository_chunks("github:owner/repo@main")},
            {"src/relevant.py", "src/secondary.py"},
        )

        store.activate_repository_generation(
            "github:owner/repo@main",
            "generation-2",
        )
        self.assertEqual(
            [chunk["file_path"] for chunk in store.get_repository_chunks("github:owner/repo@main")],
            ["src/replacement.py"],
        )
        self.assertEqual(store.get_stats()["total_vectors"], 2)

        store.delete_repository_cache("github:owner/repo@main")
        self.assertEqual(store.get_repository_chunks("github:owner/repo@main"), [])
        self.assertEqual(store.get_stats()["total_vectors"], 1)

        store.clear()
        self.assertEqual(store.get_stats()["total_vectors"], 0)


if __name__ == "__main__":
    unittest.main()
