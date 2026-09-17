import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import torch


SERVER_ROOT = Path(__file__).resolve().parents[1]
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from evals.run_eval import compute_debug_stage_metrics
from src.code_parser import CodeParser
from src.embeddings import EmbeddingGenerator, RETRIEVAL_INSTRUCTION
from src.hybrid_search import HybridSearchEngine, RERANK_SUFFIX
from src.rag_system import CodebaseRAGSystem


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


if __name__ == "__main__":
    unittest.main()
