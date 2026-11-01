import sys
import unittest
from pathlib import Path


SERVER_ROOT = Path(__file__).resolve().parents[1]
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from evals.metrics import (
    judge_faithfulness,
    score_ranking,
    stage_diagnostics,
    summarize_answer_quality,
    summarize_comparison,
    summarize_diagnostics,
)


class EvaluationMetricTests(unittest.TestCase):
    def test_faithfulness_judge_uses_exact_generation_context(self):
        class FakeRagSystem:
            def __init__(self):
                self.user_prompt = ""

            def _generate_markdown_response(self, system_prompt, user_prompt):
                self.user_prompt = user_prompt
                return "0.75", "stop"

        rag_system = FakeRagSystem()
        context = "[Source 1]\nLines shown: 1, 90-92\nlate_return_branch()"
        score = judge_faithfulness(
            rag_system,
            "How does it finish?",
            "It uses the late return branch. [1]",
            context,
        )
        self.assertEqual(score, 0.75)
        self.assertIn(context, rag_system.user_prompt)

    def test_metrics_use_first_relevant_result_with_expected_cutoffs(self):
        metrics = score_ranking(
            ["src/target.py"],
            ["src/other.py", "src/another.py", "src/target.py"],
        )
        self.assertEqual(metrics["hit_at_1"], 0)
        self.assertEqual(metrics["hit_at_5"], 1)
        self.assertEqual(metrics["hit_at_8"], 1)
        self.assertAlmostEqual(metrics["mrr_at_8"], 1 / 3)
        self.assertEqual(metrics["first_relevant_rank"], 3)

    def test_hit_at_8_and_mrr_at_8_ignore_relevant_results_beyond_eight(self):
        metrics = score_ranking(
            ["src/target.py"],
            [f"src/other-{rank}.py" for rank in range(8)] + ["src/target.py"],
        )
        self.assertEqual(metrics["hit_at_8"], 0)
        self.assertEqual(metrics["mrr_at_8"], 0)
        self.assertEqual(metrics["first_relevant_rank"], 9)

    def test_directory_relevance_and_path_normalization(self):
        metrics = score_ranking(["./src/components/"], ["src\\components\\button.py"])
        self.assertEqual(metrics["hit_at_1"], 1)
        self.assertEqual(metrics["mrr_at_8"], 1.0)

    def test_comparison_uses_same_records_for_overall_metrics(self):
        records = [
            {
                "vector_only": score_ranking(["target.py"], ["target.py"]),
                "hybrid_rerank": score_ranking(["target.py"], ["other.py", "target.py"]),
            },
            {
                "vector_only": score_ranking(["target.py"], ["other.py"]),
                "hybrid_rerank": score_ranking(["target.py"], ["target.py"]),
            },
        ]
        summary = summarize_comparison(records)
        self.assertEqual(summary["query_count"], 2)
        self.assertEqual(summary["vector_only"]["hit_at_1"], 0.5)
        self.assertEqual(summary["hybrid_rerank"]["hit_at_1"], 0.5)
        self.assertEqual(summary["vector_only"]["hit_at_5"], 0.5)

    def test_answer_quality_summary_is_separate_from_retrieval_metrics(self):
        summary = summarize_answer_quality(
            [
                {
                    "citation_present": 1,
                    "expected_source_retrieved": 1,
                    "grounded_answer_proxy": 1,
                    "faithfulness_judge": 0.75,
                    "latency_ms": 100.0,
                },
                {
                    "citation_present": 0,
                    "expected_source_retrieved": 0,
                    "grounded_answer_proxy": 0,
                    "faithfulness_judge": None,
                    "latency_ms": 200.0,
                },
            ]
        )
        self.assertEqual(summary["grounded_answer_proxy_rate"], 0.5)
        self.assertEqual(summary["faithfulness_judge_coverage"], 1)
        self.assertEqual(summary["latency_p95_ms"], 200.0)

    def test_stage_scores_are_kept_in_diagnostics_and_preserve_rank(self):
        debug = [
            {"file_path": "wrong.py", "semantic_rank": 1, "final_rank": 2},
            {"file_path": "src/expected.py", "semantic_rank": 2, "final_rank": 1},
        ]
        diagnostics = stage_diagnostics(["src/expected.py"], debug)
        self.assertEqual(
            diagnostics["semantic"]["ranking_metrics"]["mrr_at_8"],
            0.5,
        )
        self.assertEqual(diagnostics["final"]["ranking_metrics"]["hit_at_1"], 1)
        self.assertEqual(diagnostics["candidate_recall"]["expected_source_found"], 1)
        summary = summarize_diagnostics([{"stage_scores": diagnostics}])
        self.assertEqual(summary["candidate_recall_rate"], 1.0)
        self.assertEqual(summary["stages"]["semantic"]["mrr_at_8"], 0.5)


if __name__ == "__main__":
    unittest.main()
