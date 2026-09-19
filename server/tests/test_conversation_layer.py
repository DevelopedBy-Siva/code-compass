import json
import sys
import unittest
from datetime import datetime, timedelta
from pathlib import Path
from threading import RLock


SERVER_ROOT = Path(__file__).resolve().parents[1]
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from src.rag_system import CodebaseRAGSystem, Repository


class _FakeBedrock:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def converse(self, **kwargs):
        self.calls.append(kwargs)
        text = self.responses.pop(0)
        return {
            "output": {"message": {"content": [{"text": text}]}},
            "stopReason": "stop",
        }


def _repository(status="indexed"):
    return Repository(
        id=1,
        github_url="session::github:example/project@main",
        source_url="https://github.com/example/project",
        session_key="test-session",
        session_expires_at=datetime.utcnow() + timedelta(hours=1),
        owner="example",
        name="project",
        status=status,
    )


class ConversationPlanningTests(unittest.TestCase):
    def test_greeting_is_classified_without_calling_bedrock(self):
        system = CodebaseRAGSystem.__new__(CodebaseRAGSystem)
        system.llm_client = _FakeBedrock([])

        plan = system._plan_conversation("Hello!", [])

        self.assertEqual(plan.route, "casual")
        self.assertEqual(system.llm_client.calls, [])

    def test_greeting_bypasses_repository_readiness_and_retrieval(self):
        system = CodebaseRAGSystem.__new__(CodebaseRAGSystem)
        repo = _repository(status="queued")
        system.repo_lock = RLock()
        system.repositories = {repo.id: repo}
        system.repo_chunks = {}
        system.indexing_progress = {}
        system.cancelled_repo_ids = set()
        system.session_ttl_minutes = 120

        result = system.answer_question(
            repo_id=repo.id,
            session_key=repo.session_key,
            question="Hi",
        )

        self.assertEqual(result["response_type"], "casual")
        self.assertEqual(result["sources"], [])

    def test_unanchored_pronoun_gets_targeted_clarification(self):
        system = CodebaseRAGSystem.__new__(CodebaseRAGSystem)

        plan = system._plan_conversation("How does it work?", [])

        self.assertEqual(plan.route, "clarify")
        self.assertIn("“it”", plan.clarification_question)
        self.assertNotIn("unclear", plan.clarification_question.lower())

    def test_follow_up_is_rewritten_from_recent_history(self):
        system = CodebaseRAGSystem.__new__(CodebaseRAGSystem)
        system.llm_model = "test-model"
        system.llm_client = _FakeBedrock(
            [
                json.dumps(
                    {
                        "rewritten_query": (
                            "How does the authentication middleware validate tokens?"
                        ),
                        "needs_clarification": False,
                        "clarification_question": "",
                    }
                )
            ]
        )
        history = [
            {
                "role": "user",
                "content": "Where is the authentication middleware defined?",
            },
            {
                "role": "assistant",
                "content": (
                    "The authentication middleware is defined in src/auth.py and "
                    "runs before protected request handlers."
                ),
            },
        ]

        plan = system._plan_conversation("How does it validate tokens?", history)

        self.assertEqual(plan.route, "retrieve")
        self.assertEqual(
            plan.rewritten_query,
            "How does the authentication middleware validate tokens?",
        )
        self.assertIn("Recent conversation", plan.rewrite_prompt)

    def test_generic_model_clarification_is_replaced_with_targeted_question(self):
        system = CodebaseRAGSystem.__new__(CodebaseRAGSystem)
        system.llm_model = "test-model"
        system.llm_client = _FakeBedrock(
            [
                json.dumps(
                    {
                        "rewritten_query": "",
                        "needs_clarification": True,
                        "clarification_question": "Could you clarify what you mean?",
                    }
                )
            ]
        )
        history = [
            {"role": "user", "content": "Explain authentication and sessions."}
        ]

        plan = system._plan_conversation("How does it work?", history)

        self.assertEqual(plan.route, "clarify")
        self.assertIn("component, file, or behavior", plan.clarification_question)

    def test_repository_reference_is_not_treated_as_ambiguous(self):
        system = CodebaseRAGSystem.__new__(CodebaseRAGSystem)

        plan = system._plan_conversation("What does this repository do?", [])

        self.assertEqual(plan.route, "retrieve")
        self.assertEqual(plan.rewritten_query, "What does this repository do?")

    def test_relative_that_does_not_trigger_follow_up_rewriting(self):
        system = CodebaseRAGSystem.__new__(CodebaseRAGSystem)

        plan = system._plan_conversation(
            "How does the middleware that validates tokens work?",
            [],
        )

        self.assertEqual(plan.route, "retrieve")

    def test_elliptical_follow_up_without_history_requests_context(self):
        system = CodebaseRAGSystem.__new__(CodebaseRAGSystem)

        plan = system._plan_conversation("Where exactly?", [])

        self.assertEqual(plan.route, "clarify")
        self.assertIn("earlier component", plan.clarification_question)


class AnswerExperienceTests(unittest.TestCase):
    def test_prompt_requires_layered_repository_answer(self):
        system = CodebaseRAGSystem.__new__(CodebaseRAGSystem)
        system.llm_model = "test-model"
        system.indexing_progress = {}
        layered_answer = """## Answer
The request enters the API handler and is passed to the service. [1]

## Relevant implementation
[IMPLEMENTATION_SNIPPETS]

## Why this code matters
This function owns the handoff from HTTP input to the service layer. [1]

## Related files
- `app.py` — Receives the request and invokes the service. [1]
"""
        system.llm_client = _FakeBedrock([layered_answer])
        trace = {}
        source = {
            "file_path": "app.py",
            "language": "python",
            "symbol_name": "handle_request",
            "symbol_type": "function_definition",
            "line_start": 1,
            "line_end": 4,
            "signature": "def handle_request(request):",
            "content": "def handle_request(request):\n    payload = request.json()\n    result = service.run(payload)\n    return result",
            "semantic_score": 0.9,
            "bm25_score": 1.0,
            "rrf_score": 0.03,
            "rerank_score": 0.8,
        }

        result = system._generate_answer(
            _repository(),
            "Give me a repository overview",
            [source],
            rewritten_query="Give me a repository overview",
            trace=trace,
        )

        self.assertEqual(result["answer"], result["direct_answer"])
        self.assertIn("request enters the API handler", result["direct_answer"])
        self.assertIn("owns the handoff", result["why_this_code_matters"])
        self.assertEqual(result["implementation_snippets"][0]["file_path"], "app.py")
        self.assertEqual(result["related_files"][0]["file_path"], "app.py")
        self.assertIn("## Relevant implementation", trace["final_prompt"])
        self.assertIn("Standalone interpretation", trace["final_prompt"])

    def test_layered_answer_validator_rejects_missing_section(self):
        incomplete = """## Answer
Direct answer.

## Relevant implementation
[IMPLEMENTATION_SNIPPETS]
"""
        self.assertFalse(CodebaseRAGSystem._has_layered_answer(incomplete))

    def test_snippet_is_centered_and_never_exceeds_twenty_lines(self):
        content = "\n".join(f"line {index}" for index in range(1, 51))
        source = {
            "file_path": "src/service.py",
            "language": "python",
            "symbol_name": "target_handler",
            "line_start": 100,
            "signature": "def target_handler():",
            "content": content.replace("line 27", "def target_handler():"),
        }

        snippet = CodebaseRAGSystem._extract_display_snippet(source)

        self.assertLessEqual(len(snippet["code"].splitlines()), 20)
        self.assertIn("def target_handler():", snippet["code"])
        self.assertTrue(snippet["expandable"])
        self.assertLessEqual(len(snippet["expanded_code"].splitlines()), 60)

    def test_implementation_sources_rank_above_tests_and_docs(self):
        sources = [
            {
                "id": "docs",
                "file_path": "README.md",
                "language": "text",
                "symbol_type": "fallback_chunk",
                "final_score": 1.0,
            },
            {
                "id": "test",
                "file_path": "tests/test_service.py",
                "language": "python",
                "symbol_type": "function_definition",
                "final_score": 0.9,
            },
            {
                "id": "impl",
                "file_path": "src/service.py",
                "language": "python",
                "symbol_type": "function_definition",
                "final_score": 0.8,
            },
        ]

        ranked = CodebaseRAGSystem._rank_sources_for_answer(
            "How is the service implemented?",
            sources,
        )

        self.assertEqual(ranked[0]["id"], "impl")
        self.assertEqual(ranked[-1]["id"], "test")


if __name__ == "__main__":
    unittest.main()
