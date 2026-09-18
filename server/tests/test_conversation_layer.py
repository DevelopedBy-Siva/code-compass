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
    def test_overview_prompt_requires_all_structured_sections(self):
        system = CodebaseRAGSystem.__new__(CodebaseRAGSystem)
        system.llm_model = "test-model"
        system.indexing_progress = {}
        overview = """## Purpose
Explains a project. [1]

## Architecture
A service. [1]

## Technologies
- Python [1]

## Main components
- `app.py` — API [1]

## Request flow
Input → API → response. [1]
"""
        system.llm_client = _FakeBedrock([overview])
        trace = {}
        source = {
            "file_path": "README.md",
            "language": "text",
            "symbol_name": "README.md",
            "symbol_type": "module",
            "line_start": 1,
            "line_end": 20,
            "signature": "",
            "content": "Project purpose, architecture, Python API, and request flow.",
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

        self.assertTrue(system._has_structured_overview(result["answer"]))
        self.assertIn("## Purpose", trace["final_prompt"])
        self.assertIn("Standalone interpretation", trace["final_prompt"])

    def test_structured_overview_validator_rejects_missing_section(self):
        incomplete = """## Purpose
Purpose.

## Architecture
Architecture.
"""
        self.assertFalse(CodebaseRAGSystem._has_structured_overview(incomplete))


if __name__ == "__main__":
    unittest.main()
