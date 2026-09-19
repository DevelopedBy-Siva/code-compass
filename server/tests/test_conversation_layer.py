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
    def test_prompt_and_response_use_code_walkthrough_presentation(self):
        system = CodebaseRAGSystem.__new__(CodebaseRAGSystem)
        system.llm_model = "test-model"
        system.indexing_progress = {}
        presented_answer = (
            "`handle_request` in `app.py` owns the HTTP-to-service handoff. [1]\n\n"
            "1. It reads the request payload. [1]\n"
            "2. It passes that payload to `service.run` and returns the result. [1]"
        )
        system.llm_client = _FakeBedrock([presented_answer])
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
            "Where is handle_request implemented?",
            [source],
            rewritten_query="Where is handle_request implemented?",
            trace=trace,
        )

        self.assertEqual(result["answer"], result["direct_answer"])
        self.assertEqual(result["answer_mode"], "implementation")
        self.assertIn("owns the HTTP-to-service handoff", result["direct_answer"])
        self.assertEqual(result["implementation_snippets"][0]["file_path"], "app.py")
        self.assertEqual(result["related_files"][0]["file_path"], "app.py")
        self.assertEqual(
            [section["title"] for section in result["answer_sections"]],
            ["Explanation", "Relevant implementation", "Related files"],
        )
        self.assertIn("senior engineer onboarding", trace["final_prompt"])
        self.assertIn("Available implementation excerpts", trace["final_prompt"])
        self.assertIn("`handle_request`", trace["final_prompt"])
        self.assertIn("Standalone interpretation", trace["final_prompt"])
        self.assertIn("def handle_request(request):", trace["generation_context"])
        self.assertEqual(
            trace["generation_context_stats"]["total_content_budget"],
            1500,
        )

    def test_answer_mode_matches_the_question_shape(self):
        cases = {
            "How do invalid requests flow through validation?": "architecture",
            "How is the timeout configured?": "configuration",
            "Why does token validation keep failing?": "debugging",
            "Where is the request handler implemented?": "implementation",
        }

        for question, expected_mode in cases.items():
            with self.subTest(question=question):
                self.assertEqual(
                    CodebaseRAGSystem._answer_mode(question),
                    expected_mode,
                )

    def test_related_files_are_derived_from_retrieved_sources(self):
        sources = [
            {
                "file_path": "src/settings.py",
                "symbol_name": "REQUEST_TIMEOUT",
                "symbol_type": "assignment",
            },
            {
                "file_path": "tests/test_settings.py",
                "symbol_name": "test_timeout",
                "symbol_type": "function_definition",
            },
        ]

        related = CodebaseRAGSystem._build_related_files(
            "How is the request timeout configured?",
            sources,
        )

        self.assertEqual([item["file_path"] for item in related], ["src/settings.py"])
        self.assertIn("REQUEST_TIMEOUT", related[0]["description"])

    def test_related_files_follow_flow_order_and_skip_examples(self):
        sources = [
            {
                "file_path": "fastapi/exception_handlers.py",
                "symbol_name": "request_validation_exception_handler",
                "symbol_type": "function_definition",
                "content": "return JSONResponse(status_code=422, content=exc.errors())",
            },
            {
                "file_path": "docs_src/tutorial004.py",
                "symbol_name": "tutorial004",
                "symbol_type": "function_definition",
            },
            {
                "file_path": "fastapi/routing.py",
                "symbol_name": "get_request_handler",
                "symbol_type": "function_definition",
                "content": (
                    "solved_result = await solve_dependencies(request=request)\n"
                    "errors = solved_result.errors\n"
                    "raw_response = await run_endpoint_function(...)"
                ),
            },
            {
                "file_path": "fastapi/exceptions.py",
                "symbol_name": "RequestValidationError",
                "symbol_type": "class_definition",
            },
        ]

        related = CodebaseRAGSystem._build_related_files(
            "How are invalid requests converted into HTTP responses?",
            sources,
            "The flow starts in routing. [3] It raises an exception. [4] "
            "The handler renders the response. [1]",
        )

        self.assertEqual(
            [item["file_path"] for item in related],
            [
                "fastapi/routing.py",
                "fastapi/exceptions.py",
                "fastapi/exception_handlers.py",
            ],
        )
        self.assertIn("dependency resolution", related[0]["description"])
        self.assertIn("validation", related[0]["description"])
        self.assertIn("error rendering", related[1]["description"])
        self.assertIn("HTTP response", related[2]["description"])

    def test_snippets_follow_the_explanations_citation_order(self):
        snippets = [
            {"source": 1, "file_path": "exception_handlers.py"},
            {"source": 3, "file_path": "routing.py"},
        ]

        ordered = CodebaseRAGSystem._order_by_citations(
            snippets,
            "Request processing starts in routing. [3] Failures reach the handler. [1]",
        )

        self.assertEqual(
            [snippet["file_path"] for snippet in ordered],
            ["routing.py", "exception_handlers.py"],
        )

    def test_snippet_is_centered_with_surrounding_context(self):
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

        self.assertLessEqual(len(snippet["code"].splitlines()), 28)
        self.assertIn("def target_handler():", snippet["code"])
        self.assertTrue(snippet["expandable"])
        self.assertLessEqual(len(snippet["expanded_code"].splitlines()), 70)

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
