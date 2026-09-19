import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SERVER_ROOT = Path(__file__).resolve().parents[1]
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from evals.run_eval import judge_faithfulness
from src.generation_context import GenerationContextBuilder


def _source(
    content: str,
    *,
    symbol: str = "handler",
    file_path: str = "src/service.py",
    line_start: int = 1,
    language: str = "python",
    symbol_type: str = "function_definition",
) -> dict:
    return {
        "file_path": file_path,
        "language": language,
        "symbol_name": symbol,
        "symbol_type": symbol_type,
        "line_start": line_start,
        "line_end": line_start + len(content.splitlines()) - 1,
        "signature": content.splitlines()[0] if content.splitlines() else symbol,
        "content": content,
    }


def _long_function(name: str, terminal: str = "return result") -> str:
    filler = "\n".join(f"    value_{index} = {index}" for index in range(140))
    return f"def {name}(request):\n    result = prepare(request)\n{filler}\n    {terminal}"


class GenerationContextBuilderTests(unittest.TestCase):
    def setUp(self):
        self.builder = GenerationContextBuilder()

    def test_complete_short_function_is_preserved(self):
        content = """def handle(request):
    payload = request.json()
    result = service.run(payload)
    return result"""

        context = self.builder.build(
            [_source(content)],
            "How is handle implemented?",
            "implementation",
        )

        selected = context.selected_sources[0]
        self.assertTrue(selected.complete)
        self.assertEqual(selected.text, content)
        self.assertEqual(selected.line_ranges, [(1, 4)])
        self.assertNotIn("omitted lines", selected.text)

    def test_long_function_includes_structural_anchors_without_mid_line_cuts(self):
        documentation = "\n".join(f"    Documentation detail {index}." for index in range(90))
        content = f'''def validate_token(token, issuer="local"):
    """
{documentation}
    """
    state = load_state(issuer)
    if token.issuer != issuer:
        raise InvalidIssuer(token.issuer)
    result = validate_signature(token, state)
    return result'''

        context = self.builder.build(
            [_source(content, symbol="validate_token", line_start=100)],
            "Why does token validation fail for an invalid issuer?",
            "debugging",
        )

        selected = context.selected_sources[0]
        self.assertLessEqual(selected.used_chars, 1500)
        self.assertIn(
            "Signature (line 100): def validate_token(token, issuer='local'):",
            selected.text,
        )
        self.assertIn("state = load_state(issuer)", selected.text)
        self.assertIn("if token.issuer != issuer:", selected.text)
        self.assertIn("raise InvalidIssuer(token.issuer)", selected.text)
        self.assertIn("return result", selected.text)
        self.assertIn("[omitted lines", selected.text)
        self.assertIn("Lines shown:", context.blocks[0])

        original_lines = set(content.splitlines())
        selected_code_lines = [
            line
            for line in selected.text.splitlines()
            if line
            and not line.startswith("Signature (")
            and not line.startswith("... [omitted lines")
        ]
        self.assertTrue(all(line in original_lines for line in selected_code_lines))

    def test_adaptive_budget_preserves_short_sources_and_deepens_top_implementation(self):
        short = "def small():\n    return 1\n" + "# context\n" * 70
        sources = [
            _source(_long_function("primary"), symbol="primary"),
            _source(short, symbol="small", file_path="src/small.py"),
            _source(
                _long_function("secondary"),
                symbol="secondary",
                file_path="src/secondary.py",
            ),
        ]

        implementation = self.builder.build(
            sources,
            "How is primary implemented?",
            "implementation",
        )
        architecture = self.builder.build(
            sources,
            "How does the request flow across these components?",
            "architecture",
        )

        self.assertLessEqual(implementation.used_chars, implementation.total_budget)
        self.assertEqual(implementation.total_budget, 4500)
        self.assertTrue(implementation.selected_sources[1].complete)
        self.assertGreater(
            implementation.selected_sources[0].budget,
            implementation.selected_sources[2].budget,
        )
        self.assertLessEqual(
            abs(
                architecture.selected_sources[0].budget
                - architecture.selected_sources[2].budget
            ),
            1,
        )

    def test_depends_regression_includes_runtime_return_after_long_documentation(self):
        documentation = "\n".join(
            f"    Dependency documentation line {index}." for index in range(100)
        )
        content = f'''def Depends(dependency=None, *, use_cache=True, scope=None):
    """
{documentation}
    """
    return params.Depends(
        dependency=dependency,
        use_cache=use_cache,
        scope=scope,
    )'''

        selected = self.builder.build(
            [_source(content, symbol="Depends", file_path="fastapi/param_functions.py")],
            "How does FastAPI resolve dependencies declared with Depends?",
            "architecture",
        ).selected_sources[0]

        self.assertIn("def Depends(dependency=None, *, use_cache=True, scope=None):", selected.text)
        self.assertIn("return params.Depends(", selected.text)

    def test_csrf_regressions_include_late_exception_and_token_branches(self):
        mapping = "\n".join(
            f'        "message_{index}": "detail {index}",' for index in range(90)
        )
        csrf_failure = f'''def csrf_failure(request, template_name):
    context = {{
{mapping}
    }}
    try:
        body = render(template_name, context)
    except TemplateDoesNotExist:
        if template_name == DEFAULT_TEMPLATE:
            body = render_fallback(context)
        else:
            raise
    return HttpResponseForbidden(body)'''

        comments = "\n".join(
            f"    # Security rationale line {index}." for index in range(100)
        )
        process_view = f'''def process_view(self, request, callback):
    if request.method == "GET":
        return self._accept(request)
{comments}
    try:
        self._check_token(request)
    except RejectRequest as exc:
        return self._reject(request, exc.reason)
    return self._accept(request)'''

        failure_context = self.builder.build(
            [_source(csrf_failure, symbol="csrf_failure")],
            "Why does the CSRF failure view return a forbidden response?",
            "debugging",
        ).selected_sources[0].text
        middleware_context = self.builder.build(
            [_source(process_view, symbol="process_view")],
            "How does CSRF token validation accept or reject a request?",
            "architecture",
        ).selected_sources[0].text

        self.assertIn("except TemplateDoesNotExist:", failure_context)
        self.assertIn("raise", failure_context)
        self.assertIn("return HttpResponseForbidden(body)", failure_context)
        self.assertIn("self._check_token(request)", middleware_context)
        self.assertIn("return self._reject(request, exc.reason)", middleware_context)
        self.assertIn("return self._accept(request)", middleware_context)

    def test_display_snippet_reuses_query_relevant_anchor(self):
        filler = "\n".join(f"    value_{index} = {index}" for index in range(80))
        content = f"def execute(job):\n{filler}\n    result = dispatch_job(job)\n    return result"

        snippet = self.builder.build_display_snippet(
            _source(content, symbol="execute"),
            question="Where does execute dispatch the job?",
        )

        self.assertIn("dispatch_job(job)", snippet["code"])
        self.assertLessEqual(len(snippet["code"].splitlines()), 20)

    def test_document_context_selects_complete_matching_heading_section(self):
        introduction = "\n".join(
            f"General project background paragraph {index}." for index in range(70)
        )
        content = f"""# Project guide
{introduction}

## Authentication settings
Set `AUTH_ISSUER` to the trusted token issuer.
Set `AUTH_AUDIENCE` to the expected API audience.

## Deployment
Restart the service after changing deployment settings.
"""

        selected = self.builder.build(
            [
                _source(
                    content,
                    symbol="guide.md:1",
                    file_path="docs/guide.md",
                    language="text",
                    symbol_type="fallback_chunk",
                )
            ],
            "Where are authentication issuer settings configured?",
            "configuration",
        ).selected_sources[0]

        self.assertIn("## Authentication settings", selected.text)
        self.assertIn("Set `AUTH_ISSUER`", selected.text)
        self.assertIn("[omitted lines", selected.text)
        self.assertLessEqual(selected.used_chars, 1500)


class FaithfulnessContextTests(unittest.TestCase):
    def test_faithfulness_uses_exact_generation_context(self):
        class FakeRagSystem:
            def __init__(self):
                self.user_prompt = ""

            def _generate_markdown_response(self, system_prompt, user_prompt):
                self.user_prompt = user_prompt
                return "0.75", "stop"

        rag_system = FakeRagSystem()
        exact_context = "[Source 1]\nLines shown: 1, 90-92\nlate_return_branch()"

        with patch("evals.run_eval.ENABLE_FAITHFULNESS", True):
            score = judge_faithfulness(
                rag_system,
                "How does it finish?",
                "It uses the late return branch. [1]",
                exact_context,
            )

        self.assertEqual(score, 0.75)
        self.assertIn(exact_context, rag_system.user_prompt)


if __name__ == "__main__":
    unittest.main()
