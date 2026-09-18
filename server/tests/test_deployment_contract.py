import os
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch


SERVER_ROOT = Path(__file__).resolve().parents[1]
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from server_app import app
from src.config import Settings
from src.repo_fetcher import RepoFetcher


class DeploymentContractTests(unittest.TestCase):
    def test_sagemaker_routes_are_registered(self):
        routes = {(route.path, method) for route in app.routes for method in route.methods}
        self.assertIn(("/ping", "GET"), routes)
        self.assertIn(("/ping", "POST"), routes)
        self.assertIn(("/invocations", "POST"), routes)

    def test_settings_validate_required_qdrant_url(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(RuntimeError, "QDRANT_URL is required"):
                Settings.from_env()

    def test_settings_are_environment_driven(self):
        with patch.dict(
            os.environ,
            {
                "QDRANT_URL": "https://qdrant.example",
                "CORS_ORIGINS": "https://demo.example,https://preview.example",
                "BEDROCK_MODEL_ID": "provider.model-v1",
            },
            clear=True,
        ):
            settings = Settings.from_env()
        self.assertEqual(settings.qdrant_url, "https://qdrant.example")
        self.assertEqual(settings.bedrock_model_id, "provider.model-v1")
        self.assertEqual(len(settings.cors_origins), 2)

    def test_direct_qdrant_key_does_not_call_secrets_manager(self):
        with patch.dict(
            os.environ,
            {
                "QDRANT_URL": "https://qdrant.example",
                "QDRANT_API_KEY": "direct-key",
                "QDRANT_API_KEY_SECRET_ARN": "not-an-arn",
            },
            clear=True,
        ), patch("src.config._secret_value") as secret_value:
            settings = Settings.from_env()

        self.assertEqual(settings.qdrant_api_key, "direct-key")
        secret_value.assert_not_called()

    def test_repo_fetcher_can_create_repositories_in_configured_cache(self):
        with tempfile.TemporaryDirectory() as directory:
            cache_dir = Path(directory) / "codecompass-repos"
            fetcher = RepoFetcher(base_dir=str(cache_dir))

            def fake_clone(command, capture_output, text):
                self.assertEqual(command[:3], ["git", "clone", "--depth"])
                target_dir = Path(command[-1])
                target_dir.mkdir(parents=True)
                (target_dir / "README.md").write_text("# cached\n", encoding="utf-8")
                return SimpleNamespace(returncode=0, stderr="", stdout="")

            with patch("src.repo_fetcher.subprocess.run", side_effect=fake_clone):
                repository = fetcher.clone_repository("https://github.com/example/project")

            self.assertEqual(fetcher.base_dir, cache_dir)
            self.assertTrue((cache_dir / "example-project" / "README.md").exists())
            self.assertEqual(repository["local_path"], str(cache_dir / "example-project"))


if __name__ == "__main__":
    unittest.main()
