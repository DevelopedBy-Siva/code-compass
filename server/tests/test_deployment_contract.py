import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch


SERVER_ROOT = Path(__file__).resolve().parents[1]
if str(SERVER_ROOT) not in sys.path:
    sys.path.insert(0, str(SERVER_ROOT))

from server_app import app
from src.config import Settings


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


if __name__ == "__main__":
    unittest.main()
