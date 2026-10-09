from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ragmini.config import load_env


class EnvConfigTests(unittest.TestCase):
    def test_loads_values_and_preserves_existing_environment(self) -> None:
        with tempfile.TemporaryDirectory() as directory:
            env_file = Path(directory) / ".env"
            env_file.write_text(
                "# comment\nOPENAI_API_KEY='file-key'\nexport RAG_MODEL=deepseek-flash\n",
                encoding="utf-8",
            )
            with patch.dict(os.environ, {"OPENAI_API_KEY": "shell-key"}, clear=True):
                load_env(env_file)
                self.assertEqual(os.environ["OPENAI_API_KEY"], "shell-key")
                self.assertEqual(os.environ["RAG_MODEL"], "deepseek-flash")


if __name__ == "__main__":
    unittest.main()
