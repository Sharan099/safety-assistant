"""Reranker kind resolution — catches HF secret misconfiguration."""

import importlib
import os
import unittest
from unittest.mock import patch


class TestRerankerKind(unittest.TestCase):
    def _reload_reranker(self):
        import backend.app.retrieval.reranker as mod

        return importlib.reload(mod)

    def test_auto_detects_jina_from_model_name(self):
        with patch.dict(
            os.environ,
            {"RERANKER_KIND": "auto", "RERANKER_MODEL": "jinaai/jina-reranker-v3"},
            clear=False,
        ):
            mod = self._reload_reranker()
            self.assertEqual(mod._resolve_kind(), "jina")

    def test_auto_detects_crossencoder_for_bge(self):
        with patch.dict(
            os.environ,
            {"RERANKER_KIND": "auto", "RERANKER_MODEL": "BAAI/bge-reranker-v2-m3"},
            clear=False,
        ):
            mod = self._reload_reranker()
            self.assertEqual(mod._resolve_kind(), "crossencoder")

    def test_model_id_in_kind_falls_back_to_auto(self):
        """HF users sometimes paste the model name into RERANKER_KIND."""
        with patch.dict(
            os.environ,
            {
                "RERANKER_KIND": "baai/bge-reranker-v2-m3",
                "RERANKER_MODEL": "jinaai/jina-reranker-v3",
            },
            clear=False,
        ):
            mod = self._reload_reranker()
            self.assertEqual(mod._resolve_kind(), "jina")


if __name__ == "__main__":
    unittest.main()
