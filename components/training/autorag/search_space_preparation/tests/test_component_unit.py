"""Tests for the search_space_preparation thin wrapper component."""

import inspect
from unittest import mock

import pytest

from ..component import search_space_preparation

MOCKED_ENV_VARIABLES = {
    "OGX_CLIENT_BASE_URL": "https://ogx.example.com",
    "OGX_CLIENT_API_KEY": "test-api-key",
}


def _make_ai4rag_mocks():
    """Build mock modules for ai4rag search_space_preparation, ogx_client, and compat."""
    mock_create_ogx_client = mock.MagicMock(name="create_ogx_client")
    mock_prepare_report = mock.MagicMock(name="prepare_search_space_report")
    mock_ensure_sqlite3 = mock.MagicMock(name="ensure_sqlite3")

    mock_ogx_module = mock.MagicMock()
    mock_ogx_module.create_ogx_client = mock_create_ogx_client

    mock_report_module = mock.MagicMock()
    mock_report_module.prepare_search_space_report = mock_prepare_report

    mock_compat = mock.MagicMock()
    mock_compat.ensure_sqlite3 = mock_ensure_sqlite3

    modules = {
        "ai4rag": mock.MagicMock(),
        "ai4rag.components": mock.MagicMock(),
        "ai4rag.components.utils": mock.MagicMock(),
        "ai4rag.components.utils.ogx_client": mock_ogx_module,
        "ai4rag.components.optimization": mock.MagicMock(),
        "ai4rag.components.optimization.search_space_preparation": mock_report_module,
        "ai4rag.utils": mock.MagicMock(),
        "ai4rag.utils.compat": mock_compat,
    }
    return modules, mock_create_ogx_client, mock_prepare_report, mock_ensure_sqlite3


class TestSearchSpacePreparationUnitTests:
    """Unit tests for the search_space_preparation thin wrapper."""

    def test_component_function_exists(self):
        """Component factory exists and exposes python_func."""
        assert callable(search_space_preparation)
        assert hasattr(search_space_preparation, "python_func")

    def test_component_has_expected_interface(self):
        """Component has expected parameters."""
        sig = inspect.signature(search_space_preparation.python_func)
        params = list(sig.parameters)
        assert "test_data" in params
        assert "extracted_text" in params
        assert "search_space_prep_report" in params
        assert "embedding_models" in params
        assert "generation_models" in params
        assert "metric" in params

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_delegates_to_ai4rag_prepare_search_space_report(self, tmp_path):
        """Wrapper calls ensure_sqlite3, create_ogx_client, and prepare_search_space_report."""
        modules, mock_create_ogx, mock_prepare, mock_sqlite = _make_ai4rag_mocks()
        mock_ogx_client = mock.MagicMock(name="ogx_client_instance")
        mock_create_ogx.return_value = mock_ogx_client
        mock_report = mock.MagicMock()
        mock_prepare.return_value = mock_report

        test_data = mock.MagicMock()
        test_data.path = str(tmp_path / "test_data.json")
        extracted_text = mock.MagicMock()
        extracted_text.path = str(tmp_path / "extracted")
        report_artifact = mock.MagicMock()
        report_artifact.path = str(tmp_path / "report.yml")

        with mock.patch.dict("sys.modules", modules):
            search_space_preparation.python_func(
                test_data=test_data,
                extracted_text=extracted_text,
                search_space_prep_report=report_artifact,
                embedding_models=["embed-1", "embed-2"],
                generation_models=["gen-1"],
                metric="answer_correctness",
            )

        mock_sqlite.assert_called_once()
        mock_create_ogx.assert_called_once_with(
            base_url="https://ogx.example.com",
            api_key="test-api-key",
        )
        mock_prepare.assert_called_once_with(
            test_data_path=str(tmp_path / "test_data.json"),
            extracted_text_path=str(tmp_path / "extracted"),
            ogx_client=mock_ogx_client,
            embedding_models=["embed-1", "embed-2"],
            generation_models=["gen-1"],
            metric="answer_correctness",
        )
        mock_report.save_yaml.assert_called_once_with(str(tmp_path / "report.yml"))

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_default_metric_is_faithfulness(self, tmp_path):
        """When metric is None, 'faithfulness' is passed as default."""
        modules, mock_create_ogx, mock_prepare, _ = _make_ai4rag_mocks()
        mock_create_ogx.return_value = mock.MagicMock()
        mock_prepare.return_value = mock.MagicMock()

        test_data = mock.MagicMock()
        test_data.path = str(tmp_path / "test.json")
        extracted_text = mock.MagicMock()
        extracted_text.path = str(tmp_path / "ext")
        report = mock.MagicMock()
        report.path = str(tmp_path / "report.yml")

        with mock.patch.dict("sys.modules", modules):
            search_space_preparation.python_func(
                test_data=test_data,
                extracted_text=extracted_text,
                search_space_prep_report=report,
            )

        assert mock_prepare.call_args.kwargs["metric"] == "faithfulness"

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_none_models_passed_through(self, tmp_path):
        """None embedding_models and generation_models are forwarded as None."""
        modules, mock_create_ogx, mock_prepare, _ = _make_ai4rag_mocks()
        mock_create_ogx.return_value = mock.MagicMock()
        mock_prepare.return_value = mock.MagicMock()

        test_data = mock.MagicMock()
        test_data.path = str(tmp_path / "test.json")
        extracted_text = mock.MagicMock()
        extracted_text.path = str(tmp_path / "ext")
        report = mock.MagicMock()
        report.path = str(tmp_path / "report.yml")

        with mock.patch.dict("sys.modules", modules):
            search_space_preparation.python_func(
                test_data=test_data,
                extracted_text=extracted_text,
                search_space_prep_report=report,
                embedding_models=None,
                generation_models=None,
            )

        call_kwargs = mock_prepare.call_args.kwargs
        assert call_kwargs["embedding_models"] is None
        assert call_kwargs["generation_models"] is None

    def test_missing_ogx_env_raises_key_error(self, tmp_path):
        """Missing OGX env vars raise KeyError."""
        modules, _, _, _ = _make_ai4rag_mocks()

        test_data = mock.MagicMock()
        test_data.path = str(tmp_path / "test.json")
        extracted_text = mock.MagicMock()
        extracted_text.path = str(tmp_path / "ext")
        report = mock.MagicMock()
        report.path = str(tmp_path / "report.yml")

        with mock.patch.dict("os.environ", {}, clear=True):
            with mock.patch.dict("sys.modules", modules):
                with pytest.raises(KeyError):
                    search_space_preparation.python_func(
                        test_data=test_data,
                        extracted_text=extracted_text,
                        search_space_prep_report=report,
                    )

    @mock.patch.dict("os.environ", MOCKED_ENV_VARIABLES, clear=True)
    def test_propagates_ai4rag_exception(self, tmp_path):
        """Exceptions from ai4rag are propagated to the caller."""
        modules, mock_create_ogx, mock_prepare, _ = _make_ai4rag_mocks()
        mock_create_ogx.return_value = mock.MagicMock()
        mock_prepare.side_effect = ValueError("Metric not_real is not supported")

        test_data = mock.MagicMock()
        test_data.path = str(tmp_path / "test.json")
        extracted_text = mock.MagicMock()
        extracted_text.path = str(tmp_path / "ext")
        report = mock.MagicMock()
        report.path = str(tmp_path / "report.yml")

        with mock.patch.dict("sys.modules", modules):
            with pytest.raises(ValueError, match="Metric not_real is not supported"):
                search_space_preparation.python_func(
                    test_data=test_data,
                    extracted_text=extracted_text,
                    search_space_prep_report=report,
                    metric="not_real",
                )


class TestSearchSpaceReport:
    """Tests for YAML report content and ModelsPreSelector limiting."""

    @staticmethod
    def _make_ogx_with_chat(response_text: str = "en"):
        """Return ogx module stub whose chat completion returns *response_text*."""
        ogx_mod = _make_ogx_client_module()
        mock_model = mock.MagicMock()
        mock_model.identifier = "test-llm"
        mock_model.model_type = "llm"
        mock_models_response = mock.MagicMock()
        mock_models_response.data = [mock_model]
        mock_choice = mock.MagicMock()
        mock_choice.message.content = response_text
        mock_chat_response = mock.MagicMock()
        mock_chat_response.choices = [mock_choice]
        mock_client = mock.MagicMock()
        mock_client.models.list.return_value = mock_models_response
        mock_client.chat.completions.create.return_value = mock_chat_response
        ogx_mod.OgxClient.return_value = mock_client
        return ogx_mod

    def _run_with_search_space(self, tmp_path, mock_search_space, **run_kwargs):
        """Run component to completion and return captured YAML report dict."""
        import pandas as pd

        mocks = _make_all_mocks()

        benchmark_df = pd.DataFrame(
            [
                {
                    "question": "What is X?",
                    "correct_answers": [["Answer"]],
                    "correct_answer_document_ids": [["doc1"]],
                }
            ]
        )
        mock_pd = mock.MagicMock()
        mock_pd.read_json.return_value = benchmark_df
        mock_pd.DataFrame = pd.DataFrame
        mocks["pandas"] = mock_pd

        mocks["ogx_client"] = self._make_ogx_with_chat()

        mocks[
            "ai4rag.search_space.prepare.prepare_search_space"
        ].prepare_search_space_with_ogx.return_value = mock_search_space

        captured = {}

        def fake_safe_dump(data, stream):
            captured.update(data)

        mock_yaml = mock.MagicMock()
        mock_yaml.safe_dump = fake_safe_dump
        mocks["yaml"] = mock_yaml

        extracted_dir = tmp_path / "extracted"
        extracted_dir.mkdir()
        (extracted_dir / "doc0.md").write_text("sample text", encoding="utf-8")

        test_data_art = mock.MagicMock(path=str(tmp_path / "test_data.json"))
        extracted_art = mock.MagicMock(path=str(extracted_dir))
        report_art = mock.MagicMock(path=str(tmp_path / "report.yml"))

        with (
            mock.patch.dict(sys.modules, mocks),
            mock.patch.dict(
                "os.environ",
                {"OGX_CLIENT_BASE_URL": "https://ogx.example.com", "OGX_CLIENT_API_KEY": "test-api-key"},
            ),
        ):
            search_space_preparation.python_func(
                test_data=test_data_art,
                extracted_text=extracted_art,
                search_space_prep_report=report_art,
                **run_kwargs,
            )

        return captured, mocks

    @staticmethod
    def _make_search_space(*, n_generation: int = 1, n_embedding: int = 1):
        """Build a mock AI4RAGSearchSpace with chunking/retrieval params."""
        mock_foundation = mock.MagicMock()
        mock_foundation.values = [f"gen-{i}" for i in range(n_generation)]
        mock_embedding = mock.MagicMock()
        mock_embedding.values = [f"emb-{i}" for i in range(n_embedding)]
        mock_chunking = mock.MagicMock()
        mock_chunking.all_values.return_value = [
            {"method": "recursive", "chunk_size": 512, "chunk_overlap": 64},
        ]
        mock_retrieval = mock.MagicMock()
        mock_retrieval.all_values.return_value = [
            {"method": "simple", "number_of_chunks": 5, "search_mode": "vector"},
        ]

        mock_search_space = mock.MagicMock()
        mock_search_space.__getitem__ = mock.MagicMock(
            side_effect=lambda key: {
                "foundation_model": mock_foundation,
                "embedding_model": mock_embedding,
            }[key]
        )
        mock_search_space._search_space = {
            "foundation_model": mock_foundation,
            "embedding_model": mock_embedding,
            "chunking": mock_chunking,
            "retrieval": mock_retrieval,
        }
        return mock_search_space

    def test_report_includes_chunking_and_retrieval_blocks(self, tmp_path):
        """Report YAML contains chunking and retrieval search-space parameters."""
        mock_search_space = self._make_search_space()
        captured, _ = self._run_with_search_space(tmp_path, mock_search_space)

        assert captured["chunking"] == [{"method": "recursive", "chunk_size": 512, "chunk_overlap": 64}]
        assert captured["retrieval"] == [{"method": "simple", "number_of_chunks": 5, "search_mode": "vector"}]
        assert captured["foundation_model"] == ["gen-0"]
        assert captured["embedding_model"] == ["emb-0"]

    def test_models_preselector_limits_long_model_lists(self, tmp_path):
        """When model lists exceed TOP_N/TOP_K, ModelsPreSelector reduces selections."""
        mock_search_space = self._make_search_space(n_generation=5, n_embedding=4)

        import pandas as pd

        mocks = _make_all_mocks()
        benchmark_df = pd.DataFrame(
            [
                {
                    "question": "What is X?",
                    "correct_answers": [["Answer"]],
                    "correct_answer_document_ids": [["doc1"]],
                }
            ]
        )
        mock_pd = mock.MagicMock()
        mock_pd.read_json.return_value = benchmark_df
        mock_pd.DataFrame = pd.DataFrame
        mocks["pandas"] = mock_pd
        mocks["ogx_client"] = self._make_ogx_with_chat()
        mocks[
            "ai4rag.search_space.prepare.prepare_search_space"
        ].prepare_search_space_with_ogx.return_value = mock_search_space

        selected = {
            "foundation_models": ["gen-0", "gen-1", "gen-2"],
            "embedding_models": ["emb-0", "emb-1"],
        }
        mps_instance = mocks["ai4rag.core.experiment.mps"].ModelsPreSelector.return_value
        mps_instance.select_models.return_value = selected

        captured = {}

        def fake_safe_dump(data, stream):
            captured.update(data)

        mock_yaml = mock.MagicMock()
        mock_yaml.safe_dump = fake_safe_dump
        mocks["yaml"] = mock_yaml

        extracted_dir = tmp_path / "extracted"
        extracted_dir.mkdir()
        (extracted_dir / "doc0.md").write_text("sample text", encoding="utf-8")

        with (
            mock.patch.dict(sys.modules, mocks),
            mock.patch.dict(
                "os.environ",
                {"OGX_CLIENT_BASE_URL": "https://ogx.example.com", "OGX_CLIENT_API_KEY": "test-api-key"},
            ),
        ):
            search_space_preparation.python_func(
                test_data=mock.MagicMock(path=str(tmp_path / "test_data.json")),
                extracted_text=mock.MagicMock(path=str(extracted_dir)),
                search_space_prep_report=mock.MagicMock(path=str(tmp_path / "report.yml")),
                generation_models=[f"gen-{i}" for i in range(5)],
                embedding_models=[f"emb-{i}" for i in range(4)],
            )

        mps_cls = mocks["ai4rag.core.experiment.mps"].ModelsPreSelector
        mps_cls.assert_called_once()
        mps_instance.evaluate_patterns.assert_called_once()
        mps_instance.select_models.assert_called_once_with(n_embedding_models=2, n_foundation_models=3)

        assert captured["foundation_model"] == selected["foundation_models"]
        assert captured["embedding_model"] == selected["embedding_models"]
        assert len(captured["foundation_model"]) == 3
        assert len(captured["embedding_model"]) == 2

    def test_notebook_style_call_without_component_status(self):
        """Direct python_func calls without component_status work (notebook usage)."""
        import inspect

        # Verify component signature accepts component_status=None
        sig = inspect.signature(search_space_preparation.python_func)
        param = sig.parameters["component_status"]

        # Verify it has a default value of None
        assert param.default is None, "component_status should default to None for notebook usage"

        # This proves the component can be called without component_status
        # Full integration test would require real OGX/ai4rag dependencies
