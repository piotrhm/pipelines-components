"""Tests for the leaderboard_evaluation thin wrapper component."""

from unittest import mock

import pytest

from ..component import leaderboard_evaluation


def _make_ai4rag_mocks():
    """Build mock modules for ai4rag.assets_generator."""
    mock_build = mock.MagicMock(name="build_leaderboard_html")

    mock_assets = mock.MagicMock()
    mock_assets.build_leaderboard_html = mock_build

    modules = {
        "ai4rag": mock.MagicMock(),
        "ai4rag.assets_generator": mock_assets,
    }
    return modules, mock_build


class TestLeaderboardEvaluationUnitTests:
    """Unit tests for the leaderboard_evaluation thin wrapper."""

    def test_component_function_exists(self):
        """Component factory exists and exposes python_func."""
        assert callable(leaderboard_evaluation)
        assert hasattr(leaderboard_evaluation, "python_func")

    def test_delegates_to_ai4rag_build_leaderboard_html(self, tmp_path):
        """Wrapper calls build_leaderboard_html with correct args."""
        modules, mock_build = _make_ai4rag_mocks()
        mock_build.return_value = "<html>leaderboard</html>"

        html_artifact = mock.MagicMock(path=str(tmp_path / "out.html"))

        with mock.patch.dict("sys.modules", modules):
            leaderboard_evaluation.python_func(
                rag_patterns=str(tmp_path / "patterns"),
                html_artifact=html_artifact,
                optimization_metric="answer_correctness",
            )

        mock_build.assert_called_once_with(
            patterns_dir=str(tmp_path / "patterns"),
            optimization_metric="answer_correctness",
        )

    def test_writes_html_to_artifact_path(self, tmp_path):
        """HTML content is written to the artifact path."""
        modules, mock_build = _make_ai4rag_mocks()
        expected_html = "<html><body>RAG Patterns Leaderboard</body></html>"
        mock_build.return_value = expected_html

        html_path = tmp_path / "leaderboard.html"
        html_artifact = mock.MagicMock(path=str(html_path))

        with mock.patch.dict("sys.modules", modules):
            leaderboard_evaluation.python_func(
                rag_patterns=str(tmp_path),
                html_artifact=html_artifact,
            )

        assert html_path.exists()
        assert html_path.read_text(encoding="utf-8") == expected_html

    def test_default_optimization_metric(self, tmp_path):
        """Default optimization_metric is 'faithfulness'."""
        modules, mock_build = _make_ai4rag_mocks()
        mock_build.return_value = "<html></html>"

        html_artifact = mock.MagicMock(path=str(tmp_path / "out.html"))

        with mock.patch.dict("sys.modules", modules):
            leaderboard_evaluation.python_func(
                rag_patterns=str(tmp_path),
                html_artifact=html_artifact,
            )

        assert mock_build.call_args.kwargs["optimization_metric"] == "faithfulness"

    def test_creates_parent_directory(self, tmp_path):
        """Parent directory for the HTML artifact is created if missing."""
        modules, mock_build = _make_ai4rag_mocks()
        mock_build.return_value = "<html></html>"

        nested_path = tmp_path / "nested" / "dir" / "out.html"
        html_artifact = mock.MagicMock(path=str(nested_path))

        with mock.patch.dict("sys.modules", modules):
            leaderboard_evaluation.python_func(
                rag_patterns=str(tmp_path),
                html_artifact=html_artifact,
            )

        assert nested_path.exists()

    def test_propagates_ai4rag_exception(self, tmp_path):
        """Exceptions from ai4rag are propagated to the caller."""
        modules, mock_build = _make_ai4rag_mocks()
        mock_build.side_effect = FileNotFoundError("rag_patterns path is not a directory")

        html_artifact = mock.MagicMock(path=str(tmp_path / "out.html"))

        with mock.patch.dict("sys.modules", modules):
            with pytest.raises(FileNotFoundError, match="rag_patterns path is not a directory"):
                leaderboard_evaluation.python_func(
                    rag_patterns=str(tmp_path / "missing"),
                    html_artifact=html_artifact,
                )
