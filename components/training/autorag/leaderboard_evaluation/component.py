from pathlib import Path

from kfp import dsl
from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]

_AUTORAG_SHARED = Path(__file__).parents[1] / "shared"


@dsl.component(
    base_image=AUTORAG_IMAGE,  # noqa: E501
    embedded_artifact_path=str(_AUTORAG_SHARED / "component_status.py"),
    install_kfp_package=False,
)
def leaderboard_evaluation(
    rag_patterns: dsl.InputPath(dsl.Artifact),
    html_artifact: dsl.Output[dsl.HTML],
    embedded_artifact: dsl.EmbeddedInput[dsl.Dataset] = None,
    optimization_metric: str = "faithfulness",
    component_status: dsl.Output[dsl.Artifact] = None,
):
    """Build an HTML leaderboard artifact from RAG pattern evaluation results.

    Thin wrapper that delegates to ``ai4rag.assets_generator.build_leaderboard_html``.

    Args:
        rag_patterns: Path to the directory of RAG patterns; each subdir contains
            pattern.json (pattern_name, indexing_params, rag_params, scores,
            execution_time, final_score).
        html_artifact: Output HTML artifact; the leaderboard table is written to
            html_artifact.path (single file).
        component_status: Output artifact containing stage-level progress tracking.
        embedded_artifact: Embedded ``autorag.shared`` helpers injected by KFP at runtime.
        optimization_metric: Name of the metric used to rank patterns (e.g. faithfulness,
            answer_correctness, context_correctness). Defaults to "faithfulness".
    """
    import importlib.util
    import logging
    from pathlib import Path

    from ai4rag.assets_generator import build_leaderboard_html

    logging.basicConfig(level=logging.INFO)

    if component_status is None:
        from kfp_components.components.training.autorag.shared.component_status import (  # pyright: ignore[reportMissingImports]
            null_component_status_tracker,
        )

        status = null_component_status_tracker()
    else:
        _embedded_path = Path(embedded_artifact.path)
        _module_path = _embedded_path if _embedded_path.is_file() else _embedded_path / "component_status.py"
        _spec = importlib.util.spec_from_file_location("_autorag_component_status", _module_path)
        if _spec is None or _spec.loader is None:
            raise ValueError(f"Cannot load embedded module from {_module_path}")
        _status_module = importlib.util.module_from_spec(_spec)
        _spec.loader.exec_module(_status_module)
        status = _status_module.bootstrap_status_tracker(embedded_artifact, component_status, "leaderboard_evaluation")
    with status:
        if component_status is not None:
            status.set_metadata(display_name="Leaderboard Evaluation Status")
            component_status.metadata["display_name"] = "Leaderboard Evaluation Status"
        with status.stage("build_leaderboard"):
            html_content = build_leaderboard_html(
                patterns_dir=rag_patterns,
                optimization_metric=optimization_metric,
            )

            Path(html_artifact.path).parent.mkdir(parents=True, exist_ok=True)
            with open(html_artifact.path, "w", encoding="utf-8") as f:
                f.write(html_content)


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        leaderboard_evaluation,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
