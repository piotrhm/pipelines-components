from kfp import dsl
from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]


@dsl.component(
    base_image=AUTORAG_IMAGE,  # noqa: E501
)
def leaderboard_evaluation(
    rag_patterns: dsl.InputPath(dsl.Artifact),
    html_artifact: dsl.Output[dsl.HTML],
    optimization_metric: str = "faithfulness",
):
    """Build an HTML leaderboard artifact from RAG pattern evaluation results.

    Thin wrapper that delegates to ``ai4rag.assets_generator.build_leaderboard_html``.

    Args:
        rag_patterns: Path to the directory of RAG patterns; each subdir contains
            pattern.json (pattern_name, indexing_params, rag_params, scores,
            execution_time, final_score).
        html_artifact: Output HTML artifact; the leaderboard table is written to
            html_artifact.path (single file).
        optimization_metric: Name of the metric used to rank patterns (e.g. faithfulness,
            answer_correctness, context_correctness). Defaults to "faithfulness".
    """
    import logging
    from pathlib import Path

    from ai4rag.assets_generator import build_leaderboard_html

    logging.basicConfig(level=logging.INFO)

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
