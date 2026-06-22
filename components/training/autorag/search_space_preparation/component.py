from typing import List, Optional

from kfp import dsl
from kfp.compiler import Compiler
from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]


@dsl.component(
    base_image=AUTORAG_IMAGE,  # noqa: E501
)
def search_space_preparation(
    test_data: dsl.Input[dsl.Artifact],
    extracted_text: dsl.Input[dsl.Artifact],
    search_space_prep_report: dsl.Output[dsl.Artifact],
    embedding_models: Optional[List] = None,
    generation_models: Optional[List] = None,
    metric: str = None,
):
    """Search space preparation for AutoRAG experiments.

    Thin wrapper that delegates to
    ``ai4rag.components.optimization.prepare_search_space_report``.

    Args:
        test_data: Input artifact with benchmark questions and expected answers.
        extracted_text: Input artifact with extracted text documents.
        search_space_prep_report: Output artifact for the YAML search space report.
        embedding_models: List of embedding model identifiers to try.
        generation_models: List of generation model identifiers to try.
        metric: Quality metric for evaluation (e.g. "faithfulness").

    Environment variables (required):
        OGX_CLIENT_BASE_URL, OGX_CLIENT_API_KEY.
    """
    import logging
    import os

    from ai4rag.components.optimization.search_space_preparation import prepare_search_space_report
    from ai4rag.components.utils.ogx_client import create_ogx_client
    from ai4rag.utils.compat import ensure_sqlite3

    ensure_sqlite3()
    logging.basicConfig(level=logging.INFO)

    ogx_client = create_ogx_client(
        base_url=os.environ["OGX_CLIENT_BASE_URL"],
        api_key=os.environ["OGX_CLIENT_API_KEY"],
    )

    report = prepare_search_space_report(
        test_data_path=test_data.path,
        extracted_text_path=extracted_text.path,
        ogx_client=ogx_client,
        embedding_models=embedding_models,
        generation_models=generation_models,
        metric=metric or "faithfulness",
    )

    report.save_yaml(search_space_prep_report.path)


if __name__ == "__main__":
    Compiler().compile(
        search_space_preparation,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
