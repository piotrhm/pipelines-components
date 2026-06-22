from typing import Optional

from kfp import dsl
from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]


@dsl.component(
    base_image=AUTORAG_IMAGE,
    install_kfp_package=False,
)
def rag_templates_optimization(
    extracted_text: dsl.InputPath(dsl.Artifact),
    test_data: dsl.InputPath(dsl.Artifact),
    search_space_prep_report: dsl.InputPath(dsl.Artifact),
    rag_patterns: dsl.Output[dsl.Artifact],
    test_data_key: Optional[str],
    vector_io_provider_id: str,
    optimization_settings: Optional[dict] = None,
    input_data_key: Optional[str] = "",
):
    """RAG Templates Optimization component.

    Thin wrapper that delegates to
    ``ai4rag.components.optimization.run_rag_optimization``.

    Args:
        extracted_text: Path to extracted text documents.
        test_data: Path to benchmark test data JSON.
        search_space_prep_report: Path to the YAML search space report.
        rag_patterns: Output artifact for generated RAG patterns.
        test_data_key: Path to benchmark JSON in object storage.
        vector_io_provider_id: Vector I/O provider identifier in OGX.
        optimization_settings: Additional experiment settings.
        input_data_key: Path to documents dir within bucket.

    Environment variables (required):
        OGX_CLIENT_BASE_URL, OGX_CLIENT_API_KEY.
    """
    import logging
    import os
    from pathlib import Path

    from ai4rag.components.utils import create_ogx_client
    from ai4rag.components.optimization import run_rag_optimization
    from ai4rag.components.utils.compat import ensure_sqlite3

    ensure_sqlite3()
    logging.basicConfig(level=logging.INFO)

    ogx_client = create_ogx_client(
        base_url=os.environ["OGX_CLIENT_BASE_URL"],
        api_key=os.environ["OGX_CLIENT_API_KEY"],
    )

    output_dir = Path(rag_patterns.path)
    output_dir.mkdir(parents=True, exist_ok=True)

    result = run_rag_optimization(
        extracted_text_path=extracted_text,
        test_data_path=test_data,
        search_space_report_path=search_space_prep_report,
        output_dir=output_dir,
        ogx_client=ogx_client,
        vector_io_provider_id=vector_io_provider_id,
        test_data_key=test_data_key or "",
        input_data_key=input_data_key or "",
        optimization_settings=optimization_settings,
    )

    rag_patterns.metadata["name"] = "rag_patterns_artifact"
    rag_patterns.metadata["uri"] = rag_patterns.uri
    rag_patterns.metadata["metadata"] = {"patterns": result.patterns}


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        rag_templates_optimization,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
