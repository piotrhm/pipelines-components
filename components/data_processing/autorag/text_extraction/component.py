from typing import Optional

from kfp import dsl
from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]


@dsl.component(
    base_image=AUTORAG_IMAGE,  # noqa: E501
)
def text_extraction(
    documents_descriptor: dsl.Input[dsl.Artifact],
    extracted_text: dsl.Output[dsl.Artifact],
    error_tolerance: Optional[float] = None,
    max_extraction_workers: Optional[int] = None,
):
    """Text Extraction component.

    Thin wrapper that delegates to ``ai4rag.components.data.extract_text``.

    Args:
        documents_descriptor: Input artifact containing
            documents_descriptor.json with bucket, prefix, and documents list.
        extracted_text: Output artifact directory where DoclingDocument JSON files
            will be written.
        error_tolerance: Fraction of documents (0.0-1.0) allowed to fail without
            raising an error. None (the default) means zero tolerance.
        max_extraction_workers: Number of parallel worker processes used for text
            extraction. Defaults to 4. Set to None to use all available CPU cores.
    """
    import json
    import logging
    import os
    from pathlib import Path

    from ai4rag.components.data.text_extraction import extract_text

    logging.basicConfig(level=logging.INFO)

    descriptor_path = Path(documents_descriptor.path) / "documents_descriptor.json"
    with open(descriptor_path, "r", encoding="utf-8") as f:
        descriptor = json.load(f)

    output_dir = Path(extracted_text.path)
    output_dir.mkdir(parents=True, exist_ok=True)

    extract_text(
        documents=descriptor["documents"],
        bucket=descriptor["bucket"],
        output_dir=output_dir,
        s3_endpoint=os.environ.get("AWS_S3_ENDPOINT"),
        s3_access_key=os.environ.get("AWS_ACCESS_KEY_ID"),
        s3_secret_key=os.environ.get("AWS_SECRET_ACCESS_KEY"),
        s3_region=os.environ.get("AWS_DEFAULT_REGION"),
        error_tolerance=0.2,
        max_extraction_workers=max_extraction_workers,
        docling_artifacts_path=os.environ.get("DOCLING_ARTIFACTS_PATH"),
    )


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        text_extraction,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
