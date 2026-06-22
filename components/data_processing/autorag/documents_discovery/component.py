from kfp import dsl
from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]


@dsl.component(
    base_image=AUTORAG_IMAGE,  # noqa: E501
)
def documents_discovery(
    input_data_bucket_name: str,
    input_data_path: str = "",
    test_data: dsl.Input[dsl.Artifact] = None,
    sampling_enabled: bool = True,
    sampling_max_size: float = 1,
    discovered_documents: dsl.Output[dsl.Artifact] = None,
):
    """Documents discovery component.

    Thin wrapper that delegates to ``ai4rag.components.data.discover_documents``.

    Args:
        input_data_bucket_name: S3 (or compatible) bucket containing input data.
        input_data_path: Path to folder with input documents within the bucket.
        test_data: Optional input artifact containing test data for sampling.
        sampling_enabled: Whether to enable sampling or not.
        sampling_max_size: Maximum size of sampled documents (in gigabytes).
        discovered_documents: Output artifact containing the documents descriptor JSON file.

    Environment variables (required when run with pipeline secret injection):
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_ENDPOINT.
        AWS_DEFAULT_REGION is optional.
    """
    import json
    import logging
    from pathlib import Path

    from ai4rag.components.utils.s3 import create_s3_client
    from ai4rag.components.data.documents_discovery import discover_documents

    logging.basicConfig(level=logging.INFO)

    test_data_doc_names = None
    if test_data:
        with open(test_data.path, "r", encoding="utf-8") as f:
            records = json.load(f)
        test_data_doc_names = list(
            {doc_id for r in records for doc_id in r.get("correct_answer_document_ids", [])}
        )

    s3_client = create_s3_client()

    result = discover_documents(
        bucket_name=input_data_bucket_name,
        prefix=input_data_path,
        test_data_doc_names=test_data_doc_names,
        sampling_enabled=sampling_enabled,
        sampling_max_size_gb=sampling_max_size,
        s3_client=s3_client,
    )

    output_dir = Path(discovered_documents.path)
    output_dir.mkdir(parents=True, exist_ok=True)
    result.save(path=output_dir, filename="documents_descriptor.json")


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        documents_discovery,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
