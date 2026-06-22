from kfp import dsl
from kfp_components.utils.consts import AUTORAG_IMAGE  # pyright: ignore[reportMissingImports]


@dsl.component(
    base_image=AUTORAG_IMAGE,  # noqa: E501
)
def test_data_loader(
    test_data_bucket_name: str,
    test_data_path: str,
    benchmark_sample_size: int = 25,
    test_data: dsl.Output[dsl.Artifact] = None,
):
    """Download test data JSON from S3 and sample it for benchmarking.

    Thin wrapper that delegates to ``ai4rag.components.data.load_test_data``.

    Args:
        test_data_bucket_name: S3 (or compatible) bucket that contains the test
            data file.
        test_data_path: S3 object key to the JSON test data file.
        benchmark_sample_size: Maximum number of records to keep from the test
            data. When the dataset exceeds this limit, a reproducible random
            sample is drawn (seed 42). Set to 0 to disable sampling and keep
            all records.
        test_data: Output artifact that receives the (possibly sampled) file.

    Environment variables (required when run with pipeline secret injection):
        AWS_ACCESS_KEY_ID, AWS_SECRET_ACCESS_KEY, AWS_S3_ENDPOINT.
        AWS_DEFAULT_REGION is optional.
    """
    import json
    import logging

    from ai4rag.components.utils.s3 import create_s3_client
    from ai4rag.components.data.test_data_loader import load_test_data

    logging.basicConfig(level=logging.INFO)

    s3_client = create_s3_client()

    result = load_test_data(
        bucket_name=test_data_bucket_name,
        key=test_data_path,
        benchmark_sample_size=benchmark_sample_size,
        s3_client=s3_client,
    )

    with open(test_data.path, "w", encoding="utf-8") as f:
        json.dump(result.data, f, indent=2, ensure_ascii=False)


if __name__ == "__main__":
    from kfp.compiler import Compiler

    Compiler().compile(
        test_data_loader,
        package_path=__file__.replace(".py", "_component.yaml"),
    )
