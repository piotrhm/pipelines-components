# SFT Minimal Eval Hub Pipeline

> **Stability: experimental** — This asset is under active development and may change.

## Overview

SFT Training Pipeline with Eval Hub evaluation (Minimal) via KServe InferenceService.

A minimal-config 4-stage pipeline that runs 5 leaderboard benchmarks (ifeval, bbh,
mmlu_pro, musr, math_hard) -- all public datasets, no HF token required.

1) Dataset Download - Prepares training data from HuggingFace, S3, or HTTP
2) SFT Training - Fine-tunes using instructlab-training backend
3) Evaluation - 5 hardcoded leaderboard benchmarks via Eval Hub + KServe
4) Model Registry - Registers trained model to Kubeflow Model Registry

For the full-featured variant with configurable benchmarks and collections, see `sft_evalhub/`.

## Prerequisites

1. **Eval Hub** installed on the cluster (operator + CR in the target namespace).
2. **KServe** available (included with RHOAI by default).
3. **RBAC** for the pipeline ServiceAccount:

   ```bash
   oc create role evalhub-kserve-role \
     --verb=create,delete,get,list,patch \
     --resource=inferenceservices.serving.kserve.io,servingruntimes.serving.kserve.io,pods,services,secrets \
     -n <namespace> && \
   oc create rolebinding evalhub-kserve-binding \
     --role=evalhub-kserve-role \
     --serviceaccount=<namespace>:<pipeline-sa> \
     -n <namespace>
   ```

4. **Workspace PVC** with `ReadWriteMany` access mode (NFS-backed).

## Inputs

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `phase_01_dataset_man_data_uri` | `str` | `None` | Dataset location (hf://, s3://, https://). |
| `phase_02_train_man_train_batch` | `int` | `128` | Effective batch size per optimizer step. |
| `phase_02_train_man_epochs` | `int` | `1` | Number of training epochs. |
| `phase_02_train_man_gpu` | `int` | `1` | GPUs per worker. |
| `phase_02_train_man_model` | `str` | `Qwen/Qwen2.5-1.5B-Instruct` | Base model (HuggingFace ID or path). |
| `phase_02_train_man_tokens` | `int` | `10000` | Max tokens per GPU (memory cap). |
| `phase_02_train_man_workers` | `int` | `4` | Number of training pods. |
| `phase_03_eval_opt_evalhub_url` | `str` | `""` | Eval Hub API endpoint URL (empty = skip evaluation). |
| `phase_03_eval_opt_mlflow_experiment` | `str` | `""` | MLflow experiment name (non-empty = enable tracking). |
| `phase_03_eval_opt_kserve_gpu_count` | `int` | `1` | GPUs for the KServe InferenceService predictor. |
| `phase_03_eval_opt_kserve_cpu` | `str` | `"2"` | CPU for the KServe predictor. |
| `phase_03_eval_opt_kserve_memory` | `str` | `"32Gi"` | Pod memory for the KServe predictor. |
| `phase_03_eval_opt_timeout` | `int` | `7200` | Max seconds to wait for evaluation. |
| `phase_04_registry_man_address` | `str` | `""` | Model Registry address (empty = skip). |
| `phase_04_registry_man_author` | `str` | `pipeline` | Author name for the registered model. |
| `phase_04_registry_man_name` | `str` | `sft-model` | Model name in registry. |
| `phase_04_registry_man_version` | `str` | `1.0.0` | Semantic version (major.minor.patch). |
| `phase_01_dataset_opt_subset` | `int` | `0` | Limit dataset to N samples (0 = all). |
| `phase_02_train_opt_env_vars` | `str` | `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True,...` | Environment variables as KEY=VAL,KEY=VAL. |
| `phase_02_train_opt_learning_rate` | `float` | `5e-06` | Learning rate for training. |
| `phase_02_train_opt_max_seq_len` | `int` | `8192` | Maximum sequence length in tokens. |
| `phase_02_train_opt_fsdp_sharding` | `str` | `FULL_SHARD` | FSDP strategy (FULL_SHARD, HYBRID_SHARD). |
| `phase_02_train_opt_use_liger` | `bool` | `False` | Enable Liger kernel optimizations. |
| `phase_02_train_opt_runtime` | `str` | `training-hub` | Name of the ClusterTrainingRuntime to use. |
| `phase_04_registry_opt_port` | `int` | `8080` | Model Registry server port. |

## Metadata

- **Name**: sft_minimal_evalhub_pipeline
- **Stability**: experimental
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
    - Name: Trainer, Version: >=0.1.0
  - External Services:
    - Name: Eval Hub, Version: >=0.1.0
    - Name: KServe, Version: >=0.11.0
    - Name: Kubernetes, Version: >=1.28.0
- **Tags**: training, fine_tuning, sft, eval_hub, kserve, minimal, pipeline

## Additional Resources

- **Eval Hub**: [https://github.com/opendatahub-io/eval-hub](https://github.com/opendatahub-io/eval-hub)
- **Trainer**: [https://github.com/kubeflow/trainer](https://github.com/kubeflow/trainer)
