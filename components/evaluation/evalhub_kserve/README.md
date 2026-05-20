# Eval Hub KServe Component

> **Stability: experimental** — This component is under active development and may change.

## Overview

A KFP component that evaluates a fine-tuned model using the
[Eval Hub](https://github.com/opendatahub-io/eval-hub) service with a
KServe InferenceService for model serving.

The component:

1. Creates a KServe **ServingRuntime** + **InferenceService** (matching the RHOAI
   dashboard deployment pattern) to serve the fine-tuned model from the workspace PVC.
2. Submits benchmark evaluation jobs to Eval Hub, pointing at the InferenceService URL.
3. Polls for evaluation completion and collects results/metrics.
4. Optionally logs metrics to **MLflow** (when `mlflow_experiment_name` is provided).
5. Cleans up both KServe resources after evaluation (or on failure).

Both KServe resources (ServingRuntime and InferenceService) are explicitly deleted
in a `finally` block after evaluation completes or on failure.

## Inputs

| Parameter | Type | Default | Description |
| --------- | ---- | ------- | ----------- |
| `evalhub_url` | `str` | `""` | Eval Hub API endpoint URL. Empty = skip evaluation entirely. |
| `benchmarks` | `list` | `[]` | Benchmark specs, e.g. `[{"id": "leaderboard_ifeval", "provider_id": "lm_evaluation_harness"}]`. |
| `collection_id` | `str` | `""` | Eval Hub collection ID (overrides `benchmarks`). Available: `leaderboard-v2`, `safety-and-fairness-v1`, `toxicity-and-ethical-principles`. |
| `pvc_mount_path` | `str` | `""` | Workspace PVC mount path (set by KFP workspace config). |
| `model_artifact` | `dsl.Input[dsl.Model]` | `None` | KFP Model artifact from training step. |
| `model_path` | `str` | `None` | HuggingFace model ID or local path (fallback if no artifact). |
| `evalhub_tenant` | `str` | `""` | Eval Hub tenant / X-Tenant header. |
| `evalhub_auth_token` | `str` | `""` | Bearer token for Eval Hub auth. |
| `evalhub_model_name` | `str` | `"finetuned-model"` | Display name for the model in Eval Hub. |
| `base_model_name` | `str` | `""` | HF model ID for tokenizer resolution (e.g. `Qwen/Qwen2.5-1.5B-Instruct`). |
| `evalhub_job_name` | `str` | `"pipeline-eval"` | Evaluation job name in Eval Hub. |
| `evalhub_timeout` | `int` | `7200` | Max seconds to wait for evaluation. |
| `evalhub_poll_interval` | `int` | `30` | Seconds between eval status polls. |
| `mlflow_experiment_name` | `str` | `""` | MLflow experiment name. Non-empty enables MLflow tracking; empty disables it. |
| `gpu_count` | `int` | `1` | Number of GPUs for the InferenceService predictor. |
| `memory` | `str` | `"8Gi"` | Pod memory request/limit for the predictor. |
| `cpu` | `str` | `"2"` | CPU request/limit for the predictor. |
| `runtime_image` | `str` | RHOAI vLLM image | Container image for the ServingRuntime. |
| `isvc_ready_timeout` | `int` | `600` | Max seconds to wait for InferenceService readiness. |

## Outputs

| Artifact | Type | Description |
| -------- | ---- | ----------- |
| `output_metrics` | `dsl.Metrics` | Evaluation scores as KFP metrics (logged per benchmark). |
| `output_results` | `dsl.Artifact` | Full evaluation results JSON from Eval Hub. |

## Prerequisites

1. **Eval Hub** installed on the cluster (operator + CR in the target namespace).

2. **KServe** available (included with RHOAI by default).

3. **RBAC** — the pipeline ServiceAccount needs permissions for KServe resources:

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

4. **Workspace PVC** must use `ReadWriteMany` access mode (NFS-backed) so the KServe
   predictor pod can mount the fine-tuned model.

5. **`kubernetes-credentials` Secret** containing `KUBERNETES_SERVER_URL` and
   `KUBERNETES_AUTH_TOKEN` for K8s API access from within the component.

## Known Limitations

- **`trust_remote_code`**: Some HuggingFace datasets used by benchmarks require
  `trust_remote_code=True`. The 5 default leaderboard benchmarks (ifeval, bbh,
  mmlu_pro, musr, math_hard) work without it. For other benchmarks, a custom
  provider ConfigMap with `HF_DATASETS_TRUST_REMOTE_CODE=1` must be created and
  referenced in the Eval Hub CR.

- **Tokenizer resolution**: Eval Hub's lm_eval adapter uses the served model name
  to download the tokenizer. Since the served model is a local fine-tuned checkpoint,
  the `base_model_name` parameter is used to resolve the correct HF tokenizer.

## Metadata

- **Name**: evalhub_kserve
- **Stability**: experimental
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
  - External Services:
    - Name: Eval Hub, Version: >=0.1.0
    - Name: KServe, Version: >=0.11.0
    - Name: vLLM (RHOAI), Version: >=0.6.0
- **Tags**:
  - evaluation
  - llm
  - eval_hub
  - kserve
  - benchmarks
  - metrics
  - mlflow

## Additional Resources

- **Eval Hub**: [https://github.com/opendatahub-io/eval-hub](https://github.com/opendatahub-io/eval-hub)
- **KServe**: [https://kserve.github.io/website/](https://kserve.github.io/website/)
