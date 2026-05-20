# LoRA Eval Hub Pipeline

> **Stability: experimental** — This asset is under active development and may change.

## Overview

LoRA Training Pipeline with Eval Hub evaluation via KServe InferenceService.

A 4-stage ML pipeline for parameter-efficient fine-tuning:

1) Dataset Download - Prepares training data from HuggingFace, S3, or HTTP
2) LoRA Training - Fine-tunes using unsloth backend (low-rank adapters)
3) Evaluation - Benchmarks via Eval Hub with KServe model serving, results optionally tracked in MLflow
4) Model Registry - Registers trained model to Kubeflow Model Registry

For the minimal-config variant, see `lora_minimal_evalhub/`.

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
| `phase_01_dataset_man_data_uri` | `str` | `None` | [REQUIRED] Dataset location (hf://, s3://, https://) |
| `phase_02_train_man_train_batch` | `int` | `128` | Effective batch size (samples per optimizer step) |
| `phase_02_train_man_train_epochs` | `int` | `2` | Number of training epochs. LoRA typically needs 2-3 |
| `phase_02_train_man_train_gpu` | `int` | `1` | GPUs per worker |
| `phase_02_train_man_train_model` | `str` | `Qwen/Qwen2.5-1.5B-Instruct` | Base model (HuggingFace ID or path) |
| `phase_02_train_man_train_tokens` | `int` | `32000` | Max tokens per GPU (memory cap) |
| `phase_02_train_man_lora_r` | `int` | `16` | [LoRA] Rank of the low-rank matrices (4, 8, 16, 32, 64) |
| `phase_02_train_man_lora_alpha` | `int` | `32` | [LoRA] Scaling factor (typically 2x lora_r) |
| `phase_03_eval_opt_evalhub_url` | `str` | `""` | Eval Hub API endpoint URL (empty = skip evaluation) |
| `phase_03_eval_opt_collection` | `str` | `""` | Eval Hub collection ID: `leaderboard-v2`, `safety-and-fairness-v1`, or `toxicity-and-ethical-principles` |
| `phase_04_registry_man_address` | `str` | `""` | Model Registry address (empty = skip registration) |
| `phase_04_registry_man_reg_author` | `str` | `pipeline` | Author name for the registered model |
| `phase_04_registry_man_reg_name` | `str` | `lora-model` | Model name in registry |
| `phase_04_registry_man_reg_version` | `str` | `1.0.0` | Semantic version (major.minor.patch) |
| `phase_01_dataset_opt_subset` | `int` | `0` | Limit to first N examples (0 = all) |
| `phase_02_train_opt_annotations` | `str` | `""` | K8s annotations (key=val,...) |
| `phase_02_train_opt_cpu` | `str` | `4` | CPU cores per worker |
| `phase_02_train_opt_env_vars` | `str` | `""` | Env vars (KEY=VAL,...) |
| `phase_02_train_opt_labels` | `str` | `""` | K8s labels (key=val,...) |
| `phase_02_train_opt_learning_rate` | `float` | `0.0002` | Learning rate. 2e-4 recommended for LoRA |
| `phase_02_train_opt_lr_scheduler` | `str` | `cosine` | LR schedule (cosine, linear, constant) |
| `phase_02_train_opt_lr_warmup` | `int` | `0` | Warmup steps before full LR |
| `phase_02_train_opt_max_seq_len` | `int` | `8192` | Max sequence length in tokens |
| `phase_02_train_opt_memory` | `str` | `32Gi` | RAM per worker |
| `phase_02_train_opt_num_procs` | `str` | `auto` | Processes per worker ('auto' = one per GPU) |
| `phase_02_train_opt_save_epoch` | `bool` | `False` | Save checkpoint at each epoch |
| `phase_02_train_opt_seed` | `int` | `42` | Random seed for reproducibility |
| `phase_02_train_opt_use_liger` | `bool` | `True` | Enable Liger kernel optimizations |
| `phase_02_train_opt_lora_dropout` | `float` | `0.0` | [LoRA] Dropout rate for LoRA layers |
| `phase_02_train_opt_lora_target_modules` | `str` | `""` | [LoRA] Modules to apply LoRA (empty=auto-detect) |
| `phase_02_train_opt_lora_use_rslora` | `bool` | `False` | [LoRA] Use Rank-Stabilized LoRA |
| `phase_02_train_opt_lora_use_dora` | `bool` | `False` | [LoRA] Use Weight-Decomposed LoRA (DoRA) |
| `phase_02_train_opt_lora_load_in_4bit` | `bool` | `True` | [QLoRA] Enable 4-bit quantization |
| `phase_02_train_opt_lora_load_in_8bit` | `bool` | `False` | [QLoRA] Enable 8-bit quantization |
| `phase_02_train_opt_lora_sample_packing` | `bool` | `False` | [LoRA] Pack multiple samples for efficiency |
| `phase_02_train_opt_micro_batch_size` | `int` | `2` | Micro batch size per GPU |
| `phase_02_train_opt_grad_accum_steps` | `int` | `1` | Gradient accumulation steps |
| `phase_02_train_opt_flash_attention` | `bool` | `True` | Enable flash attention |
| `phase_02_train_opt_bf16` | `bool` | `True` | Use bfloat16 precision |
| `phase_02_train_opt_fp16` | `bool` | `False` | Use float16 precision |
| `phase_02_train_opt_tf32` | `bool` | `True` | Enable TF32 on Ampere+ GPUs |
| `phase_02_train_opt_save_steps` | `int` | `500` | Save checkpoint every N steps |
| `phase_02_train_opt_eval_steps` | `int` | `500` | Run evaluation every N steps |
| `phase_02_train_opt_logging_steps` | `int` | `10` | Log metrics every N steps |
| `phase_02_train_opt_save_total_limit` | `int` | `3` | Max checkpoints to keep |
| `phase_02_train_opt_wandb_project` | `str` | `""` | Weights & Biases project name |
| `phase_02_train_opt_wandb_entity` | `str` | `""` | Weights & Biases entity/team |
| `phase_02_train_opt_wandb_run_name` | `str` | `""` | Weights & Biases run name |
| `phase_02_train_opt_tensorboard_log_dir` | `str` | `""` | TensorBoard log directory |
| `phase_02_train_opt_dataset_type` | `str` | `""` | Dataset format type |
| `phase_02_train_opt_field_messages` | `str` | `""` | Field name for messages in dataset |
| `phase_02_train_opt_field_instruction` | `str` | `""` | Field name for instruction in dataset |
| `phase_02_train_opt_field_input` | `str` | `""` | Field name for input in dataset |
| `phase_02_train_opt_field_output` | `str` | `""` | Field name for output in dataset |
| `phase_02_train_opt_enable_model_splitting` | `bool` | `False` | Enable model splitting across GPUs |
| `phase_02_train_opt_runtime` | `str` | `training-hub` | Name of the ClusterTrainingRuntime to use |
| `phase_03_eval_opt_benchmarks` | `list` | 5 leaderboard benchmarks | Benchmarks to evaluate (ifeval, bbh, mmlu_pro, musr, math_hard) |
| `phase_03_eval_opt_mlflow_experiment` | `str` | `""` | MLflow experiment name (non-empty = enable tracking) |
| `phase_03_eval_opt_timeout` | `int` | `7200` | Max seconds to wait for evaluation |
| `phase_03_eval_opt_kserve_gpu_count` | `int` | `1` | GPUs for the KServe InferenceService predictor |
| `phase_03_eval_opt_kserve_cpu` | `str` | `"2"` | CPU for the KServe predictor |
| `phase_03_eval_opt_kserve_memory` | `str` | `"32Gi"` | Pod memory for the KServe predictor |
| `phase_04_registry_opt_description` | `str` | `""` | Model description |
| `phase_04_registry_opt_format_name` | `str` | `pytorch` | Model format (pytorch, onnx, tensorflow) |
| `phase_04_registry_opt_format_version` | `str` | `1.0` | Model format version |
| `phase_04_registry_opt_port` | `int` | `8080` | Model registry server port |

## Metadata

- **Name**: lora_evalhub_pipeline
- **Stability**: experimental
- **Dependencies**:
  - Kubeflow:
    - Name: Pipelines, Version: >=2.15.2
    - Name: Trainer, Version: >=0.1.0
  - External Services:
    - Name: Eval Hub, Version: >=0.1.0
    - Name: KServe, Version: >=0.11.0
    - Name: Kubernetes, Version: >=1.28.0
    - Name: Model Registry, Version: >=0.3.4
- **Tags**: training, fine_tuning, lora, peft, eval_hub, kserve, pipeline

## Additional Resources

- **Eval Hub**: [https://github.com/opendatahub-io/eval-hub](https://github.com/opendatahub-io/eval-hub)
- **Trainer**: [https://github.com/kubeflow/trainer](https://github.com/kubeflow/trainer)
