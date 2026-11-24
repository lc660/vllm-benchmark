# Fine-tuning Qwen-Image 2.1 (Direction 3)

This folder fine-tunes the Qwen-Image 2.1 editor on the multi-task data in `finetune_data/`, using several methods.
It also lets the evaluation configs score the checkpoints with the same prompts, decoders and metrics as the
zero-shot baseline.

Contents: [1 Methods](#1-methods) · [2 Quick start](#2-quick-start) · [3 Experiment plan](#3-experiment-plan) ·
[4 Configuration](#4-configuration) · [5 Evaluation](#5-evaluation) · [6 How training matches evaluation](#6-how-training-matches-evaluation) ·
[7 Status and known limits](#7-status-and-known-limits) · [8 Code map](#8-code-map)

## 1. Methods

All methods train the 7B diffusion transformer only. The Qwen3-VL text encoder and the VAE stay frozen, and their
outputs are cached once. Trainable parameter counts are computed from the layer shapes.

| Config | Method | What is trained | Trainable | Why it is in the matrix |
|---|---|---|---|---|
| `shared_lora` | LoRA r64 | low-rank update on attention + MLP, all 32 blocks | 168M | **main run**: one adapter for all tasks, chosen by the instruction |
| `lora_<task>` | LoRA r64 | same, but trained on one task only | 168M | **ablation**: does mixing tasks help or interfere? |
| `lora_attn` | LoRA r16 | attention projections only (diffusers default) | 17M | capacity: is a small adapter enough? |
| `dora` | DoRA r64 | LoRA + learned per-channel magnitude | ~170M | same rank as `shared_lora`, isolates the decomposition |
| `lokr` | LoKr | Kronecker-factored, **full-rank** update | 27M | full rank for few parameters; often strongest on big DiTs |
| `loha` | LoHa r32 | Hadamard product of two rank-32 updates on attention + MLP, all 32 blocks | 168M | same parameters as `shared_lora`, but the update can reach rank 1024 |
| `partial_ft` | partial full FT | full weights of the last 4 blocks + output head | 906M | "few full weights at the end" vs "low rank everywhere" |
| `qlora` | QLoRA | `shared_lora` on a 4-bit NF4 base | 168M | the recipe on 24 GB GPUs (A10G / g5) |
| `onestep_lora` | LoRA r64, one-step objective | as `shared_lora`, with sigma fixed at 1 and zero noise | 168M | deterministic one-step predictor (E2E-FT / Lotus); 40x cheaper eval |

The **objective** is independent of the method (`objective.type`):

- `flow` (default): the model's own rectified-flow loss. The model predicts the velocity `noise - x0` at a random
  noise level (logit-normal, as SD3). Eval samples with 40 steps, like the baseline.
- `onestep`: the noise level is always 1 and the starting latents are 0. One Euler step then maps the image straight
  to the target, so the output is deterministic. Diffusion-based dense predictors (E2E-FT, Lotus) found this
  works better than multi-step sampling. It should also remove the colour and value-range drift between images
  that the report notes for the zero-shot editor. Eval must use 1 step and zero initial latents. The checkpoint
  records this, and `make_eval_configs` sets it.

## 2. Quick start

```bash
uv lock && uv sync --extra finetune          # adds peft + safetensors (the lock file needs regenerating once)
bash scripts/build_finetune_data.sh          # the training data (finetune_data/README.md), if not built yet

# 1. Smoke test (~minutes): 20 steps on NYUv2 depth. Run this first on a new machine.
uv run --extra finetune python -m finetune.encode smoke
uv run --extra finetune python -m finetune.train smoke --fresh
uv run --extra finetune pytest tests/test_finetune.py     # also runs the tiny-model tests once torch/peft exist

# 2. One experiment
uv run --extra finetune python -m finetune.encode shared_lora     # cache frozen-encoder outputs (resumable)
uv run --extra finetune python -m finetune.train shared_lora      # train (resumes from runs/shared_lora/last)

# 3. Score it with the eval configs
uv run python -m finetune.make_eval_configs runs/shared_lora/best   # -> configs/finetuned/shared_lora-best/
bash scripts/run_eval_configs.sh configs/finetuned/shared_lora-best

# Or everything: all experiments queued over all GPUs, one run per GPU (logs in runs/<name>/log.txt)
bash scripts/finetune_all.sh
```

Any config key can be overridden on the command line, e.g.
`python -m finetune.train shared_lora --set name=lora_lr2e-4 --set train.lr=2e-4`. Give the run a new `name`. The
trainer refuses to resume a run whose config changed.

The encode step can run on several GPUs at once: `--shard 0/4` … `--shard 3/4`. Runs share the cache, so a sample
used by several experiments is encoded once.

## 3. Experiment plan

Suggested order. Each step is useful on its own.

1. **`smoke`**: check that the loop runs. Its log shows `s_per_step` and `max_mem_gb`; use them to size the real runs.
2. **`lora_depth`**: the cheapest real signal. Zero-shot VKITTI2 depth is inverted in 76% of images (AbsRel 0.281).
   Fine-tuning should fix this, so it validates the whole pipeline end to end, eval included.
3. **`shared_lora`**: the main result, on all test sets.
4. **Per-task ablation** `lora_segmentation`, `lora_depth`, `lora_normals`, `lora_pose`, compared with
   `shared_lora` per test set.
5. **Method comparison** at the same data and steps: `lora_attn`, `dora`, `lokr`, `loha`, `partial_ft`, `onestep_lora`
   (and `qlora` if only 24 GB GPUs are available).

For every checkpoint, also report the pose-preservation check (`coco_pose_qwen`), which `make_eval_configs` always
adds. It shows how much editing ability each method keeps, a question the report raises.

Expected large gains, from the zero-shot analysis in `finetune_data/README.md`:

- NYUv2 segmentation with all 40 classes listed (7.2 mIoU zero-shot, against 32.8 with only the present classes).
- VKITTI2 depth direction (outdoor scenes painted inverted).

## 4. Configuration

`finetune/configs/default.yaml` lists every key with a comment. Each experiment YAML holds only what differs from
the defaults. The keys you are most likely to touch:

| Key | Default | Notes |
|---|---|---|
| `method.type` / `rank` / `alpha` / `targets` | lora / 64 / 64 / attention + MLP | PEFT scale is `alpha / rank` |
| `objective.type` | flow | `onestep`: see section 1 |
| `objective.shift` | null | `auto` trains at the noise levels the sampler actually uses for this image size |
| `data.tasks`, `data.sources` | all | shortcuts for per-task / per-source runs |
| `data.finetune_data` | {} | any `finetune_data` option, e.g. `{tasks: {segmentation: {weight: 2.0}}}` |
| `train.max_steps` × `grad_accum` × `batch_size` | 3000 × 4 × 1 | 12,000 samples, about 0.8 passes over the ~15k images |
| `train.lr` | 1e-4 | 2e-4 for LoKr, 1e-5 for `partial_ft` (starting points, not tuned) |
| `train.val_every` / `save_every` | 250 / 500 | validation loss per task; `best/` = lowest task-averaged val loss |
| `encode.mode` | cache | `online` re-encodes every step: no disk, slower, needs the encoder in memory |

**Batch size.** Keep `batch_size: 1` and use `grad_accum`. Samples differ in image size and prompt length, and
the transformer needs every sample in a batch to share one image-token layout. The trainer raises an error
rather than mixing them.

**Disk.** The cache keeps only the text part of the prompt embeddings, plus the two latents: about 1–3 MB per
sample, so 20–40 GB for all ~15k training images. LoRA checkpoints are ~0.3 GB each (bf16). `partial_ft`
checkpoints are ~1.8 GB each, and its `last/` resume copy is 3.6 GB.

## 5. Evaluation

`make_eval_configs` copies each zero-shot config of the trained tasks and changes only:

- `generation_version` becomes `ft-<tag>-<original>`, so outputs never mix with the baseline's.
- `model.options.adapter_path` points to the checkpoint.
- The sampling options come from the checkpoint: `num_inference_steps` and `init_noise` (1 / zeros for one-step).

The data, prompts, decoders and metrics are unchanged.

The Qwen backend (`visionbench/models/qwen.py`) now loads adapters through three new model options:
`adapter_path`, `adapter_scale` (PEFT methods only), and `init_noise` (`gaussian` / `zeros`). It refuses to run a
checkpoint with different steps or noise than it was trained for, or on another base revision.

- **Quantized bases.** LoRA and DoRA run on the NF4 base that some baseline configs use (`vram_16gb`). LoKr,
  LoHa and `partial` cannot. For those, `make_eval_configs` switches the transformer to bf16 and notes it in the
  file. Rerun that config's zero-shot baseline in bf16 for a strict comparison (normals baselines were NF4).
- **Checkpoint selection.** `best/` is chosen by validation loss. To select by the real metric instead, score a
  few `step-<n>` checkpoints on the val split (no test data involved):
  `python -m finetune.make_eval_configs runs/shared_lora/step-1500 --split val`. Val configs exist for the
  in-domain NYUv2 / VKITTI2 segmentation and depth sets. Normals and pose val would need loaders for the store.

## 6. How training matches evaluation

The model is scored by running `QwenImage21Pipeline(image=x, prompt=t, height=H, width=W)` through visionbench.
`finetune/encode.py` reproduces what that call does to the inputs, step by step and with the pipeline's own
functions:

- The condition image is converted to RGBA and resized once to `calculate_dimensions(1024², aspect)`, which is
  about 1 megapixel whatever the input size. That same resized image feeds Qwen3-VL and the VAE.
- The prompt is encoded together with that image, giving the embeddings and the image-pad mask.
- The condition latents are the VAE posterior mode, normalised with the VAE's statistics.
- The target is encoded at the input size, since that is the size the pipeline generates.

Prompts are the `finetune_data` instructions. These are the eval prompts, and for normals they already include
the system prompt. The forward pass (`finetune/objectives.py`) is the same as the pipeline's and as diffusers'
official img2img LoRA trainer:

- condition latents first, then the noisy target;
- `img_mask` = the prompt's mask plus one slot per 2×2 target latents;
- the prediction is read from the tail of the joint sequence.

## 7. Status and known limits

- **Not yet run on a GPU.** The code was written against the pinned diffusers commit (`uv.lock`, fef717f) and PEFT
  main. In the authoring environment, the torch-free tests pass: configs, eval-config generation, and the
  backend's option checks. The tiny-model tests in `tests/test_finetune.py` need torch, diffusers and peft; run
  them first. They check, on the real transformer class, that:
  - every method trains, saves and reloads to identical outputs;
  - an untrained adapter leaves the model unchanged;
  - the forward layout matches the pipeline;
  - the cache shortcut is exact.
  Then run `smoke`.
- **No measured time or memory yet.** Read them off the `smoke` log (`s_per_step`, `max_mem_gb`) before sizing
  `max_steps`. `qlora` and `adamw8bit` are the levers if memory is short.
- **One GPU per run.** For several GPUs, run different experiments in parallel (`scripts/finetune_all.sh`), which
  is what the method comparison needs anyway.
- **Hyperparameters are starting points.** Learning rates and steps are not tuned. Only the per-task val loss in
  `metrics.jsonl` shows when a run has converged or is overfitting.

## 8. Code map

| Path | Purpose |
|---|---|
| `finetune/configs/default.yaml`, `*.yaml` | all options; one file per experiment |
| `finetune/config.py` | load / merge / validate configs, `--set` overrides |
| `finetune/methods.py` | LoRA / DoRA / LoKr / LoHa / partial: what is trained |
| `finetune/objectives.py` | flow and one-step losses, timestep sampling and shift, the transformer forward pass |
| `finetune/encode.py` | pipeline-exact encoding of samples, disk cache, `python -m finetune.encode` |
| `finetune/train.py` | the training loop, validation, checkpoints, resume |
| `finetune/make_eval_configs.py` | eval configs for a checkpoint (test or val split) |
| `visionbench/models/adapters.py` | checkpoint format; save / load for every method (used by training and eval) |
| `visionbench/models/qwen.py` | `adapter_path`, `adapter_scale`, `init_noise` model options |
| `scripts/finetune_all.sh`, `scripts/run_eval_configs.sh` | run the matrix; run a folder of eval configs |
| `tests/test_finetune.py` | config, eval-config and backend tests; tiny-model tests of every method |
