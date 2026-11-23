"""Fine-tune the Qwen-Image 2.1 transformer on the union multi-task data (finetune_data/).

    python -m finetune.encode shared_lora          # once: cache the frozen encoders' outputs
    python -m finetune.train shared_lora           # train; resumes automatically from <run>/last
    python -m finetune.train shared_lora --set train.max_steps=20 --set name=debug   # any override

Writes to runs/<name>/:
    config.yaml        the fully merged config
    metrics.jsonl      train loss (overall and per task) and validation loss per task
    step-<n>/          adapter checkpoints (visionbench adapter format, load with model.options.adapter_path)
    best/              the checkpoint with the lowest task-averaged validation loss
    final/             the last step
    last/              resume state (adapter + optimizer + data position)

One GPU per run. To use several GPUs, run several methods / tasks in parallel (scripts/finetune_all.sh).
"""

from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import yaml

from finetune import config as cfg
from finetune import encode as enc
from finetune import methods, objectives
from visionbench.models import adapters

VAL_SIGMAS = (0.25, 0.5, 0.75)  # fixed timesteps for a low-variance flow validation loss


def load_transformer(model: dict, device: str) -> Any:
    import torch
    from diffusers import BitsAndBytesConfig, QwenImage21Transformer2DModel

    kwargs: dict[str, Any] = {"revision": model["revision"], "cache_dir": str(cfg.repo_path(model["cache_dir"])),
                              "dtype": torch.bfloat16, "low_cpu_mem_usage": True}
    if model["transformer_precision"] == "nf4":
        kwargs["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True)
        kwargs["device_map"] = {"": device}
    transformer = QwenImage21Transformer2DModel.from_pretrained(model["id"], subfolder="transformer", **kwargs)
    if model["transformer_precision"] == "nf4":
        from peft import prepare_model_for_kbit_training

        transformer = prepare_model_for_kbit_training(transformer, use_gradient_checkpointing=False)
    else:
        transformer.to(device)
    return transformer


def scheduler_config(model: dict) -> dict:
    """The sampler's shift settings (only needed for objective.shift: auto)."""
    from diffusers import FlowMatchEulerDiscreteScheduler

    sched = FlowMatchEulerDiscreteScheduler.from_pretrained(
        model["id"], subfolder="scheduler", revision=model["revision"], cache_dir=str(cfg.repo_path(model["cache_dir"])))
    return dict(sched.config)


def make_optimizer(params: list, train: dict) -> Any:
    import torch

    kwargs = {"lr": float(train["lr"]), "betas": tuple(train["betas"]), "weight_decay": float(train["weight_decay"])}
    if train["optimizer"] == "adamw8bit":
        import bitsandbytes as bnb

        return bnb.optim.AdamW8bit(params, **kwargs)
    return torch.optim.AdamW(params, **kwargs)


def to_device(tensors: dict, device: str) -> dict:
    return {k: v.to(device, non_blocking=True) for k, v in tensors.items()}


def collate(samples: list[dict]) -> dict:
    """Stack samples into one batch. Only valid when they share sizes and the image-pad layout (same prompt)."""
    import torch

    if len(samples) == 1:
        return samples[0]
    first = samples[0]
    for s in samples[1:]:
        for key in enc.SAMPLE_KEYS:
            if s[key].shape != first[key].shape:
                raise ValueError("train.batch_size > 1 needs samples of equal size and prompt length; "
                                 "use batch_size 1 with grad_accum instead")
        if not torch.equal(s["image_pad_mask"], first["image_pad_mask"]):
            raise ValueError("samples in a batch place the condition image differently; use batch_size 1")
    return {k: torch.cat([s[k] for s in samples], dim=0) for k in enc.SAMPLE_KEYS}


def validate(transformer: Any, source: enc.SampleSource, indices: list[int], objective: dict, device: str,
             sched_cfg: dict | None) -> dict:
    """Mean velocity MSE per task on fixed samples, timesteps and noise, so values are comparable across steps."""
    import torch

    transformer.eval()
    per_task: dict[str, list[float]] = defaultdict(list)
    with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda")):
        for index in indices:
            meta, tensors = source(index)
            sample = to_device(tensors, device)
            x0 = sample["target_latents"]
            if objective["type"] == "onestep":
                sigma_list, values = [1.0], []
            else:
                sigma_list, values = list(VAL_SIGMAS), []
            gen = torch.Generator().manual_seed(1000 + index)
            for sigma in sigma_list:
                noise = None
                if not (objective["type"] == "onestep" and objective.get("zero_noise", True)):
                    noise = torch.randn(x0.shape, generator=gen).to(device, x0.dtype)
                values.append(float(objectives.loss(transformer, sample, objective, noise=noise,
                                                    sigmas=torch.tensor([sigma])).mean()))
            per_task[meta["task"]].append(sum(values) / len(values))
    transformer.train()
    result = {f"val/{task}": sum(v) / len(v) for task, v in per_task.items()}
    result["val/mean"] = sum(result.values()) / len(result)  # task-balanced, the selection criterion
    return result


def save_checkpoint(transformer: Any, path: Path, saved: dict, config: dict, step: int, extra: dict | None = None,
                    save_dtype: str = "bf16") -> None:
    inference = ({"num_inference_steps": 1, "init_noise": "zeros" if config["objective"].get("zero_noise", True)
                  else "gaussian"} if config["objective"]["type"] == "onestep"
                 else {"num_inference_steps": 40, "init_noise": "gaussian"})
    tmp = path.with_name(path.name + ".tmp")
    if tmp.exists():
        shutil.rmtree(tmp)
    adapters.save(transformer, tmp, {
        **saved, "step": step, "inference": inference,
        "base_model": {"id": config["model"]["id"], "revision": config["model"]["revision"]},
        "trained_on_precision": config["model"]["transformer_precision"],
        "tasks": sorted(config["data"].get("tasks") or cfg.TASKS),
        "run": config["name"], "train_config": config, "save_dtype": save_dtype, **(extra or {}),
    })
    if path.exists():
        shutil.rmtree(path)
    tmp.replace(path)


def resume_state(run: Path) -> dict | None:
    state = run / "last" / "state.pt"
    if not state.is_file():
        return None
    import torch

    return torch.load(state, map_location="cpu", weights_only=False)


def train(config: dict, *, fresh: bool = False) -> Path:
    import torch
    from diffusers.optimization import get_scheduler

    run = cfg.run_dir(config)
    if fresh and run.exists():
        shutil.rmtree(run)
    run.mkdir(parents=True, exist_ok=True)
    previous = run / "config.yaml"
    if previous.is_file():
        old = yaml.safe_load(previous.read_text(encoding="utf-8"))
        if _comparable(old) != _comparable(config):
            raise SystemExit(f"{run} holds a run with a different config; pick a new name or pass --fresh")
    previous.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")

    t, device = config["train"], config["train"]["device"]
    random.seed(t["seed"])
    torch.manual_seed(t["seed"])
    if device.startswith("cuda"):
        torch.backends.cuda.matmul.allow_tf32 = True

    train_set, order, val_set, val_idx = enc.planned_indices(config)
    encoder = None
    if config["encode"]["mode"] == "online":
        encoder = enc.SampleEncoder(config["model"], config["encode"], device=device)
    train_source = enc.SampleSource(train_set, config, encoder)
    val_source = enc.SampleSource(val_set, config, encoder)
    counts = {f"{task}/{source}": n for (task, source), n in sorted(train_set.counts().items())}
    print(f"[train] {config['name']}: {len(train_set)} train items {counts}, {len(order)} draws, "
          f"{len(val_idx)} val items", flush=True)

    transformer = load_transformer(config["model"], device)
    trainable_dtype = torch.float32 if t["trainable_dtype"] == "fp32" else torch.bfloat16
    saved = methods.apply(transformer, config["method"], trainable_dtype)
    if t["gradient_checkpointing"]:
        transformer.enable_gradient_checkpointing()
    n_train, n_total = methods.count_parameters(transformer)
    print(f"[train] method {saved['method']}: {n_train / 1e6:.1f}M trainable of {n_total / 1e9:.2f}B", flush=True)
    sched_cfg = scheduler_config(config["model"]) if config["objective"].get("shift") == "auto" else None

    params = [p for p in transformer.parameters() if p.requires_grad]
    optimizer = make_optimizer(params, t)
    lr_scheduler = get_scheduler(t["lr_scheduler"], optimizer=optimizer, num_warmup_steps=t["warmup_steps"],
                                 num_training_steps=t["max_steps"])

    step, position, best = 0, 0, math.inf
    state = resume_state(run)
    if state is not None:
        _load_trainable(transformer, run / "last", saved)
        optimizer.load_state_dict(state["optimizer"])
        lr_scheduler.load_state_dict(state["lr_scheduler"])
        step, position, best = state["step"], state["position"], state["best"]
        torch.set_rng_state(state["torch_rng"])
        random.setstate(state["python_rng"])
        print(f"[train] resumed at step {step}", flush=True)

    metrics = (run / "metrics.jsonl").open("a", encoding="utf-8")
    window: dict[str, list[float]] = defaultdict(list)
    transformer.train()
    started = time.time()
    autocast = torch.autocast("cuda", dtype=torch.bfloat16, enabled=device.startswith("cuda"))
    micro = t["batch_size"]
    while step < t["max_steps"]:
        for _ in range(t["grad_accum"]):
            batch_idx = order[position:position + micro]
            position += micro
            metas, tensors = zip(*(train_source(i) for i in batch_idx))
            sample = to_device(collate(list(tensors)), device)
            with autocast:
                per_sample = objectives.loss(transformer, sample, config["objective"], scheduler_config=sched_cfg)
            (per_sample.mean() / t["grad_accum"]).backward()
            for meta, value in zip(metas, per_sample.detach().float().tolist()):
                window[meta["task"]].append(value)
                window["all"].append(value)
        grad_norm = torch.nn.utils.clip_grad_norm_(params, t["max_grad_norm"])
        optimizer.step()
        lr_scheduler.step()
        optimizer.zero_grad(set_to_none=True)
        step += 1

        if step % t["log_every"] == 0 or step == t["max_steps"]:
            row = {"step": step, "lr": lr_scheduler.get_last_lr()[0], "grad_norm": float(grad_norm),
                   "s_per_step": (time.time() - started) / t["log_every"],
                   **{f"train/{k}": sum(v) / len(v) for k, v in window.items()}}
            if device.startswith("cuda"):
                row["max_mem_gb"] = torch.cuda.max_memory_allocated() / 2**30
            print("[train] " + " ".join(f"{k}={v:.4g}" if isinstance(v, float) else f"{k}={v}"
                                        for k, v in row.items()), flush=True)
            metrics.write(json.dumps(row) + "\n")
            metrics.flush()
            window.clear()
            started = time.time()

        if step % t["val_every"] == 0 or step == t["max_steps"]:
            row = {"step": step, **validate(transformer, val_source, val_idx, config["objective"], device, sched_cfg)}
            print("[val] " + " ".join(f"{k}={v:.4g}" for k, v in row.items() if k != "step"), flush=True)
            metrics.write(json.dumps(row) + "\n")
            metrics.flush()
            if row["val/mean"] < best:
                best = row["val/mean"]
                save_checkpoint(transformer, run / "best", saved, config, step, {"val": row})

        if step % t["save_every"] == 0 or step == t["max_steps"]:
            save_checkpoint(transformer, run / f"step-{step}", saved, config, step)
            save_checkpoint(transformer, run / "last", saved, config, step, save_dtype="fp32")  # exact resume
            torch.save({"optimizer": optimizer.state_dict(), "lr_scheduler": lr_scheduler.state_dict(), "step": step,
                        "position": position, "best": best, "torch_rng": torch.get_rng_state(),
                        "python_rng": random.getstate()}, run / "last" / "state.pt")

    save_checkpoint(transformer, run / "final", saved, config, step)
    metrics.close()
    print(f"[train] done: {run / 'final'} (best val/mean {best:.4g} in {run / 'best'})", flush=True)
    return run


def _load_trainable(transformer: Any, path: Path, saved: dict) -> None:
    """Resume: load saved tensors into the already-built trainable layers (keeps them trainable)."""
    from safetensors.torch import load_file

    state = load_file(str(path / adapters.WEIGHTS_NAME))
    if saved["method"] in adapters.PEFT_METHODS:
        from peft import set_peft_model_state_dict

        set_peft_model_state_dict(transformer, state, adapter_name="default")
    else:
        transformer.load_state_dict(state, strict=False)


def _comparable(config: dict) -> dict:
    """Config keys that may change between a run and its resume (logging cadence, device) are ignored."""
    c = json.loads(json.dumps(config))
    for key in ("log_every", "val_every", "save_every", "device", "max_steps"):
        c["train"].pop(key, None)
    c["encode"].pop("text_encoder_cpu_offload", None)
    return c


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="experiment YAML or a name in finetune/configs/")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE",
                        help="override a config key, e.g. --set train.lr=2e-4 --set data.tasks=[depth]")
    parser.add_argument("--fresh", action="store_true", help="delete an existing run with this name first")
    parser.add_argument("--print-config", action="store_true", help="print the merged config and exit")
    args = parser.parse_args(argv)
    config = cfg.load(args.config, args.set)
    if args.print_config:
        print(yaml.safe_dump(config, sort_keys=False))
        return
    train(config, fresh=args.fresh)


if __name__ == "__main__":
    main()
