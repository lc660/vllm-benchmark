"""Experiment configs: finetune/configs/default.yaml deep-merged with an experiment YAML and --set overrides."""

from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = Path(__file__).with_name("configs")
DEFAULT_CONFIG = CONFIG_DIR / "default.yaml"
TASKS = ("segmentation", "depth", "normals", "pose")


def deep_update(base: dict, new: dict) -> dict:
    for key, value in new.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            deep_update(base[key], value)
        else:
            base[key] = copy.deepcopy(value)
    return base


def parse_override(text: str) -> dict:
    """'train.lr=2e-4' -> {'train': {'lr': 2e-4}}; the value is parsed as YAML ('[depth]', 'true', 'null')."""
    if "=" not in text:
        raise ValueError(f"override {text!r} must look like key.subkey=value")
    key, value = text.split("=", 1)
    parts = [p for p in key.strip().split(".") if p]
    if not parts:
        raise ValueError(f"override {text!r} has an empty key")
    node: Any = yaml.safe_load(value) if value.strip() else None
    if isinstance(node, str):
        try:  # YAML 1.1 reads '2e-4' (no decimal point) as a string
            node = float(node)
        except ValueError:
            pass
    for part in reversed(parts):
        node = {part: node}
    return node


def resolve_config_path(value: str | Path) -> Path:
    """A YAML path, or a name in finetune/configs/ (e.g. 'shared_lora')."""
    path = Path(value)
    if path.is_file():
        return path.resolve()
    candidate = CONFIG_DIR / f"{path.stem if path.suffix in {'.yaml', '.yml'} else path.name}.yaml"
    if candidate.is_file():
        return candidate
    raise FileNotFoundError(f"no experiment config {value!r} (looked for a file and for {candidate})")


def load(experiment: str | Path | None = None, overrides: list[str] | None = None) -> dict:
    config = yaml.safe_load(DEFAULT_CONFIG.read_text(encoding="utf-8"))
    name = None
    if experiment is not None:
        path = resolve_config_path(experiment)
        deep_update(config, yaml.safe_load(path.read_text(encoding="utf-8")) or {})
        name = path.stem
    for text in overrides or []:
        deep_update(config, parse_override(text))
    config["name"] = config.get("name") or name or "run"
    validate(config)
    return config


def validate(config: dict) -> None:
    method, objective, train = config["method"], config["objective"], config["train"]
    if method["type"] not in {"lora", "dora", "lokr", "loha", "partial"}:
        raise ValueError(f"method.type {method['type']!r}: choose lora, dora, lokr, loha or partial")
    if objective["type"] not in {"flow", "onestep"}:
        raise ValueError(f"objective.type {objective['type']!r}: choose flow or onestep")
    if objective["timestep_sampling"] not in {"logit_normal", "uniform", "mode"}:
        raise ValueError(f"objective.timestep_sampling {objective['timestep_sampling']!r}")
    shift = objective.get("shift")
    if shift is not None and shift != "auto" and not (isinstance(shift, (int, float)) and shift > 0):
        raise ValueError("objective.shift must be null, 'auto' or a positive number")
    if config["model"]["transformer_precision"] not in {"bf16", "nf4"}:
        raise ValueError("model.transformer_precision must be bf16 or nf4")
    if config["model"]["transformer_precision"] == "nf4" and method["type"] not in {"lora", "dora"}:
        raise ValueError("an NF4 (QLoRA) base only supports method lora or dora")
    if config["encode"]["mode"] not in {"cache", "online"}:
        raise ValueError("encode.mode must be cache or online")
    for key in ("max_steps", "batch_size", "grad_accum", "val_every", "save_every", "log_every"):
        if not isinstance(train[key], int) or train[key] < 1:
            raise ValueError(f"train.{key} must be a positive integer")
    if train["optimizer"] not in {"adamw", "adamw8bit"}:
        raise ValueError("train.optimizer must be adamw or adamw8bit")
    if train["trainable_dtype"] not in {"fp32", "bf16"}:
        raise ValueError("train.trainable_dtype must be fp32 or bf16")
    for key in ("tasks",):
        values = config["data"].get(key)
        if values is not None and (not values or set(values) - set(TASKS)):
            raise ValueError(f"data.{key} must be a non-empty subset of {list(TASKS)}")


def dataset_config(config: dict, split: str) -> dict:
    """The finetune_data config (dict) for one split, with the task / source shortcuts applied."""
    from finetune_data.dataset import subset_config

    data = config["data"]
    base = copy.deepcopy(data.get("finetune_data") or {})
    base["splits"] = [split]
    root = Path(base.get("root", "data/finetune_canonical"))
    base["root"] = str(root if root.is_absolute() else REPO_ROOT / root)
    return subset_config(base, tasks=data.get("tasks"), sources=data.get("sources"))


def run_dir(config: dict) -> Path:
    root = Path(config["output_root"])
    return (root if root.is_absolute() else REPO_ROOT / root) / config["name"]


def repo_path(value: str | Path) -> Path:
    path = Path(value).expanduser()
    return path if path.is_absolute() else REPO_ROOT / path
