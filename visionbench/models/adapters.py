"""Save and load fine-tuned weights for the Qwen-Image 2.1 transformer.

One on-disk format covers every fine-tuning method in `finetune/` so the evaluation backend needs a single
option (`adapter_path`) whatever method produced the checkpoint:

    <dir>/adapter_config.json       method, PEFT config or trained parameter prefixes, inference settings
    <dir>/adapter_model.safetensors the trained tensors only (LoRA / DoRA / LoKr / LoHa factors, or the
                                    unfrozen full weights for method "partial")

Methods:
    lora, dora, lokr, loha  PEFT adapters injected into the transformer's Linear layers
    partial                 plain weights of the unfrozen modules (e.g. the last N blocks), loaded over the base

`inference` in the config records how the checkpoint must be sampled (e.g. the one-step objective needs
`num_inference_steps: 1` and zero initial noise); the backend applies it unless the run config overrides it.

Heavy imports (torch, peft, safetensors) happen inside the functions, so importing this module is free.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

FORMAT = "visionbench-adapter-v1"
CONFIG_NAME = "adapter_config.json"
WEIGHTS_NAME = "adapter_model.safetensors"
PEFT_METHODS = {"lora", "dora", "lokr", "loha"}
METHODS = PEFT_METHODS | {"partial"}
# PEFT tuners that can wrap bitsandbytes 4/8-bit Linear layers. LoKr / LoHa and plain weights cannot.
QUANTIZATION_SAFE = {"lora", "dora"}


def read_config(path: str | Path) -> dict:
    path = Path(path)
    config_file = path / CONFIG_NAME
    if not config_file.is_file():
        raise FileNotFoundError(f"{config_file} not found: adapter_path must be a directory written by finetune.train")
    config = json.loads(config_file.read_text(encoding="utf-8"))
    if config.get("format") != FORMAT:
        raise ValueError(f"{config_file}: unknown adapter format {config.get('format')!r}, expected {FORMAT!r}")
    if config.get("method") not in METHODS:
        raise ValueError(f"{config_file}: unknown method {config.get('method')!r}; choose from {sorted(METHODS)}")
    return config


def peft_config(method: str, options: dict) -> Any:
    """The PEFT config object for a PEFT method, from the plain-dict options stored in adapter_config.json."""
    import peft

    if method in {"lora", "dora"}:
        return peft.LoraConfig(**{**options, "use_dora": method == "dora"})
    if method == "lokr":
        return peft.LoKrConfig(**options)
    if method == "loha":
        return peft.LoHaConfig(**options)
    raise ValueError(f"{method!r} is not a PEFT method")


def inject(transformer: Any, method: str, options: dict) -> None:
    """Add a fresh (untrained) PEFT adapter named 'default' to the transformer."""
    from peft import inject_adapter_in_model

    inject_adapter_in_model(peft_config(method, options), transformer, adapter_name="default")


def trainable_state_dict(transformer: Any, method: str, prefixes: list[str] | None = None) -> dict:
    """Only the tensors this method trains, with names that `load` understands."""
    if method in PEFT_METHODS:
        from peft.utils import get_peft_model_state_dict

        return get_peft_model_state_dict(transformer, adapter_name="default")
    if not prefixes:
        raise ValueError("method 'partial' needs the list of trained module prefixes")
    return {name: p for name, p in transformer.named_parameters() if name.startswith(tuple(prefixes))}


def save(transformer: Any, path: str | Path, config: dict) -> Path:
    """Write the trained tensors and the adapter config. `config` needs at least `method` and, for PEFT
    methods, `peft_config` (plain dict) or, for `partial`, `trainable_prefixes`."""
    import torch
    from safetensors.torch import save_file

    method = config["method"]
    if method not in METHODS:
        raise ValueError(f"unknown method {method!r}")
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    state = trainable_state_dict(transformer, method, config.get("trainable_prefixes"))
    if not state:
        raise ValueError("nothing to save: the transformer has no trained tensors for this method")
    dtype = torch.float32 if config.get("save_dtype") == "fp32" else torch.bfloat16
    state = {k: v.detach().to("cpu", dtype=dtype).contiguous() for k, v in state.items()}
    save_file(state, str(path / WEIGHTS_NAME), metadata={"format": FORMAT, "method": method})
    (path / CONFIG_NAME).write_text(json.dumps({"format": FORMAT, **config}, indent=2, sort_keys=True), encoding="utf-8")
    return path


def _scale_peft(transformer: Any, scale: float) -> None:
    if scale == 1.0:
        return
    touched = 0
    for module in transformer.modules():
        scaling = getattr(module, "scaling", None)
        if isinstance(scaling, dict) and "default" in scaling:
            scaling["default"] = scaling["default"] * scale
            touched += 1
    if not touched:
        raise ValueError("adapter_scale was set but no PEFT layer with a 'default' scaling was found")


def load(transformer: Any, path: str | Path, *, scale: float = 1.0, quantized: bool = False) -> dict:
    """Apply a saved adapter to a loaded base transformer in place. Returns the adapter config.

    `quantized` must be True when the transformer's Linear layers are bitsandbytes int8 / NF4: only LoRA and DoRA
    can be stacked on those, other methods need a bf16 / fp16 transformer."""
    config = read_config(path)
    method = config["method"]
    if quantized and method not in QUANTIZATION_SAFE:
        raise ValueError(
            f"a {method!r} checkpoint cannot be applied to an int8 / NF4 transformer; "
            "set transformer_precision to bf16 (profile: custom) for this run"
        )
    from safetensors.torch import load_file

    state = load_file(str(Path(path) / WEIGHTS_NAME))
    if method in PEFT_METHODS:
        from peft import set_peft_model_state_dict

        inject(transformer, method, config["peft_config"])
        result = set_peft_model_state_dict(transformer, state, adapter_name="default")
        unexpected = list(getattr(result, "unexpected_keys", None) or [])
        if unexpected:
            raise ValueError(f"adapter tensors that match no layer: {unexpected[:5]} ({len(unexpected)} total)")
        _scale_peft(transformer, scale)
    else:
        if scale != 1.0:
            raise ValueError("adapter_scale is only defined for PEFT adapters, not for method 'partial'")
        params = dict(transformer.named_parameters())
        missing = [k for k in state if k not in params]
        if missing:
            raise ValueError(f"checkpoint tensors not in the transformer: {missing[:5]} ({len(missing)} total)")
        bad_shape = [k for k, v in state.items() if tuple(params[k].shape) != tuple(v.shape)]
        if bad_shape:
            raise ValueError(f"shape mismatch for {bad_shape[:5]}: was the checkpoint trained on another model?")
        transformer.load_state_dict(state, strict=False)
    return config
