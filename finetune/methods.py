"""Fine-tuning methods: which transformer weights are trained, and how.

    lora     low-rank update B @ A on the targeted Linear layers (Hu et al., 2021)
    dora     LoRA plus a learned per-output-channel magnitude (Liu et al., 2024)
    lokr     Kronecker-product update, near full rank for few parameters (LyCORIS)
    loha     Hadamard product of two low-rank updates (LyCORIS)
    partial  full weights of the last N transformer blocks (+ output head) unfrozen, everything else frozen

QLoRA is `lora` / `dora` on a 4-bit NF4 base (model.transformer_precision: nf4).
The objective (flow vs one-step) is independent of the method; see finetune/objectives.py.
"""

from __future__ import annotations

from typing import Any

from visionbench.models import adapters


def peft_options(method: dict) -> dict:
    """Plain-dict PEFT options (stored in adapter_config.json, so evaluation can rebuild the same layers)."""
    kind = method["type"]
    common = {"r": int(method["rank"]), "target_modules": list(method["targets"])}
    if kind in {"lora", "dora"}:
        return {**common, "lora_alpha": float(method["alpha"]), "lora_dropout": float(method.get("dropout", 0.0)),
                "init_lora_weights": "gaussian"}
    if kind == "lokr":
        return {**common, "alpha": float(method["alpha"]), **dict(method.get("lokr") or {})}
    if kind == "loha":
        return {**common, "alpha": float(method["alpha"])}
    raise ValueError(f"{kind!r} is not a PEFT method")


def partial_prefixes(transformer: Any, method: dict) -> list[str]:
    opts = method.get("partial") or {}
    num_blocks = len(transformer.transformer_blocks)
    last = int(opts.get("last_blocks", 0))
    if not 0 <= last <= num_blocks:
        raise ValueError(f"method.partial.last_blocks must be in [0, {num_blocks}]")
    prefixes = [f"transformer_blocks.{i}." for i in range(num_blocks - last, num_blocks)]
    prefixes += [f"{name}." for name in opts.get("modules") or []]
    names = {n for n, _ in transformer.named_parameters()}
    for prefix in prefixes:
        if not any(n.startswith(prefix) for n in names):
            raise ValueError(f"method.partial: no parameters under {prefix!r}")
    if not prefixes:
        raise ValueError("method.partial trains nothing: set last_blocks or modules")
    return prefixes


def apply(transformer: Any, method: dict, trainable_dtype: Any) -> dict:
    """Freeze the transformer, add / unfreeze what this method trains. Returns the adapter-config entries
    needed to save and reload it (method, peft_config or trainable_prefixes)."""
    transformer.requires_grad_(False)
    kind = method["type"]
    if kind in adapters.PEFT_METHODS:
        options = peft_options(method)
        adapters.inject(transformer, kind, options)
        saved = {"method": kind, "peft_config": options}
        for name, param in transformer.named_parameters():
            if _is_adapter_param(name):
                param.requires_grad_(True)
    else:
        prefixes = partial_prefixes(transformer, method)
        saved = {"method": "partial", "trainable_prefixes": prefixes}
        for name, param in transformer.named_parameters():
            if name.startswith(tuple(prefixes)):
                param.requires_grad_(True)
    for param in transformer.parameters():
        if param.requires_grad:
            param.data = param.data.to(trainable_dtype)
    if not any(p.requires_grad for p in transformer.parameters()):
        raise ValueError(f"method {kind!r} left no trainable parameter (check method.targets)")
    return saved


ADAPTER_TOKENS = ("lora_A", "lora_B", "lora_magnitude_vector", "lokr_", "hada_")


def _is_adapter_param(name: str) -> bool:
    return any(token in name for token in ADAPTER_TOKENS)


def count_parameters(transformer: Any) -> tuple[int, int]:
    trainable = sum(p.numel() for p in transformer.parameters() if p.requires_grad)
    total = sum(p.numel() for p in transformer.parameters())
    return trainable, total
