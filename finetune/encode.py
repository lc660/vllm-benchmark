"""Encode (input, instruction, target) triplets the way QwenImage21Pipeline does at evaluation, and cache them.

What the frozen parts of the pipeline compute from a sample never changes during fine-tuning, so it is computed
once (Qwen3-VL prompt embeddings with the condition image, VAE latents of the condition image and of the target)
and the training loop only runs the transformer.

Exactly as `QwenImage21Pipeline.__call__` with `height, width = input size` (how visionbench calls it):
  * the condition image is converted to RGBA and resized to `calculate_dimensions(output_resolution ** 2, aspect)`
    (about 1024 x 1024 pixels, whatever the input size); that one resize feeds both Qwen3-VL and the VAE;
  * the prompt is encoded with that image (`encode_prompt(image=[...])`), giving embeddings and the image-pad mask;
  * condition latents are the VAE posterior mode, normalised with the VAE's latents_mean / latents_std;
  * the target is generated at the input size, so the target image is encoded at that size (mode, normalised).

Only the text rows of the prompt embeddings are stored: the transformer overwrites the rows at the image-pad positions
with the condition latents (`joint_hidden_states[:, image_pad_mask] = hidden_states`), so their values never matter and
`expand` puts zeros there. That cuts a sample from ~10 MB to ~1-3 MB (tests/test_finetune.py checks the equivalence).

Cache: one safetensors file per sample under encode.cache_dir, keyed by a hash of everything that changes the
encoding (model revision, output resolution, instruction, input and target pixels). Changing a finetune_data
option (prompt, depth encoding, ...) therefore re-encodes only the affected samples.

    python -m finetune.encode shared_lora                 # everything the run will use (train order + val)
    python -m finetune.encode shared_lora --shard 0/4     # one of 4 parallel workers (one per GPU)
"""

from __future__ import annotations

import argparse
import hashlib
import sys
import time
from pathlib import Path
from typing import Any

from finetune import config as cfg

STORED_KEYS = ("text_embeds", "image_pad_mask", "cond_latents", "target_latents")
SAMPLE_KEYS = ("prompt_embeds", "image_pad_mask", "cond_latents", "target_latents")


def compact(prompt_embeds: Any, image_pad_mask: Any) -> Any:
    """(1, L, D) prompt embeddings -> (1, L_text, D): drop the rows the transformer overwrites."""
    return prompt_embeds[:, ~image_pad_mask[0].bool()]


def expand(stored: dict) -> dict:
    """Stored (compact) tensors -> the sample the transformer takes, zeros at the image-pad rows."""
    mask = stored["image_pad_mask"].bool()
    text = stored["text_embeds"]
    prompt = text.new_zeros(text.shape[0], mask.shape[1], text.shape[2])
    prompt[:, ~mask[0]] = text
    return {"prompt_embeds": prompt, "image_pad_mask": mask,
            "cond_latents": stored["cond_latents"], "target_latents": stored["target_latents"]}


def sample_key(item: dict, model: dict) -> str:
    h = hashlib.sha256()
    for part in (model["id"], model["revision"], str(model["output_resolution"]), item["task"], item["source"],
                 item["id"], item["instruction"]):
        h.update(part.encode("utf-8"))
        h.update(b"\0")
    for image in (item["input"], item["target"]):
        h.update(f"{image.mode}{image.size}".encode())
        h.update(image.tobytes())
    return h.hexdigest()[:40]


def cache_path(cache_dir: Path, key: str) -> Path:
    return cache_dir / key[:2] / f"{key}.safetensors"


class SampleEncoder:
    """Loads Qwen3-VL + VAE (no transformer) as a QwenImage21Pipeline and encodes samples like its __call__."""

    def __init__(self, model: dict, encode: dict, device: str = "cuda"):
        import torch
        from accelerate import cpu_offload
        from diffusers import AutoencoderKLQwenImage21, QwenImage21Pipeline
        from transformers import Qwen3VLForConditionalGeneration

        self.torch = torch
        self.model = model
        self.device = device
        common = {"revision": model["revision"], "cache_dir": str(cfg.repo_path(model["cache_dir"]))}
        dtypes = {"bf16": torch.bfloat16, "fp16": torch.float16, "fp32": torch.float32}
        text_encoder = Qwen3VLForConditionalGeneration.from_pretrained(
            model["id"], subfolder="text_encoder", dtype=torch.bfloat16, low_cpu_mem_usage=True, **common)
        vae = AutoencoderKLQwenImage21.from_pretrained(
            model["id"], subfolder="vae", dtype=dtypes[encode.get("vae_precision", "bf16")], **common)
        self.pipe = QwenImage21Pipeline.from_pretrained(
            model["id"], transformer=None, text_encoder=text_encoder, vae=vae, **common)
        text_encoder.requires_grad_(False)
        vae.requires_grad_(False)
        if encode.get("text_encoder_cpu_offload") and str(device).startswith("cuda"):
            cpu_offload(text_encoder, execution_device=device)
        else:
            text_encoder.to(device)
        vae.to(device)
        self.text_encoder = text_encoder
        self.vae = vae

    def condition_size(self, size: tuple[int, int]) -> tuple[int, int]:
        from diffusers.pipelines.qwenimage21.pipeline_qwenimage21 import calculate_dimensions

        res = int(self.model["output_resolution"])
        width, height, _ = calculate_dimensions(res * res, size[0] / size[1])
        return width, height

    def encode(self, item: dict) -> dict:
        torch = self.torch
        pipe = self.pipe
        image, target = item["input"], item["target"]
        if image.size != target.size:
            raise ValueError(f"{item['id']}: input {image.size} and target {target.size} differ")
        width, height = target.size
        if width % 32 or height % 32:
            raise ValueError(f"{item['id']}: size {target.size} is not a multiple of 32")
        with torch.no_grad():
            cond = image if image.mode == "RGBA" else image.convert("RGBA")
            cond_width, cond_height = self.condition_size(cond.size)
            cond_for_text = pipe.image_processor.resize(cond, width=cond_width, height=cond_height)
            cond_pixels = pipe.image_processor.preprocess(cond, width=cond_width, height=cond_height).unsqueeze(2)
            prompt_embeds, prompt_mask, image_pad_mask = pipe.encode_prompt(
                prompt=item["instruction"], image=[cond_for_text], device=self.device)
            if prompt_mask is not None and not bool(prompt_mask.all()):
                raise ValueError("a single prompt should never be padded")
            dtype = self.vae.dtype
            cond_latents = pipe._encode_vae_image(cond_pixels.to(self.device, dtype), generator=None)
            tgt = target if target.mode == "RGBA" else target.convert("RGBA")
            tgt_pixels = pipe.image_processor.preprocess(tgt, width=width, height=height).unsqueeze(2)
            target_latents = pipe._encode_vae_image(tgt_pixels.to(self.device, dtype), generator=None)
        slots = cond_latents.shape[-2] * cond_latents.shape[-1] // 4
        if int(image_pad_mask[0].sum()) != slots:
            raise ValueError(f"{item['id']}: {int(image_pad_mask[0].sum())} image slots in the prompt, "
                             f"{slots} needed by the condition latents")
        image_pad_mask = image_pad_mask.bool().cpu()
        return {
            "text_embeds": compact(prompt_embeds.to(torch.bfloat16).cpu(), image_pad_mask),
            "image_pad_mask": image_pad_mask,
            "cond_latents": cond_latents.to(torch.bfloat16).cpu(),
            "target_latents": target_latents.to(torch.bfloat16).cpu(),
        }

    def close(self) -> None:
        self.pipe = self.text_encoder = self.vae = None
        if self.torch.cuda.is_available():
            self.torch.cuda.empty_cache()


def save_sample(path: Path, tensors: dict) -> None:
    from safetensors.torch import save_file

    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    save_file({k: tensors[k].contiguous() for k in STORED_KEYS}, str(tmp))
    tmp.replace(path)  # atomic: a killed worker never leaves a truncated file behind


def load_sample(path: Path) -> dict:
    from safetensors.torch import load_file

    return load_file(str(path))


def planned_indices(config: dict) -> tuple[Any, list[int], Any, list[int]]:
    """(train dataset, the exact train order of the run, val dataset, val indices). The order is fixed by the
    seed, so precomputing and training agree on which samples are needed."""
    from finetune_data.dataset import FinetuneDataset

    train = FinetuneDataset(cfg.dataset_config(config, "train"))
    t = config["train"]
    order = train.sample_indices(t["max_steps"] * t["grad_accum"] * t["batch_size"], seed=t["seed"])
    val = FinetuneDataset(cfg.dataset_config(config, "val"))
    val_idx = val_indices(val, int(config["data"]["val_per_group"]))
    return train, order, val, val_idx


def val_indices(val: Any, per_group: int) -> list[int]:
    """The first `per_group` val items of every (task, source) group, in store order (deterministic)."""
    seen: dict[tuple[str, str], int] = {}
    chosen = []
    for i, (source, _row, task) in enumerate(val.items):
        if seen.get((task, source), 0) < per_group:
            seen[(task, source)] = seen.get((task, source), 0) + 1
            chosen.append(i)
    return chosen


class SampleSource:
    """Gives encoded samples by dataset index: from the disk cache (mode 'cache') or encoded on the fly."""

    def __init__(self, dataset: Any, config: dict, encoder: SampleEncoder | None = None):
        self.dataset = dataset
        self.config = config
        self.cache_dir = cfg.repo_path(config["encode"]["cache_dir"])
        self.mode = config["encode"]["mode"]
        self.encoder = encoder
        self._keys: dict[int, str] = {}

    def item(self, index: int) -> dict:
        return self.dataset[index]

    def __call__(self, index: int) -> tuple[dict, dict]:
        """(metadata: task / source / id, the sample's tensors on CPU, as `predict` takes them)."""
        item = self.dataset[index]
        meta = {k: item[k] for k in ("task", "source", "id")}
        if self.mode == "online":
            if self.encoder is None:
                raise RuntimeError("encode.mode online needs an encoder")
            return meta, expand(self.encoder.encode(item))
        key = self._keys.get(index) or sample_key(item, self.config["model"])
        self._keys[index] = key
        path = cache_path(self.cache_dir, key)
        if not path.is_file():
            raise FileNotFoundError(
                f"{item['id']} ({item['task']}) is not in the encode cache {self.cache_dir}. Run "
                f"`python -m finetune.encode <config>` with the same config first, or set encode.mode: online")
        return meta, expand(load_sample(path))


def precompute(config: dict, shard: tuple[int, int] = (0, 1), limit: int | None = None) -> dict:
    train, order, val, val_idx = planned_indices(config)
    jobs = [(train, i) for i in sorted(set(order))] + [(val, i) for i in val_idx]
    if limit:
        jobs = jobs[:limit]
    worker, workers = shard
    jobs = jobs[worker::workers]
    cache_dir = cfg.repo_path(config["encode"]["cache_dir"])
    todo = []
    for dataset, index in jobs:
        item = dataset[index]
        path = cache_path(cache_dir, sample_key(item, config["model"]))
        if not path.is_file():
            todo.append((item, path))
    print(f"[encode] {len(jobs)} samples in this shard ({len(set(order))} unique train of {len(train)}, "
          f"{len(val_idx)} val overall); {len(todo)} to encode, rest cached in {cache_dir}", flush=True)
    if not todo:
        return {"encoded": 0, "cached": len(jobs)}
    encoder = SampleEncoder(config["model"], config["encode"], device=config["train"]["device"])
    start, size = time.time(), 0
    for n, (item, path) in enumerate(todo, 1):
        save_sample(path, encoder.encode(item))
        size += path.stat().st_size
        if n % 50 == 0 or n == len(todo):
            rate = (time.time() - start) / n
            print(f"[encode] {n}/{len(todo)}  {rate:.2f} s/sample  {size / n / 2**20:.1f} MB/sample  "
                  f"ETA {(len(todo) - n) * rate / 60:.0f} min", flush=True)
    encoder.close()
    return {"encoded": len(todo), "cached": len(jobs) - len(todo), "bytes": size}


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("config", help="experiment YAML or a name in finetune/configs/")
    parser.add_argument("--set", action="append", default=[], metavar="KEY=VALUE", help="override a config key")
    parser.add_argument("--shard", default="0/1", help="i/n: encode every n-th sample starting at i")
    parser.add_argument("--limit", type=int, default=None, help="only the first N planned samples (debugging)")
    args = parser.parse_args(argv)
    config = cfg.load(args.config, args.set)
    worker, workers = (int(x) for x in args.shard.split("/"))
    if not 0 <= worker < workers:
        sys.exit("--shard must be i/n with 0 <= i < n")
    print(precompute(config, (worker, workers), args.limit))


if __name__ == "__main__":
    main()
