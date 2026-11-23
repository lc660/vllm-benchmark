"""Training objectives and the transformer forward pass on one encoded sample.

Both objectives use the model's own rectified-flow parameterisation, as the pipeline samples it:

    z_sigma = (1 - sigma) * x0 + sigma * noise,  the transformer predicts v = noise - x0,
    one Euler step from sigma to sigma' is  z' = z + (sigma' - sigma) * v.

flow     sigma ~ logit-normal (SD3), optionally shifted like the sampler; eval samples with 40 steps.
onestep  sigma = 1 always, noise = 0 by default, so z = 0 and the target is v = -x0: one Euler step from
         sigma 1 to 0 returns x0. The model becomes a deterministic one-step predictor (the E2E-FT / Lotus
         recipe for diffusion-based dense prediction); eval with num_inference_steps 1 and zero initial latents.

The forward pass mirrors QwenImage21Pipeline.__call__ and diffusers' img2img LoRA trainer exactly: condition latents
first, the noisy target after them, `img_mask` = the prompt's image-pad mask plus one slot per 2x2 target latents,
and the prediction is the tail of the joint sequence.
"""

from __future__ import annotations

import math
from typing import Any

import torch


def calculate_shift(image_seq_len: int, base_seq_len: int = 256, max_seq_len: int = 4096,
                    base_shift: float = 0.5, max_shift: float = 1.15) -> float:
    """The pipeline's `mu` for dynamic shifting (same formula and scheduler defaults)."""
    m = (max_shift - base_shift) / (max_seq_len - base_seq_len)
    return image_seq_len * m + (base_shift - m * base_seq_len)


def shift_sigmas(sigmas: torch.Tensor, shift: float | str | None, image_seq_len: int,
                 scheduler_config: dict | None = None) -> torch.Tensor:
    """Apply a time shift. 'auto' uses exp(mu) for this image's sequence length, i.e. what the sampler applies
    at inference (FlowMatchEulerDiscreteScheduler with use_dynamic_shifting, exponential time shift)."""
    if shift is None:
        return sigmas
    if shift == "auto":
        cfg = scheduler_config or {}
        mu = calculate_shift(image_seq_len, cfg.get("base_image_seq_len", 256), cfg.get("max_image_seq_len", 4096),
                             cfg.get("base_shift", 0.5), cfg.get("max_shift", 1.15))
        factor = math.exp(mu)
    else:
        factor = float(shift)
    return factor * sigmas / (1 + (factor - 1) * sigmas)


def sample_sigmas(objective: dict, batch_size: int, image_seq_len: int, generator: torch.Generator | None = None,
                  scheduler_config: dict | None = None) -> torch.Tensor:
    if objective["type"] == "onestep":
        return torch.ones(batch_size)
    scheme = objective["timestep_sampling"]
    if scheme == "logit_normal":
        u = torch.sigmoid(torch.randn(batch_size, generator=generator) * objective["logit_std"] + objective["logit_mean"])
    elif scheme == "mode":
        u = torch.rand(batch_size, generator=generator)
        u = 1 - u - objective["mode_scale"] * (torch.cos(math.pi * u / 2) ** 2 - 1 + u)
    else:
        u = torch.rand(batch_size, generator=generator)
    sigmas = shift_sigmas(u, objective.get("shift"), image_seq_len, scheduler_config)
    return sigmas.clamp(1e-3, 1.0)


def pack(latents: torch.Tensor) -> torch.Tensor:
    """(B, C, 1, H, W) -> (B, H*W, C), QwenImage21Pipeline._pack_latents."""
    b, c, f, h, w = latents.shape
    if f != 1:
        raise ValueError(f"expected one latent frame, got {f}")
    return latents.reshape(b, c, h * w).transpose(1, 2)


def unpack(packed: torch.Tensor, height: int, width: int) -> torch.Tensor:
    """(B, H*W, C) -> (B, C, 1, H, W) in latent units."""
    b, _, c = packed.shape
    return packed.transpose(1, 2).reshape(b, c, 1, height, width)


def predict(transformer: Any, noisy: torch.Tensor, sample: dict, sigmas: torch.Tensor) -> torch.Tensor:
    """Velocity prediction for the noisy target latents (B, C, 1, H, W), conditioned on one encoded sample."""
    cond = sample["cond_latents"]
    image_pad_mask = sample["image_pad_mask"]
    batch = noisy.shape[0]
    height, width = noisy.shape[-2:]
    cond_height, cond_width = cond.shape[-2:]
    target_slots = height * width // 4
    cond_slots = cond_height * cond_width // 4
    if int(image_pad_mask[0].sum()) != cond_slots:
        raise ValueError(f"prompt has {int(image_pad_mask[0].sum())} image slots but the condition latents need "
                         f"{cond_slots}; the cache was built with a different output_resolution or processor")
    img_mask = torch.cat([image_pad_mask, image_pad_mask.new_ones(batch, target_slots)], dim=1)
    hidden = torch.cat([pack(cond), pack(noisy)], dim=1)
    out = transformer(
        hidden_states=hidden,
        encoder_hidden_states=sample["prompt_embeds"],
        encoder_hidden_states_mask=sample.get("prompt_embeds_mask"),
        timestep=sigmas.to(device=noisy.device, dtype=noisy.dtype),  # the pipeline passes t / 1000 = sigma
        img_shapes=[[(1, cond_height, cond_width), (1, height, width)]] * batch,
        img_mask=img_mask,
        return_dict=False,
    )[0]
    return unpack(out[:, -height * width:], height, width)


def loss(transformer: Any, sample: dict, objective: dict, *, sigmas: torch.Tensor | None = None,
         noise: torch.Tensor | None = None, generator: torch.Generator | None = None,
         scheduler_config: dict | None = None) -> torch.Tensor:
    """Per-sample MSE between predicted and true velocity, shape (B,). `sigmas` / `noise` fix them (validation)."""
    x0 = sample["target_latents"]
    batch = x0.shape[0]
    if sigmas is None:
        sigmas = sample_sigmas(objective, batch, x0.shape[-2] * x0.shape[-1], generator, scheduler_config)
    sigmas = sigmas.to(device=x0.device, dtype=torch.float32)
    if noise is None:
        if objective["type"] == "onestep" and objective.get("zero_noise", True):
            noise = torch.zeros_like(x0)
        else:
            noise = torch.randn(x0.shape, generator=generator, dtype=torch.float32).to(x0.device, x0.dtype)
    s = sigmas.view(-1, 1, 1, 1, 1)
    noisy = ((1 - s) * x0.float() + s * noise.float()).to(x0.dtype)
    target = noise.float() - x0.float()
    pred = predict(transformer, noisy, sample, sigmas)
    return ((pred.float() - target) ** 2).flatten(1).mean(1)
