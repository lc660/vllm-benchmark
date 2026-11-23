"""Fine-tuning code (finetune/) and adapter loading (visionbench/models/adapters.py, qwen.py).

The first group needs only PyYAML. The second builds a tiny QwenImage21Transformer2DModel (the config diffusers'
own tests use) and needs torch, diffusers and peft (`uv sync --extra finetune`); it runs on CPU in seconds.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml

from finetune import config as cfg
from finetune import make_eval_configs, methods
from visionbench.models import adapters
from visionbench.models.base import FatalModelError
from visionbench.models.qwen import QwenBackend, QwenConfig

EXPERIMENTS = sorted(p.stem for p in cfg.CONFIG_DIR.glob("*.yaml") if p.stem != "default")


# --------------------------------------------------------------------------- configs (no torch)

@pytest.mark.parametrize("name", EXPERIMENTS)
def test_every_experiment_config_loads(name: str) -> None:
    config = cfg.load(name)
    assert config["name"] == name
    if config["method"]["type"] in adapters.PEFT_METHODS:
        methods.peft_options(config["method"])  # builds without error


def test_overrides_are_parsed_as_yaml_and_deep_merged() -> None:
    config = cfg.load("shared_lora", ["train.lr=2e-4", "data.tasks=[depth]", "name=x", "objective.shift=auto"])
    assert config["train"]["lr"] == 2e-4 and config["data"]["tasks"] == ["depth"] and config["name"] == "x"
    assert config["train"]["grad_accum"] == 4  # untouched default survives the merge
    assert cfg.parse_override("a.b=null") == {"a": {"b": None}}
    with pytest.raises(ValueError):
        cfg.parse_override("no_equals_sign")


@pytest.mark.parametrize("override, message", [
    ("method.type=adalora", "method.type"),
    ("objective.type=ddpm", "objective.type"),
    ("data.tasks=[flow]", "data.tasks"),
    ("objective.shift=-1", "shift"),
    ("train.grad_accum=0", "grad_accum"),
])
def test_invalid_configs_fail_early(override: str, message: str) -> None:
    with pytest.raises(ValueError, match=message):
        cfg.load("shared_lora", [override])


def test_qlora_only_with_lora_family() -> None:
    assert cfg.load("qlora")["model"]["transformer_precision"] == "nf4"
    with pytest.raises(ValueError, match="NF4"):
        cfg.load("lokr", ["model.transformer_precision=nf4"])


def test_lokr_config_is_full_rank_at_unit_scale() -> None:
    method = cfg.load("lokr")["method"]
    options = methods.peft_options(method)
    assert options["alpha"] / options["r"] == 1.0
    assert options["r"] >= 12288 // 16 / 2  # PEFT keeps W2 unsplit (full rank) when r >= max(W2 dims) / 2


def test_dataset_config_applies_task_and_source_shortcuts() -> None:
    config = cfg.load("smoke")
    data = cfg.dataset_config(config, "val")
    assert data["splits"] == ["val"]
    assert [t for t, o in data["tasks"].items() if o["enabled"]] == ["depth"]
    assert [s for s, o in data["sources"].items() if o["enabled"]] == ["nyuv2"]
    assert Path(data["root"]).is_absolute()


# --------------------------------------------------------------------------- eval configs (no torch)

def _fake_adapter(path: Path, method: str = "lora", tasks=("depth", "normals"), steps: int = 40,
                  noise: str = "gaussian") -> Path:
    path.mkdir(parents=True)
    (path / adapters.CONFIG_NAME).write_text(json.dumps({
        "format": adapters.FORMAT, "method": method, "tasks": list(tasks), "step": 7,
        "inference": {"num_inference_steps": steps, "init_noise": noise},
        "base_model": {"id": "Qwen/Qwen-Image-2.1", "revision": "790c92633540aa0cb11d9abf19eb46d861714758"},
    }))
    return path


def test_eval_configs_point_at_the_checkpoint(tmp_path: Path) -> None:
    adapter = _fake_adapter(tmp_path / "run" / "best", steps=1, noise="zeros")
    written = make_eval_configs.make(adapter, "run-best", "test", tmp_path / "out")
    names = {p.stem for p in written}
    assert names == {"qwen_nyuv2_depth", "indomain_vkitti2_depth", "diode_depth", "qwen_normal_nyuv2",
                     "qwen_normal_ibims", "qwen_normal_diode", "coco_pose_qwen"}
    for path in written:
        data = yaml.safe_load(path.read_text())
        options = data["model"]["options"]
        assert options["adapter_path"] == str(adapter.resolve())
        assert options["num_inference_steps"] == 1 and options["init_noise"] == "zeros"
        assert data["generation_version"].startswith("ft-run-best-")
        assert Path(data["output_root"]).is_absolute() and Path(data["model"]["cache_dir"]).is_absolute()
        original = yaml.safe_load((make_eval_configs.CONFIGS / f"{path.stem}.yaml").read_text())
        assert data["dataset"] == original["dataset"] and data["evaluator"] == original["evaluator"]
        assert data["system_prompt"] == original["system_prompt"]


def test_non_lora_checkpoints_switch_quantized_configs_to_bf16(tmp_path: Path) -> None:
    lokr = _fake_adapter(tmp_path / "lokr", method="lokr", tasks=["normals"])
    path = make_eval_configs.make(lokr, "lokr", "test", tmp_path / "out", ["qwen_normal_nyuv2"])[0]
    data = yaml.safe_load(path.read_text())
    assert data["model"]["profile"] == "custom" and data["model"]["options"]["transformer_precision"] == "bf16"
    assert "rerun it in bf16" in path.read_text()
    lora = _fake_adapter(tmp_path / "lora", method="lora", tasks=["normals"])
    data = yaml.safe_load(make_eval_configs.make(lora, "lora", "test", tmp_path / "o2", ["qwen_normal_nyuv2"])[0]
                          .read_text())
    assert data["model"]["profile"] == "vram_16gb"  # LoRA runs on the same NF4 base as the zero-shot baseline


def test_val_split_configs(tmp_path: Path) -> None:
    adapter = _fake_adapter(tmp_path / "a", tasks=["segmentation", "depth", "pose"])
    written = make_eval_configs.make(adapter, "a", "val", tmp_path / "out")
    assert {p.stem for p in written} == {"indomain_nyuv2_semseg_all_val", "indomain_vkitti2_semseg_all_val",
                                         "indomain_vkitti2_depth_val"}
    for path in written:
        data = yaml.safe_load(path.read_text())
        assert data["dataset"]["params"]["split"] == "val" and data["generation_version"].endswith("-val")


# --------------------------------------------------------------------------- backend options (no torch)

BASE = {"model_id": "Qwen/Qwen-Image-2.1", "revision": "790c92633540aa0cb11d9abf19eb46d861714758",
        "cache_dir": "/models"}


def test_backend_validates_adapter_options() -> None:
    with pytest.raises(ValueError, match="init_noise"):
        QwenConfig(**BASE, init_noise="uniform")
    with pytest.raises(ValueError, match="adapter_scale"):
        QwenConfig(**BASE, adapter_scale=0)
    assert QwenConfig(**BASE).adapter_path is None


def test_backend_refuses_wrong_sampling_for_a_checkpoint(tmp_path: Path) -> None:
    adapter = _fake_adapter(tmp_path / "onestep", steps=1, noise="zeros")
    backend = QwenBackend(QwenConfig(**BASE, adapter_path=str(adapter)))  # 40 steps, gaussian: wrong
    with pytest.raises(FatalModelError, match="num_inference_steps=40 but the checkpoint needs 1"):
        backend._apply_adapter(transformer=None)
    other = _fake_adapter(tmp_path / "rev")
    data = json.loads((other / adapters.CONFIG_NAME).read_text())
    data["base_model"]["revision"] = "0" * 40
    (other / adapters.CONFIG_NAME).write_text(json.dumps(data))
    with pytest.raises(FatalModelError, match="trained on revision"):
        QwenBackend(QwenConfig(**BASE, adapter_path=str(other)))._apply_adapter(transformer=None)


def test_adapter_loader_rejects_methods_that_cannot_run_quantized(tmp_path: Path) -> None:
    adapter = _fake_adapter(tmp_path / "p", method="partial")
    with pytest.raises(ValueError, match="bf16"):
        adapters.load(object(), adapter, quantized=True)


# --------------------------------------------------------------------------- tiny model (torch, diffusers, peft)

def _tiny():
    torch = pytest.importorskip("torch")
    pytest.importorskip("peft")
    diffusers = pytest.importorskip("diffusers")
    torch.manual_seed(0)
    model = diffusers.QwenImage21Transformer2DModel(
        patch_size=1, in_channels=4, out_channels=4, num_layers=2, attention_head_dim=16, num_attention_heads=2,
        context_in_dim=8, mlp_ratio=2, axes_dims_rope=(4, 6, 6))
    return torch, model


def _sample(torch, batch: int = 1) -> dict:
    """A prompt of 9 tokens whose positions 2..5 are the condition image's 4 slots (4x4 latents), target 2x4."""
    g = torch.Generator().manual_seed(1)
    mask = torch.zeros(batch, 9, dtype=torch.bool)
    mask[:, 2:6] = True
    return {"prompt_embeds": torch.randn(batch, 9, 8, generator=g), "image_pad_mask": mask,
            "cond_latents": torch.randn(batch, 4, 1, 4, 4, generator=g),
            "target_latents": torch.randn(batch, 4, 1, 2, 4, generator=g)}


TINY_METHODS = {
    "lora": {"type": "lora", "rank": 4, "alpha": 4, "targets": ["to_q", "to_k", "to_v", "to_out.0", "img_mlp.proj"]},
    "dora": {"type": "dora", "rank": 4, "alpha": 4, "targets": ["to_q", "to_v", "img_mlp.out"]},
    "lokr": {"type": "lokr", "rank": 1000, "alpha": 1000, "targets": ["to_q", "to_k", "to_v", "to_out.0"],
             "lokr": {"decompose_factor": 4, "decompose_both": False}},
    "loha": {"type": "loha", "rank": 4, "alpha": 4, "targets": ["to_q", "to_v"]},
    "partial": {"type": "partial", "partial": {"last_blocks": 1, "modules": ["proj_out"]}},
}
FLOW = {"type": "flow", "timestep_sampling": "logit_normal", "logit_mean": 0.0, "logit_std": 1.0, "shift": None}


@pytest.mark.parametrize("name", sorted(TINY_METHODS))
def test_method_trains_saves_and_reloads(name: str, tmp_path: Path) -> None:
    torch, model = _tiny()
    from finetune import objectives

    saved = methods.apply(model, TINY_METHODS[name], torch.float32)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}
    assert trainable and len(trainable) < len(list(model.parameters()))
    sample = _sample(torch)
    fixed = {"sigmas": torch.tensor([0.5]), "noise": torch.randn(1, 4, 1, 2, 4, generator=torch.Generator().manual_seed(2))}
    before = objectives.loss(model, sample, FLOW, **fixed).item()

    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=1e-2)
    for step in range(5):
        loss = objectives.loss(model, sample, FLOW, generator=torch.Generator().manual_seed(step)).mean()
        loss.backward()
        assert all((p.grad is not None) == (n in trainable) for n, p in model.named_parameters())
        optimizer.step()
        optimizer.zero_grad()
    after = objectives.loss(model, sample, FLOW, **fixed).item()
    assert after != before

    adapters.save(model, tmp_path / "ckpt", {**saved, "save_dtype": "fp32"})
    _, fresh = _tiny()  # same seed: identical base weights
    adapters.load(fresh, tmp_path / "ckpt")
    with torch.no_grad():
        assert torch.allclose(objectives.loss(fresh, sample, FLOW, **fixed),
                              objectives.loss(model, sample, FLOW, **fixed), atol=1e-6)


def test_untrained_peft_adapter_does_not_change_the_model() -> None:
    torch, model = _tiny()
    from finetune import objectives

    sample = _sample(torch)
    fixed = {"sigmas": torch.tensor([0.3]), "noise": torch.zeros(1, 4, 1, 2, 4)}
    with torch.no_grad():
        base = objectives.loss(model, sample, FLOW, **fixed)
        methods.apply(model, TINY_METHODS["lora"], torch.float32)
        assert torch.allclose(objectives.loss(model, sample, FLOW, **fixed), base)


def test_adapter_scale_zero_point_five_halves_a_lora_update(tmp_path: Path) -> None:
    torch, model = _tiny()
    from finetune import objectives

    saved = methods.apply(model, TINY_METHODS["lora"], torch.float32)
    for n, p in model.named_parameters():
        if "lora_B" in n:
            torch.nn.init.normal_(p, std=0.5)
    adapters.save(model, tmp_path / "c", {**saved, "save_dtype": "fp32"})
    sample, sig = _sample(torch), torch.tensor([0.5])
    noisy = torch.zeros(1, 4, 1, 2, 4)
    outputs = {}
    for scale in (0.0001, 0.5, 1.0):
        _, fresh = _tiny()
        adapters.load(fresh, tmp_path / "c", scale=scale)
        # The whole transformer is nonlinear, so its final prediction need not be
        # monotonic in adapter scale.  The first LoRA target itself is linear,
        # which lets this test check the actual scale contract deterministically.
        target = fresh.transformer_blocks[0].attn.to_q
        captured = []
        handle = target.register_forward_hook(lambda _module, _inputs, output: captured.append(output.detach()))
        with torch.no_grad():
            objectives.predict(fresh, noisy, sample, sig)
        handle.remove()
        assert len(captured) == 1
        outputs[scale] = captured[0]
    d_half = (outputs[0.5] - outputs[0.0001]).abs().mean()
    d_full = (outputs[1.0] - outputs[0.0001]).abs().mean()
    assert 0 < d_half < d_full


def test_prediction_matches_the_pipeline_layout() -> None:
    """predict() must hand the transformer [condition, target] latents and read the target back from the tail."""
    torch, model = _tiny()
    from finetune import objectives

    sample = _sample(torch, batch=2)
    noisy = torch.randn(2, 4, 1, 2, 4)
    with torch.no_grad():
        pred = objectives.predict(model, noisy, sample, torch.tensor([0.4, 0.4]))
        img_mask = torch.cat([sample["image_pad_mask"], torch.ones(2, 2, dtype=torch.bool)], dim=1)
        hidden = torch.cat([objectives.pack(sample["cond_latents"]), objectives.pack(noisy)], dim=1)
        raw = model(hidden_states=hidden, encoder_hidden_states=sample["prompt_embeds"], timestep=torch.tensor([0.4, 0.4]),
                    img_shapes=[[(1, 4, 4), (1, 2, 4)]] * 2, img_mask=img_mask, return_dict=False)[0]
    assert pred.shape == noisy.shape
    assert torch.equal(objectives.pack(pred), raw[:, -8:])
    assert torch.equal(objectives.unpack(objectives.pack(noisy), 2, 4), noisy)


def test_prediction_rejects_a_condition_size_mismatch() -> None:
    torch, model = _tiny()
    from finetune import objectives

    sample = _sample(torch)
    sample["cond_latents"] = torch.randn(1, 4, 1, 2, 2)  # 1 slot, prompt has 4
    with pytest.raises(ValueError, match="image slots"):
        objectives.predict(model, torch.randn(1, 4, 1, 2, 4), sample, torch.tensor([0.5]))


def test_onestep_objective_targets_the_clean_latents() -> None:
    """At sigma 1 with zero noise the input is all zeros and one Euler step z - v must give x0."""
    torch = pytest.importorskip("torch")
    from finetune import objectives

    onestep = {"type": "onestep", "zero_noise": True}
    assert torch.equal(objectives.sample_sigmas(onestep, 3, 100), torch.ones(3))

    class Oracle(torch.nn.Module):  # a transformer that returns exactly the target velocity -x0
        def forward(self, hidden_states, **kwargs):
            assert torch.count_nonzero(hidden_states[:, -8:]) == 0  # noisy target is all zeros
            return (-objectives.pack(sample["target_latents"]),)

    sample = _sample(torch)
    sample["prompt_embeds_mask"] = None
    loss = objectives.loss(Oracle(), sample, onestep)
    assert float(loss) == pytest.approx(0.0, abs=1e-12)


def test_timestep_shift() -> None:
    torch = pytest.importorskip("torch")
    from finetune import objectives

    u = torch.linspace(0.01, 0.99, 50)
    assert torch.equal(objectives.shift_sigmas(u, None, 4096), u)
    shifted = objectives.shift_sigmas(u, 3.0, 4096)
    assert bool((shifted >= u).all()) and float(shifted[-1]) <= 1.0
    # auto: the sampler's exp(mu); mu = 1.15 at 4096 tokens and 0.5 at 256 with the shipped scheduler defaults
    assert torch.allclose(objectives.shift_sigmas(u, "auto", 4096), objectives.shift_sigmas(u, 3.1581929, 4096), atol=1e-5)
    assert torch.allclose(objectives.shift_sigmas(u, "auto", 256), objectives.shift_sigmas(u, 1.6487213, 256), atol=1e-5)
    sig = objectives.sample_sigmas({**FLOW, "shift": "auto"}, 1000, 1200, torch.Generator().manual_seed(0))
    assert float(sig.min()) >= 1e-3 and float(sig.max()) <= 1.0


def test_image_pad_rows_of_the_prompt_never_reach_the_output() -> None:
    """The encode cache stores only text rows (finetune/encode.py). That is exact only if the transformer overwrites
    the image-pad rows with the condition latents; check it on the real model class."""
    torch, model = _tiny()
    from finetune import encode, objectives

    sample = _sample(torch)
    stored = {"text_embeds": encode.compact(sample["prompt_embeds"], sample["image_pad_mask"]),
              **{k: sample[k] for k in ("image_pad_mask", "cond_latents", "target_latents")}}
    assert stored["text_embeds"].shape == (1, 5, 8)
    restored = encode.expand(stored)
    assert torch.equal(restored["prompt_embeds"][:, ~sample["image_pad_mask"][0]],
                       sample["prompt_embeds"][:, ~sample["image_pad_mask"][0]])
    noisy, sigma = torch.randn(1, 4, 1, 2, 4), torch.tensor([0.6])
    with torch.no_grad():
        assert torch.equal(objectives.predict(model, noisy, restored, sigma),
                           objectives.predict(model, noisy, sample, sigma))
