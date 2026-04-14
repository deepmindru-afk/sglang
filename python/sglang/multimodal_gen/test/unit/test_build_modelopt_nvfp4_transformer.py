import json
import sys

import transformers
from safetensors import safe_open
from safetensors.torch import save_file
import torch

if not hasattr(transformers, "Qwen2_5_VLProcessor"):
    class _Qwen2_5_VLProcessor:
        pass

    transformers.Qwen2_5_VLProcessor = _Qwen2_5_VLProcessor
    sys.modules["transformers"].Qwen2_5_VLProcessor = _Qwen2_5_VLProcessor
    if hasattr(transformers, "__all__") and "Qwen2_5_VLProcessor" not in transformers.__all__:
        transformers.__all__.append("Qwen2_5_VLProcessor")

from sglang.multimodal_gen.tools.build_modelopt_nvfp4_transformer import (
    _canonicalize_ltx2_output_name,
    _is_ltx2_x0_export,
    _load_selected_tensors,
    _preset_patterns,
    _should_keep_ltx2_transformer_key,
    build_modelopt_nvfp4_transformer,
)


def test_ltx2_nvfp4_preset_includes_expected_sensitive_modules():
    patterns = _preset_patterns("ltx2-nvfp4")
    assert "patchify_proj" in patterns
    assert "audio_patchify_proj" in patterns
    assert "caption_projection*" in patterns
    assert "transformer_blocks.*.attn2.to_*" in patterns
    assert "transformer_blocks.*.audio_attn2.to_*" in patterns
    assert "transformer_blocks.1.attn1.to_*" in patterns
    assert "transformer_blocks.43.attn1.to_*" not in patterns


def test_ltx2_nvfp4_detects_velocity_model_x0_export():
    assert _is_ltx2_x0_export(
        config={"_class_name": "X0Model"},
        source_metadata={},
        source_weight_map={
            "velocity_model.patchify_proj.weight": "model.safetensors",
            "velocity_model.transformer_blocks.0.audio_to_video_attn.to_q.weight": "model.safetensors",
        },
    )


def test_ltx2_nvfp4_keeps_velocity_transformer_keys_only():
    assert _should_keep_ltx2_transformer_key("velocity_model.patchify_proj.weight")
    assert not _should_keep_ltx2_transformer_key(
        "velocity_model.audio_embeddings_connector.linear.weight"
    )


def test_ltx2_nvfp4_canonicalizes_output_names():
    assert (
        _canonicalize_ltx2_output_name(
            "velocity_model.transformer_blocks.0.ff.net.0.proj.weight"
        )
        == "transformer_blocks.0.ff.proj_in.weight"
    )
    assert (
        _canonicalize_ltx2_output_name(
            "velocity_model.transformer_blocks.0.ff.net.2.bias"
        )
        == "transformer_blocks.0.ff.proj_out.bias"
    )


def test_ltx2_nvfp4_resolves_base_tensor_aliases(tmp_path):
    shard_path = tmp_path / "model.safetensors"
    adaln = torch.randn(4, 4, dtype=torch.bfloat16)
    ff_proj = torch.randn(8, 4, dtype=torch.bfloat16)
    proj_in = torch.randn(8, 4, dtype=torch.bfloat16)
    save_file(
        {
            "time_embed.emb.timestep_embedder.linear_1.weight": adaln,
            "transformer_blocks.0.ff.proj_in.weight": ff_proj,
            "proj_in.weight": proj_in,
        },
        str(shard_path),
    )

    loaded = _load_selected_tensors(
        str(tmp_path),
        {
            "time_embed.emb.timestep_embedder.linear_1.weight": shard_path.name,
            "transformer_blocks.0.ff.proj_in.weight": shard_path.name,
            "proj_in.weight": shard_path.name,
        },
        [
            "velocity_model.adaln_single.emb.timestep_embedder.linear_1.weight",
            "velocity_model.transformer_blocks.0.ff.net.0.proj.weight",
            "velocity_model.patchify_proj.weight",
        ],
    )

    assert torch.equal(
        loaded["velocity_model.adaln_single.emb.timestep_embedder.linear_1.weight"],
        adaln,
    )
    assert torch.equal(
        loaded["velocity_model.transformer_blocks.0.ff.net.0.proj.weight"],
        ff_proj,
    )
    assert torch.equal(loaded["velocity_model.patchify_proj.weight"], proj_in)


def test_ltx2_nvfp4_builder_replaces_fallback_tensors_and_canonicalizes(tmp_path):
    source_dir = tmp_path / "source"
    base_dir = tmp_path / "base"
    output_dir = tmp_path / "out"
    source_dir.mkdir()
    base_dir.mkdir()

    with open(source_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "_class_name": "X0Model",
                "quantization_config": {
                    "quant_method": "modelopt",
                    "quant_algo": "NVFP4",
                    "ignore": [],
                },
            },
            f,
        )
    with open(base_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"_class_name": "LTX2VideoTransformer3DModel"}, f)

    source_weight = torch.tensor([[0xAB]], dtype=torch.uint8)
    source_scale = torch.tensor([1.0], dtype=torch.float32)
    base_weight = torch.randn(4, 4, dtype=torch.bfloat16)
    save_file(
        {
            "velocity_model.patchify_proj.weight": source_weight,
            "velocity_model.patchify_proj.weight_scale": source_scale,
        },
        str(source_dir / "model.safetensors"),
        metadata={
            "config": json.dumps({"transformer": {"_class_name": "LTX2VideoTransformer3DModel"}})
        },
    )
    save_file({"proj_in.weight": base_weight}, str(base_dir / "model.safetensors"))

    stats = build_modelopt_nvfp4_transformer(
        base_transformer_dir=str(base_dir),
        modelopt_hf_dir=str(source_dir),
        output_dir=str(output_dir),
        pattern_preset="ltx2-nvfp4",
        overwrite=True,
    )

    assert stats["fallback_modules"] > 0
    with safe_open(
        str(output_dir / "model.safetensors"), framework="pt", device="cpu"
    ) as f:
        output_weight = f.get_tensor("patchify_proj.weight")
        keys = list(f.keys())
    assert output_weight.dtype == torch.bfloat16
    assert torch.equal(output_weight, base_weight)
    assert "patchify_proj.weight_scale" not in keys

    output_config = json.loads((output_dir / "config.json").read_text())
    assert output_config["_class_name"] == "LTX2VideoTransformer3DModel"
    assert "patchify_proj" in output_config["quantization_config"]["ignore"]


def test_ltx2_nvfp4_custom_keep_bf16_pattern_drops_aux_tensors(tmp_path):
    source_dir = tmp_path / "source"
    base_dir = tmp_path / "base"
    output_dir = tmp_path / "out"
    source_dir.mkdir()
    base_dir.mkdir()

    with open(source_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "_class_name": "X0Model",
                "quantization_config": {
                    "quant_method": "modelopt",
                    "quant_algo": "NVFP4",
                    "ignore": [],
                },
            },
            f,
        )
    with open(base_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"_class_name": "LTX2VideoTransformer3DModel"}, f)

    source_weight = torch.tensor([[0xAB]], dtype=torch.uint8)
    source_weight_scale = torch.ones((1, 1), dtype=torch.float8_e4m3fn)
    source_weight_scale_2 = torch.tensor([1.0], dtype=torch.float32)
    source_input_scale = torch.tensor([1.0], dtype=torch.float32)
    base_weight = torch.randn(4, 4, dtype=torch.bfloat16)
    save_file(
        {
            "velocity_model.transformer_blocks.3.attn2.to_v.weight": source_weight,
            "velocity_model.transformer_blocks.3.attn2.to_v.weight_scale": source_weight_scale,
            "velocity_model.transformer_blocks.3.attn2.to_v.weight_scale_2": source_weight_scale_2,
            "velocity_model.transformer_blocks.3.attn2.to_v.input_scale": source_input_scale,
        },
        str(source_dir / "model.safetensors"),
        metadata={
            "config": json.dumps({"transformer": {"_class_name": "LTX2VideoTransformer3DModel"}})
        },
    )
    save_file(
        {"transformer_blocks.3.attn2.to_v.weight": base_weight},
        str(base_dir / "model.safetensors"),
    )

    stats = build_modelopt_nvfp4_transformer(
        base_transformer_dir=str(base_dir),
        modelopt_hf_dir=str(source_dir),
        output_dir=str(output_dir),
        pattern_preset="none",
        keep_bf16_patterns=["transformer_blocks.3.attn2.to_*"],
        overwrite=True,
    )

    assert stats["replaced_tensors"] == 1
    assert stats["removed_aux_tensors"] == 3
    with safe_open(
        str(output_dir / "model.safetensors"), framework="pt", device="cpu"
    ) as f:
        keys = set(f.keys())
        output_weight = f.get_tensor("transformer_blocks.3.attn2.to_v.weight")
    assert torch.equal(output_weight, base_weight)
    assert "transformer_blocks.3.attn2.to_v.weight_scale" not in keys
    assert "transformer_blocks.3.attn2.to_v.weight_scale_2" not in keys
    assert "transformer_blocks.3.attn2.to_v.input_scale" not in keys
