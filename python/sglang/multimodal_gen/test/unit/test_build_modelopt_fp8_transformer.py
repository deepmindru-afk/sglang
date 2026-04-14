import json

import torch
import transformers
import sys
from safetensors import safe_open
from safetensors.torch import save_file

if not hasattr(transformers, "Qwen2_5_VLProcessor"):
    class _Qwen2_5_VLProcessor:
        pass

    transformers.Qwen2_5_VLProcessor = _Qwen2_5_VLProcessor
    sys.modules["transformers"].Qwen2_5_VLProcessor = _Qwen2_5_VLProcessor
    if hasattr(transformers, "__all__") and "Qwen2_5_VLProcessor" not in transformers.__all__:
        transformers.__all__.append("Qwen2_5_VLProcessor")

from sglang.multimodal_gen.tools.build_modelopt_fp8_transformer import (
    _build_output_config,
    _canonicalize_ltx2_output_name,
    _is_ltx2_x0_export,
    _load_selected_tensors,
    _should_canonicalize_ltx2_output_names,
    _should_keep_ltx2_transformer_key,
    build_modelopt_fp8_transformer,
    extract_disabled_modelopt_layer_names,
    get_default_keep_bf16_patterns,
    output_weight_scale_for_modelopt_fp8_source,
    should_keep_bf16,
)


def test_canonicalize_ltx2_output_name_strips_prefixes_and_ff_aliases():
    assert (
        _canonicalize_ltx2_output_name(
            "velocity_model.transformer_blocks.0.ff.net.0.proj.weight"
        )
        == "transformer_blocks.0.ff.proj_in.weight"
    )
    assert (
        _canonicalize_ltx2_output_name(
            "velocity_model.transformer_blocks.0.ff.net.2.input_scale"
        )
        == "transformer_blocks.0.ff.proj_out.input_scale"
    )
    assert (
        _canonicalize_ltx2_output_name(
            "model.diffusion_model.transformer_blocks.1.audio_to_video_attn.q_norm.weight"
        )
        == "transformer_blocks.1.audio_to_video_attn.q_norm.weight"
    )
    assert (
        _canonicalize_ltx2_output_name(
            "velocity_model.transformer_blocks.2.scale_shift_table"
        )
        == "transformer_blocks.2.scale_shift_table"
    )


def test_should_canonicalize_ltx2_output_names_detects_ltx2_exports():
    source_weight_map = {
        "velocity_model.transformer_blocks.0.audio_to_video_attn.to_q.weight": "model.safetensors",
        "velocity_model.transformer_blocks.0.audio_to_video_attn.q_norm.weight": "model.safetensors",
    }
    assert _should_canonicalize_ltx2_output_names(
        model_type="auto",
        class_name="LTX2VideoTransformer3DModel",
        source_weight_map=source_weight_map,
    )


def test_is_ltx2_x0_export_accepts_velocity_model_prefix():
    assert _is_ltx2_x0_export(
        config={"_class_name": "X0Model"},
        source_metadata={},
        source_weight_map={
            "velocity_model.patchify_proj.weight": "model.safetensors",
            "velocity_model.transformer_blocks.0.audio_to_video_attn.to_q.weight": "model.safetensors",
        },
    )


def test_should_keep_ltx2_transformer_key_accepts_velocity_model_prefix():
    assert _should_keep_ltx2_transformer_key("velocity_model.patchify_proj.weight")
    assert not _should_keep_ltx2_transformer_key(
        "velocity_model.audio_embeddings_connector.linear.weight"
    )


def test_auto_ltx2_keep_bf16_patterns_cover_ltx2_transformer_class():
    patterns = get_default_keep_bf16_patterns(
        model_type="auto",
        class_name="LTX2VideoTransformer3DModel",
    )
    assert patterns
    assert any("adaln_single" in pattern for pattern in patterns)
    assert should_keep_bf16("transformer_blocks.1.attn1.to_q.weight", patterns)
    assert should_keep_bf16("transformer_blocks.2.ff.proj_in.weight", patterns)
    assert not should_keep_bf16("transformer_blocks.43.attn1.to_q.weight", patterns)


def test_load_selected_tensors_resolves_ltx2_base_name_variants(tmp_path):
    shard_path = tmp_path / "model.safetensors"
    adaln = torch.randn(4, 4, dtype=torch.bfloat16)
    ff_proj = torch.randn(8, 4, dtype=torch.bfloat16)
    proj_in = torch.randn(8, 4, dtype=torch.bfloat16)
    save_file(
        {
            "adaln_single.emb.timestep_embedder.linear_1.weight": adaln,
            "transformer_blocks.0.ff.proj_in.weight": ff_proj,
            "proj_in.weight": proj_in,
        },
        str(shard_path),
    )

    loaded = _load_selected_tensors(
        str(tmp_path),
        {
            "adaln_single.emb.timestep_embedder.linear_1.weight": shard_path.name,
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


def test_build_output_config_falls_back_when_ltx2_metadata_is_missing():
    output_config = _build_output_config(
        source_config={"_class_name": "X0Model", "foo": "bar"},
        source_metadata={},
        quant_config={"quant_method": "modelopt", "quant_algo": "FP8"},
        is_ltx2_x0_export=True,
    )
    assert output_config["_class_name"] == "LTX2VideoTransformer3DModel"
    assert output_config["foo"] == "bar"


def test_build_modelopt_fp8_transformer_applies_bf16_fallback_before_ignore(tmp_path):
    source_dir = tmp_path / "source"
    base_dir = tmp_path / "base"
    output_dir = tmp_path / "out"
    backbone_path = tmp_path / "backbone.pt"
    source_dir.mkdir()
    base_dir.mkdir()

    with open(source_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "_class_name": "FluxTransformer2DModel",
                "quantization_config": {
                    "quant_method": "modelopt",
                    "quant_algo": "FP8",
                    "ignore": [],
                },
            },
            f,
        )
    with open(base_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump({"_class_name": "FluxTransformer2DModel"}, f)

    source_weight = torch.randn(4, 4, dtype=torch.float32).to(torch.float8_e4m3fn)
    base_weight = torch.randn(4, 4, dtype=torch.bfloat16)
    save_file({"foo.weight": source_weight}, str(source_dir / "model.safetensors"))
    save_file({"foo.weight": base_weight}, str(base_dir / "model.safetensors"))
    torch.save({"model_state_dict": {}}, backbone_path)

    stats = build_modelopt_fp8_transformer(
        modelopt_hf_dir=str(source_dir),
        modelopt_backbone_ckpt=str(backbone_path),
        output_dir=str(output_dir),
        base_transformer_dir=str(base_dir),
        model_type="none",
        keep_bf16_patterns=[r"^foo$"],
        overwrite=True,
    )

    assert stats["bf16_fallback_weights"] == 1
    with safe_open(
        str(output_dir / "model.safetensors"), framework="pt", device="cpu"
    ) as f:
        tensor = f.get_tensor("foo.weight")
    assert tensor.dtype == torch.bfloat16
    assert torch.equal(tensor, base_weight)


def test_output_weight_scale_uses_unit_scale_for_fp8_serialized_source():
    source_weight = torch.randn(4, 4, dtype=torch.float32).to(torch.float8_e4m3fn)
    recovered_scale = torch.tensor(0.125, dtype=torch.float32)

    output_scale = output_weight_scale_for_modelopt_fp8_source(
        source_weight,
        recovered_scale,
    )

    assert output_scale.dtype == torch.float32
    assert torch.equal(output_scale, torch.ones_like(recovered_scale))


def test_extract_disabled_modelopt_layer_names_prefers_final_quantizer_state():
    modelopt_state = {
        "modelopt_state_dict": [
            (
                "quantize",
                {
                    "metadata": {
                        "quantizer_state": {
                            "foo.weight_quantizer": {"_disabled": False},
                            "foo.input_quantizer": {"_disabled": False},
                            "bar.weight_quantizer": {"_disabled": False},
                        }
                    }
                },
            ),
            (
                "max_calibrate",
                {
                    "metadata": {
                        "quantizer_state": {
                            "foo.weight_quantizer": {"_disabled": True},
                            "foo.input_quantizer": {"_disabled": True},
                            "bar.weight_quantizer": {"_disabled": False},
                            "bar.input_quantizer": {"_disabled": True},
                            "baz.output_quantizer": {"_disabled": True},
                        }
                    }
                },
            ),
        ]
    }

    assert extract_disabled_modelopt_layer_names(modelopt_state) == {"foo", "bar"}


def test_build_modelopt_fp8_transformer_respects_disabled_modelopt_quantizers(tmp_path):
    source_dir = tmp_path / "source"
    output_dir = tmp_path / "out"
    backbone_path = tmp_path / "backbone.pt"
    source_dir.mkdir()

    with open(source_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(
            {
                "_class_name": "FluxTransformer2DModel",
                "quantization_config": {
                    "quant_method": "modelopt",
                    "quant_algo": "FP8",
                    "ignore": [],
                },
            },
            f,
        )

    source_weight = torch.randn(4, 4, dtype=torch.bfloat16)
    save_file({"foo.weight": source_weight}, str(source_dir / "model.safetensors"))
    torch.save(
        {
            "model_state_dict": {
                "foo.weight_quantizer._amax": torch.tensor(56.0),
                "foo.input_quantizer._amax": torch.tensor(28.0),
            },
            "modelopt_state": {
                "modelopt_state_dict": [
                    (
                        "max_calibrate",
                        {
                            "metadata": {
                                "quantizer_state": {
                                    "foo.weight_quantizer": {"_disabled": True},
                                    "foo.input_quantizer": {"_disabled": True},
                                }
                            }
                        },
                    )
                ]
            },
        },
        backbone_path,
    )

    stats = build_modelopt_fp8_transformer(
        modelopt_hf_dir=str(source_dir),
        modelopt_backbone_ckpt=str(backbone_path),
        output_dir=str(output_dir),
        model_type="none",
        overwrite=True,
    )

    assert stats["quantized_weights"] == 0
    with safe_open(
        str(output_dir / "model.safetensors"), framework="pt", device="cpu"
    ) as f:
        output_weight = f.get_tensor("foo.weight")
        keys = list(f.keys())
    assert output_weight.dtype == torch.bfloat16
    assert torch.equal(output_weight, source_weight)
    assert "foo.weight_scale" not in keys
    assert "foo.input_scale" not in keys

    output_config = json.loads((output_dir / "config.json").read_text())
    assert "foo" in output_config["quantization_config"]["ignore"]
