"""Build an SGLang-loadable ModelOpt NVFP4 diffusion transformer.

This tool keeps the ModelOpt-exported NVFP4 tensors for most transformer
modules, but can replace a validated subset of numerically sensitive modules
with their original BF16 tensors from the base transformer checkpoint.

It is primarily intended for FLUX.1-dev style ModelOpt NVFP4 exports where:
- the base pipeline should remain separate from the quantized transformer
- fallback BF16 modules are model-family specific
- the serialized FP4 weight byte order may already match the runtime kernel
"""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Iterable, Mapping, Sequence

from safetensors import safe_open
from safetensors.torch import load_file, save_file

INDEX_FILENAMES = [
    "model.safetensors.index.json",
    "diffusion_pytorch_model.safetensors.index.json",
]

DEFAULT_FLUX1_NVFP4_FALLBACK_PATTERNS = [
    "transformer_blocks.*.norm1.linear*",
    "transformer_blocks.*.norm1_context.linear*",
    "transformer_blocks.*.ff.net.0.proj*",
    "transformer_blocks.*.ff.net.2*",
    "transformer_blocks.*.ff_context.net.0.proj*",
    "transformer_blocks.*.ff_context.net.2*",
    "single_transformer_blocks.*.norm.linear*",
    "single_transformer_blocks.*.proj_mlp*",
]
DEFAULT_LTX2_NVFP4_FALLBACK_PATTERNS = [
    "patchify_proj",
    "audio_patchify_proj",
    "caption_projection*",
    "audio_caption_projection*",
    "adaln_single*",
    "audio_adaln_single*",
    "av_ca_video_scale_shift_adaln_single*",
    "av_ca_a2v_gate_adaln_single*",
    "av_ca_audio_scale_shift_adaln_single*",
    "av_ca_v2a_gate_adaln_single*",
    "proj_out",
    "audio_proj_out",
    # Blackwell NVFP4 bring-up for LTX-2 is only stable when all cross-attn
    # projections stay in BF16. Keep the rest of the block selective so we
    # still get useful NVFP4 coverage on the denoiser.
    "transformer_blocks.*.attn2.to_*",
    "transformer_blocks.*.audio_attn2.to_*",
    "transformer_blocks.0.attn1.to_*",
    "transformer_blocks.0.audio_attn1.to_*",
    "transformer_blocks.0.audio_to_video_attn.to_*",
    "transformer_blocks.0.video_to_audio_attn.to_*",
    "transformer_blocks.0.ff.proj_*",
    "transformer_blocks.0.audio_ff.proj_*",
    "transformer_blocks.1.attn1.to_*",
    "transformer_blocks.1.audio_attn1.to_*",
    "transformer_blocks.1.audio_to_video_attn.to_*",
    "transformer_blocks.1.video_to_audio_attn.to_*",
    "transformer_blocks.1.ff.proj_*",
    "transformer_blocks.1.audio_ff.proj_*",
    "transformer_blocks.2.attn1.to_*",
    "transformer_blocks.2.audio_attn1.to_*",
    "transformer_blocks.2.audio_to_video_attn.to_*",
    "transformer_blocks.2.video_to_audio_attn.to_*",
    "transformer_blocks.2.ff.proj_*",
    "transformer_blocks.2.audio_ff.proj_*",
    "transformer_blocks.45.attn1.to_*",
    "transformer_blocks.45.audio_attn1.to_*",
    "transformer_blocks.45.audio_to_video_attn.to_*",
    "transformer_blocks.45.video_to_audio_attn.to_*",
    "transformer_blocks.45.ff.proj_*",
    "transformer_blocks.45.audio_ff.proj_*",
    "transformer_blocks.46.attn1.to_*",
    "transformer_blocks.46.audio_attn1.to_*",
    "transformer_blocks.46.audio_to_video_attn.to_*",
    "transformer_blocks.46.video_to_audio_attn.to_*",
    "transformer_blocks.46.ff.proj_*",
    "transformer_blocks.46.audio_ff.proj_*",
    "transformer_blocks.47.attn1.to_*",
    "transformer_blocks.47.audio_attn1.to_*",
    "transformer_blocks.47.audio_to_video_attn.to_*",
    "transformer_blocks.47.video_to_audio_attn.to_*",
    "transformer_blocks.47.ff.proj_*",
    "transformer_blocks.47.audio_ff.proj_*",
]

_TENSOR_MODULE_SUFFIXES = (
    ".weight_scale_2",
    ".weight_scale",
    ".input_scale",
    ".weight",
    ".bias",
)

_QUANT_AUX_SUFFIXES = (
    ".weight_scale_2",
    ".weight_scale",
    ".input_scale",
)


def _resolve_transformer_dir(path: str) -> str:
    candidate = Path(path).expanduser().resolve()
    if (candidate / "config.json").is_file():
        return str(candidate)
    transformer_dir = candidate / "transformer"
    if (transformer_dir / "config.json").is_file():
        return str(transformer_dir)
    raise FileNotFoundError(f"Could not resolve a transformer directory from: {path}")


def _find_index_file(model_dir: str) -> str | None:
    for filename in INDEX_FILENAMES:
        candidate = os.path.join(model_dir, filename)
        if os.path.isfile(candidate):
            return filename

    matches = sorted(
        filename
        for filename in os.listdir(model_dir)
        if filename.endswith(".safetensors.index.json")
    )
    return matches[0] if matches else None


def _load_weight_map(model_dir: str) -> tuple[dict[str, str], str | None]:
    index_filename = _find_index_file(model_dir)
    if index_filename is not None:
        with open(os.path.join(model_dir, index_filename), encoding="utf-8") as f:
            index_data = json.load(f)
        return dict(index_data["weight_map"]), index_filename

    safetensors_files = sorted(
        filename
        for filename in os.listdir(model_dir)
        if filename.endswith(".safetensors")
    )
    if len(safetensors_files) != 1:
        raise ValueError(
            f"Expected an index file or a single safetensors shard in {model_dir}, "
            f"found {len(safetensors_files)} shard(s)."
        )

    shard_name = safetensors_files[0]
    with safe_open(
        os.path.join(model_dir, shard_name), framework="pt", device="cpu"
    ) as f:
        weight_map = {key: shard_name for key in f.keys()}
    index_filename = f"{Path(shard_name).stem}.safetensors.index.json"
    return weight_map, index_filename


def _load_config(model_dir: str) -> dict:
    config_path = os.path.join(model_dir, "config.json")
    with open(config_path, encoding="utf-8") as f:
        return json.load(f)


def _load_first_shard_metadata(
    model_dir: str, weight_map: Mapping[str, str]
) -> dict[str, str]:
    if not weight_map:
        return {}
    first_shard = next(iter(weight_map.values()))
    with safe_open(
        os.path.join(model_dir, first_shard), framework="pt", device="cpu"
    ) as f:
        return dict(f.metadata() or {})


def _write_config(model_dir: Path, config: Mapping[str, object]) -> None:
    with open(model_dir / "config.json", "w", encoding="utf-8") as f:
        json.dump(config, f, indent=2, sort_keys=True)
        f.write("\n")


def _copy_non_shard_files(source_dir: str, output_dir: str) -> None:
    ignored = set(INDEX_FILENAMES)
    for entry in os.listdir(source_dir):
        if entry.endswith(".safetensors") or entry in ignored:
            continue
        source_path = os.path.join(source_dir, entry)
        output_path = os.path.join(output_dir, entry)
        if os.path.isdir(source_path):
            shutil.copytree(source_path, output_path, dirs_exist_ok=True)
        else:
            shutil.copy2(source_path, output_path)


def _load_selected_tensors(
    model_dir: str,
    weight_map: Mapping[str, str],
    tensor_names: Iterable[str],
):
    tensors = {}
    names_by_file: dict[str, list[tuple[str, str]]] = defaultdict(list)
    for name in tensor_names:
        resolved_name = _resolve_tensor_name(name, weight_map)
        names_by_file[weight_map[resolved_name]].append((name, resolved_name))

    for filename, name_pairs in names_by_file.items():
        shard_path = os.path.join(model_dir, filename)
        with safe_open(shard_path, framework="pt", device="cpu") as f:
            for original_name, resolved_name in name_pairs:
                tensors[original_name] = f.get_tensor(resolved_name).contiguous()
    return tensors


def _module_name_variants(weight_name: str) -> list[str]:
    module_name = _module_name_for_tensor(weight_name)
    variants = [module_name]

    for prefix in ("model.diffusion_model.", "velocity_model."):
        if module_name.startswith(prefix):
            variants.append(module_name[len(prefix) :])

    canonicalized: list[str] = []
    for variant in variants:
        canonicalized.append(
            re.sub(r"(\.audio_ff|\.ff)\.net\.0\.proj$", r"\1.proj_in", variant)
        )
        canonicalized.append(
            re.sub(r"(\.audio_ff|\.ff)\.net\.2$", r"\1.proj_out", variant)
        )
    variants.extend(canonicalized)

    deduped: list[str] = []
    for variant in variants:
        if variant not in deduped:
            deduped.append(variant)
    return deduped


def _preferred_module_name(weight_name: str) -> str:
    return _module_name_variants(weight_name)[-1]


def _tensor_name_variants(tensor_name: str) -> list[str]:
    variants = [tensor_name]
    for suffix in _TENSOR_MODULE_SUFFIXES:
        if not tensor_name.endswith(suffix):
            continue
        module_name = tensor_name[: -len(suffix)]
        variants.extend(
            candidate + suffix
            for candidate in _module_name_variants(f"{module_name}.weight")
        )
        break

    deduped: list[str] = []
    for variant in variants:
        if variant not in deduped:
            deduped.append(variant)
    return deduped


def _base_tensor_name_variants(tensor_name: str) -> list[str]:
    variants = list(_tensor_name_variants(tensor_name))
    ltx2_aliases: list[str] = []
    for variant in variants:
        alias = variant
        alias = re.sub(r"^audio_adaln_single\.", "audio_time_embed.", alias)
        alias = re.sub(r"^adaln_single\.", "time_embed.", alias)
        alias = re.sub(
            r"^av_ca_audio_scale_shift_adaln_single\.",
            "av_cross_attn_audio_scale_shift.",
            alias,
        )
        alias = re.sub(
            r"^av_ca_v2a_gate_adaln_single\.",
            "av_cross_attn_audio_v2a_gate.",
            alias,
        )
        alias = re.sub(
            r"^av_ca_a2v_gate_adaln_single\.",
            "av_cross_attn_video_a2v_gate.",
            alias,
        )
        alias = re.sub(
            r"^av_ca_video_scale_shift_adaln_single\.",
            "av_cross_attn_video_scale_shift.",
            alias,
        )
        alias = re.sub(r"^audio_patchify_proj(?=\.|$)", "audio_proj_in", alias)
        alias = re.sub(r"^patchify_proj(?=\.|$)", "proj_in", alias)
        alias = re.sub(r"\.q_norm(?=\.|$)", ".norm_q", alias)
        alias = re.sub(r"\.k_norm(?=\.|$)", ".norm_k", alias)
        ltx2_aliases.append(alias)

    for alias in ltx2_aliases:
        if alias not in variants:
            variants.append(alias)
    return variants


def _resolve_tensor_name(
    tensor_name: str,
    weight_map: Mapping[str, str],
) -> str:
    for candidate in _base_tensor_name_variants(tensor_name):
        if candidate in weight_map:
            return candidate
    raise KeyError(tensor_name)


def _module_name_for_tensor(tensor_name: str) -> str:
    for suffix in _TENSOR_MODULE_SUFFIXES:
        if tensor_name.endswith(suffix):
            return tensor_name[: -len(suffix)]
    return tensor_name


def _matches_any_pattern(module_name: str, patterns: Sequence[str]) -> bool:
    if not patterns:
        return False
    for pattern in patterns:
        regex_str = pattern.replace(".", r"\.").replace("*", r".*")
        if any(re.fullmatch(regex_str, variant) for variant in _module_name_variants(module_name)):
            return True
    return False


def _is_ltx2_x0_export(
    *,
    config: Mapping[str, object],
    source_metadata: Mapping[str, str],
    source_weight_map: Mapping[str, str],
) -> bool:
    if config.get("_class_name") != "X0Model":
        return False
    if not any(
        name.startswith(prefix)
        for name in source_weight_map
        for prefix in ("model.diffusion_model.", "velocity_model.")
    ):
        return False
    try:
        metadata_config = json.loads(str(source_metadata.get("config", "")))
    except json.JSONDecodeError:
        metadata_config = None

    if isinstance(metadata_config, dict) and isinstance(
        metadata_config.get("transformer"), dict
    ):
        return True

    return any(
        ".audio_to_video_attn." in name
        or ".video_to_audio_attn." in name
        or ".audio_attn1." in name
        or ".audio_attn2." in name
        or ".audio_patchify_proj." in name
        or ".audio_proj_out." in name
        for name in source_weight_map
    )


def _build_output_config(
    *,
    source_config: Mapping[str, object],
    source_metadata: Mapping[str, str],
    is_ltx2_x0_export: bool,
) -> dict[str, object]:
    if is_ltx2_x0_export:
        metadata_config_raw = source_metadata.get("config")
        output_config: dict[str, object] | None = None
        if metadata_config_raw:
            metadata_config = json.loads(str(metadata_config_raw))
            transformer_config = metadata_config.get("transformer")
            if isinstance(transformer_config, dict):
                output_config = dict(transformer_config)
        if output_config is None:
            output_config = dict(source_config)
        elif "quantization_config" not in output_config and isinstance(
            source_config.get("quantization_config"), dict
        ):
            # LTX-2 ModelOpt exports may stash the transformer config in metadata
            # without re-copying the top-level quantization config.
            output_config["quantization_config"] = dict(source_config["quantization_config"])
        output_config["_class_name"] = "LTX2VideoTransformer3DModel"
        return output_config
    return dict(source_config)


def _should_keep_ltx2_transformer_key(weight_name: str) -> bool:
    if not weight_name.startswith(("model.diffusion_model.", "velocity_model.")):
        return False
    connector_prefixes = (
        "model.diffusion_model.audio_embeddings_connector.",
        "model.diffusion_model.video_embeddings_connector.",
        "velocity_model.audio_embeddings_connector.",
        "velocity_model.video_embeddings_connector.",
    )
    return not weight_name.startswith(connector_prefixes)


def _canonicalize_ltx2_output_name(weight_name: str) -> str:
    for suffix in _TENSOR_MODULE_SUFFIXES:
        if weight_name.endswith(suffix):
            module_name = weight_name[: -len(suffix)]
            return _preferred_module_name(f"{module_name}.weight") + suffix
    return _preferred_module_name(weight_name)


def _should_canonicalize_ltx2_output_names(
    *,
    source_weight_map: Mapping[str, str],
    class_name: str | None,
    pattern_preset: str,
) -> bool:
    if not (
        pattern_preset == "ltx2-nvfp4"
        or class_name == "LTX2VideoTransformer3DModel"
        or any(
            name.startswith(prefix)
            for name in source_weight_map
            for prefix in ("model.diffusion_model.", "velocity_model.")
        )
    ):
        return False

    return any(
        ".audio_to_video_attn." in name
        or ".video_to_audio_attn." in name
        or name.startswith("model.diffusion_model.")
        or name.startswith("velocity_model.")
        for name in source_weight_map
    )


def _preset_patterns(pattern_preset: str) -> list[str]:
    if pattern_preset == "none":
        return []
    if pattern_preset == "flux1-nvfp4":
        return list(DEFAULT_FLUX1_NVFP4_FALLBACK_PATTERNS)
    if pattern_preset == "ltx2-nvfp4":
        return list(DEFAULT_LTX2_NVFP4_FALLBACK_PATTERNS)
    raise ValueError(f"Unsupported pattern preset: {pattern_preset}")


def _updated_quant_config(
    source_config: Mapping[str, object],
    *,
    fallback_patterns: Sequence[str],
    swap_weight_nibbles: bool,
) -> dict[str, object]:
    output_config = json.loads(json.dumps(source_config))
    quant_config = output_config.get("quantization_config")
    if not isinstance(quant_config, dict):
        raise ValueError("Expected a flat quantization_config dict in config.json.")
    if (
        quant_config.get("quant_method") == "modelopt"
        and not quant_config.get("quant_algo")
    ):
        # Some diffusion ModelOpt HF exports only record quant_method in config.json.
        # This builder is specifically for NVFP4, so normalize the missing field here.
        quant_config["quant_algo"] = "NVFP4"
    quant_config.setdefault("group_size", 16)
    if (
        quant_config.get("quant_method") != "modelopt"
        or "FP4" not in str(quant_config.get("quant_algo", "")).upper()
    ):
        raise ValueError(
            "This tool only supports ModelOpt diffusion NVFP4 exports "
            "(quant_method=modelopt, quant_algo=FP4/NVFP4)."
        )

    ignore_patterns = list(quant_config.get("ignore", []) or [])
    for pattern in fallback_patterns:
        if pattern not in ignore_patterns:
            ignore_patterns.append(pattern)

    quant_config["ignore"] = ignore_patterns
    quant_config.setdefault(
        "quant_type", str(quant_config.get("quant_algo", "")).upper()
    )
    quant_config["swap_weight_nibbles"] = swap_weight_nibbles
    return output_config


def build_modelopt_nvfp4_transformer(
    *,
    base_transformer_dir: str,
    modelopt_hf_dir: str,
    output_dir: str,
    pattern_preset: str = "none",
    keep_bf16_patterns: Sequence[str] | None = None,
    swap_weight_nibbles: bool | None = None,
    overwrite: bool = False,
) -> dict[str, int | bool]:
    source_dir = _resolve_transformer_dir(modelopt_hf_dir)
    base_dir = _resolve_transformer_dir(base_transformer_dir)

    source_config = _load_config(source_dir)
    source_weight_map_all, index_filename = _load_weight_map(source_dir)
    source_metadata = _load_first_shard_metadata(source_dir, source_weight_map_all)
    is_ltx2_export = _is_ltx2_x0_export(
        config=source_config,
        source_metadata=source_metadata,
        source_weight_map=source_weight_map_all,
    )
    patterns = _preset_patterns(pattern_preset)
    if keep_bf16_patterns:
        patterns.extend(keep_bf16_patterns)

    resolved_swap_weight_nibbles = (
        swap_weight_nibbles
        if swap_weight_nibbles is not None
        else (False if pattern_preset == "flux1-nvfp4" else True)
    )
    output_config = _updated_quant_config(
        _build_output_config(
            source_config=source_config,
            source_metadata=source_metadata,
            is_ltx2_x0_export=is_ltx2_export,
        ),
        fallback_patterns=patterns,
        swap_weight_nibbles=resolved_swap_weight_nibbles,
    )
    quant_config = output_config["quantization_config"]
    serialized_quant_config = json.dumps(quant_config, sort_keys=True)

    output_path = Path(output_dir).expanduser().resolve()
    if output_path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory already exists: {output_path}. "
                "Use --overwrite to replace it."
            )
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    _copy_non_shard_files(source_dir, str(output_path))
    _write_config(output_path, output_config)

    if is_ltx2_export:
        source_weight_map = {
            name: filename
            for name, filename in source_weight_map_all.items()
            if _should_keep_ltx2_transformer_key(name)
        }
    else:
        source_weight_map = source_weight_map_all
    canonicalize_ltx2_output_names = _should_canonicalize_ltx2_output_names(
        source_weight_map=source_weight_map,
        class_name=output_config.get("_class_name")
        if isinstance(output_config.get("_class_name"), str)
        else None,
        pattern_preset=pattern_preset,
    )
    base_weight_map, _ = _load_weight_map(base_dir)

    fallback_tensor_names = sorted(
        name
        for name in source_weight_map
        if "_quantizer." not in name
        and not name.endswith(_QUANT_AUX_SUFFIXES)
        and _matches_any_pattern(_module_name_for_tensor(name), patterns)
    )
    fallback_tensors = _load_selected_tensors(
        base_dir,
        base_weight_map,
        fallback_tensor_names,
    )
    fallback_modules = {
        _preferred_module_name(tensor_name) for tensor_name in fallback_tensor_names
    }

    weights_by_file: dict[str, list[str]] = defaultdict(list)
    for tensor_name, filename in source_weight_map.items():
        weights_by_file[filename].append(tensor_name)

    updated_weight_map: dict[str, str] = {}
    total_size = 0
    replaced_tensor_count = 0
    removed_aux_tensor_count = 0

    for filename, tensor_names in sorted(weights_by_file.items()):
        shard_path = os.path.join(source_dir, filename)
        shard_tensors = load_file(shard_path, device="cpu")

        with safe_open(shard_path, framework="pt", device="cpu") as f:
            metadata = dict(f.metadata() or {})

        metadata.setdefault("format", "pt")
        metadata["_class_name"] = str(
            output_config.get("_class_name", metadata.get("_class_name", ""))
        )
        metadata["config"] = json.dumps(output_config, sort_keys=True)
        metadata["quantization_config"] = serialized_quant_config
        metadata["_quantization_metadata"] = serialized_quant_config

        for name in list(shard_tensors.keys()):
            if "_quantizer." in name:
                del shard_tensors[name]
                removed_aux_tensor_count += 1
                continue

            module_name = _preferred_module_name(name)
            if module_name not in fallback_modules:
                continue

            if name in fallback_tensors:
                shard_tensors[name] = fallback_tensors[name]
                replaced_tensor_count += 1
            else:
                del shard_tensors[name]
                removed_aux_tensor_count += 1

        if canonicalize_ltx2_output_names:
            canonical_tensors = {}
            for name, tensor in shard_tensors.items():
                canonical_name = _canonicalize_ltx2_output_name(name)
                if canonical_name in canonical_tensors:
                    raise ValueError(
                        "Duplicate canonicalized LTX-2 tensor name "
                        f"{canonical_name!r} derived from {name!r}."
                    )
                canonical_tensors[canonical_name] = tensor
            shard_tensors = canonical_tensors

        save_file(shard_tensors, os.path.join(output_path, filename), metadata=metadata)

        for name, tensor in shard_tensors.items():
            updated_weight_map[name] = filename
            total_size += tensor.element_size() * tensor.numel()

    if index_filename is None:
        raise ValueError(
            "Expected a sharded or indexed ModelOpt HF export, but no index file was found."
        )

    with open(output_path / index_filename, "w", encoding="utf-8") as f:
        json.dump(
            {
                "metadata": {"total_size": total_size},
                "weight_map": updated_weight_map,
            },
            f,
            indent=2,
            sort_keys=True,
        )
        f.write("\n")

    return {
        "fallback_modules": len(fallback_modules),
        "replaced_tensors": replaced_tensor_count,
        "removed_aux_tensors": removed_aux_tensor_count,
        "output_shards": len(weights_by_file),
        "swap_weight_nibbles": resolved_swap_weight_nibbles,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Build an SGLang-loadable ModelOpt NVFP4 diffusion transformer and "
            "optionally keep selected modules in BF16."
        )
    )
    parser.add_argument(
        "--base-transformer-dir",
        required=True,
        help="Original BF16 transformer directory, or a parent model directory.",
    )
    parser.add_argument(
        "--modelopt-hf-dir",
        required=True,
        help="ModelOpt --hf-ckpt-dir output, or its transformer subdirectory.",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        help="Directory to write the mixed transformer checkpoint.",
    )
    parser.add_argument(
        "--pattern-preset",
        choices=["none", "flux1-nvfp4", "ltx2-nvfp4"],
        default="none",
        help="Optional model-family BF16 fallback preset.",
    )
    parser.add_argument(
        "--keep-bf16-pattern",
        action="append",
        default=[],
        help=(
            "Glob-style pattern matched against module names without trailing tensor "
            "suffixes such as .weight or .bias."
        ),
    )
    parser.add_argument(
        "--swap-weight-nibbles",
        action=argparse.BooleanOptionalAction,
        default=None,
        help=(
            "Whether the runtime should swap packed FP4 nibbles before padding. "
            "Defaults to false for --pattern-preset flux1-nvfp4 and true otherwise."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Replace --output-dir if it already exists.",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    stats = build_modelopt_nvfp4_transformer(
        base_transformer_dir=args.base_transformer_dir,
        modelopt_hf_dir=args.modelopt_hf_dir,
        output_dir=args.output_dir,
        pattern_preset=args.pattern_preset,
        keep_bf16_patterns=args.keep_bf16_pattern,
        swap_weight_nibbles=args.swap_weight_nibbles,
        overwrite=args.overwrite,
    )
    print(json.dumps(stats, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
