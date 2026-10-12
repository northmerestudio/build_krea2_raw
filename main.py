#!/usr/bin/env python3
"""Build a complete Krea 2 RAW Diffusers model from a Turbo fine-tune.

Transfer: RAW + (fine-tuned Turbo - original Turbo).
The transformer is rebuilt in FP16. All other RAW pipeline components are
copied without changing their contents.
"""

from __future__ import annotations

import argparse
import json
import math
import os
import shutil
import struct
import sys
from contextlib import ExitStack
from pathlib import Path

import safetensors
import torch
from safetensors import safe_open

# Fixed deliberately: no scaling, dtype or stage knobs for this workflow.
MERGE_CHUNK_BYTES = 8 * 1024 * 1024
NATIVE_PREFIXES = ("model.diffusion_model.", "diffusion_model.")
SUPPORTED_DTYPES = {"F16", "BF16", "F32"}
RAW_COMPONENTS = ("scheduler", "tokenizer", "text_encoder", "vae")
PROVENANCE_VERSION = 1


def info(message: str) -> None:
    """Write ordinary progress/results to stdout (suitable for log capture)."""
    print(message, flush=True)


def warn(message: str) -> None:
    """Write actionable warnings to stderr; never treat these as success."""
    print(f"warning: {message}", file=sys.stderr, flush=True)


def step(number: int, title: str) -> None:
    info(f"\n[{number}/4] {title}")


def human_size(size_bytes: int) -> str:
    size = float(size_bytes)
    for unit in ("B", "KB", "MB", "GB", "TB"):
        if size < 1000 or unit == "TB":
            return f"{size:.0f} {unit}" if unit == "B" else f"{size:.2f} {unit}"
        size /= 1000
    raise AssertionError("unreachable")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def open_checkpoint(path: Path):
    """Use non-mmap reads on Windows when Safetensors >= 0.8 supports them."""
    kwargs = {"framework": "pt", "device": "cpu"}
    major, minor = (int(part) for part in safetensors.__version__.split(".")[:2])
    if os.name == "nt" and (major, minor) >= (0, 8):
        kwargs["backend"] = "pread"
    return safe_open(str(path), **kwargs)


def validate_raw_pipeline(folder: Path) -> None:
    """Validate the RAW pipeline support files; its transformer weights are replaced."""
    model_index = folder / "model_index.json"
    transformer_config = folder / "transformer" / "config.json"
    if not model_index.is_file() or not transformer_config.is_file():
        raise ValueError(f"Not a RAW Diffusers pipeline directory: {folder}")

    config = read_json(model_index)
    if config.get("_class_name") != "Krea2Pipeline" or config.get("is_distilled") is not False:
        raise ValueError("The Diffusers base must be Krea-2-Raw (is_distilled=false), not Turbo")
    if read_json(transformer_config).get("_class_name") != "Krea2Transformer2DModel":
        raise ValueError("The RAW transformer/config.json has an unexpected model class")

    for name in (*RAW_COMPONENTS, "transformer"):
        entry = config.get(name)
        if not isinstance(entry, list) or len(entry) != 2 or not entry[1]:
            raise ValueError(f"Missing {name!r} in RAW model_index.json")

    for name in RAW_COMPONENTS:
        component = folder / name
        if not component.is_dir() or not any(component.iterdir()):
            raise ValueError(f"Missing RAW component: {component}")
        if name in ("vae", "text_encoder"):
            weights = list(component.glob("*.safetensors")) + list(component.glob("*.bin"))
            if not weights or any(p.stat().st_size < 1024 for p in weights):
                raise ValueError(f"Missing/incomplete RAW weights (or Git LFS pointers): {component}")
        for index_file in component.glob("*.index.json"):
            shards = set(read_json(index_file).get("weight_map", {}).values())
            if not shards or any(not (component / shard).is_file() for shard in shards):
                raise ValueError(f"Incomplete weight shard index: {index_file}")


def tensor_index(reader, description: str) -> dict[str, tuple[str, tuple[int, ...]]]:
    """Normalize optional ComfyUI prefixes while remembering original key names."""
    result = {}
    for original_name in reader.keys():
        name = next(
            (original_name[len(prefix):] for prefix in NATIVE_PREFIXES if original_name.startswith(prefix)),
            original_name,
        )
        if name in result:
            raise ValueError(f"Duplicate normalized tensor {name!r} in {description}")
        tensor = reader.get_slice(original_name)
        if tensor.get_dtype() not in SUPPORTED_DTYPES:
            raise ValueError(f"Non-FP16/BF16/FP32 tensor in {description}: {original_name}")
        result[name] = (original_name, tuple(tensor.get_shape()))
    if not result:
        raise ValueError(f"Empty checkpoint: {description}")
    return result


def validate_tensor_layout(indexes: list[dict]) -> None:
    first = set(indexes[0])
    for label, other in zip(("fine-tuned Turbo", "RAW"), indexes[1:]):
        if set(other) != first:
            raise ValueError(
                f"Tensor names differ in {label}: "
                f"missing={sorted(first - set(other))[:3]}, extra={sorted(set(other) - first)[:3]}"
            )
    for name in first:
        shapes = [index[name][1] for index in indexes]
        if not (shapes[0] == shapes[1] == shapes[2]):
            raise ValueError(f"Tensor shapes differ at {name}: {shapes}")
    info(f"  Tensor layout: {len(first):,} matching tensors")


def read_chunks(reader, key: str, shape: tuple[int, ...], max_elements: int):
    if not shape:
        yield reader.get_tensor(key)
        return
    row_size = max(1, math.prod(shape[1:]))
    rows_per_chunk = max(1, max_elements // row_size)
    tensor_slice = reader.get_slice(key)
    for start in range(0, shape[0], rows_per_chunk):
        yield tensor_slice[start:min(start + rows_per_chunk, shape[0])]


def make_native_checkpoint(original_turbo: Path, fine_tuned_turbo: Path,
                           original_raw: Path, destination: Path) -> None:
    """Stream one native FP16 .safetensors file; arithmetic is always in FP32."""
    paths = (original_turbo, fine_tuned_turbo, original_raw)
    with ExitStack() as stack:
        readers = [stack.enter_context(open_checkpoint(path)) for path in paths]
        indexes = [tensor_index(reader, str(path)) for reader, path in zip(readers, paths)]
        validate_tensor_layout(indexes)
        raw_index = indexes[2]
        ordered = sorted(raw_index, key=lambda name: raw_index[name][0])

        header = {"__metadata__": {"format": "pt"}}
        data_size = 0
        for name in ordered:
            original_name, shape = raw_index[name]
            end = data_size + math.prod(shape) * 2  # Output is FP16.
            header[original_name] = {
                "dtype": "F16", "shape": list(shape), "data_offsets": [data_size, end]
            }
            data_size = end
        header_bytes = json.dumps(header, separators=(",", ":")).encode("utf-8")
        header_bytes += b" " * (-len(header_bytes) % 8)
        if len(header_bytes) > 100 * 1024 * 1024:
            raise ValueError("Unexpectedly large Safetensors header")

        info("  Computing delta in FP32; writing FP16 weights")
        with torch.no_grad(), destination.open("xb") as output:
            output.write(struct.pack("<Q", len(header_bytes)))
            output.write(header_bytes)
            for number, name in enumerate(ordered, 1):
                shape = raw_index[name][1]
                generators = [
                    read_chunks(reader, index[name][0], shape, MERGE_CHUNK_BYTES // 4)
                    for reader, index in zip(readers, indexes)
                ]
                for original, fine_tuned, raw in zip(*generators, strict=True):
                    original, fine_tuned, raw = original.float(), fine_tuned.float(), raw.float()
                    if not all(torch.isfinite(t).all().item() for t in (original, fine_tuned, raw)):
                        raise ValueError(f"NaN or Inf in source tensor: {name}")
                    # Don't round the delta until after addition to RAW.
                    result = (raw + (fine_tuned - original)).to(torch.float16)
                    if not torch.isfinite(result).all().item():
                        raise ValueError(f"FP16 overflow in merged tensor: {name}")
                    output.write(result.contiguous().numpy().tobytes())
                if number % 100 == 0:
                    info(f"  Merged {number:,}/{len(ordered):,} tensors")

        expected_size = 8 + len(header_bytes) + data_size
        if destination.stat().st_size != expected_size:
            raise OSError("Merged checkpoint has an incorrect file size")

    # Close source readers before verifying/reopening the output on Windows.
    with open_checkpoint(destination) as saved:
        if set(saved.keys()) != {raw_index[name][0] for name in ordered}:
            raise ValueError("Merged checkpoint has incorrect tensor names")
        for name in ordered:
            original_name, shape = raw_index[name]
            tensor = saved.get_slice(original_name)
            if tuple(tensor.get_shape()) != shape or tensor.get_dtype() != "F16":
                raise ValueError(f"Invalid serialized tensor: {original_name}")
    info(f"  Native checkpoint: {human_size(destination.stat().st_size)}")


def validate_reusable_checkpoint(merged: Path, original_raw: Path) -> None:
    """Check existing file format, not the identity or provenance of its weights."""
    with open_checkpoint(merged) as checkpoint, open_checkpoint(original_raw) as reference:
        current = tensor_index(checkpoint, "merged checkpoint")
        baseline = tensor_index(reference, "original RAW")
        if set(current) != set(baseline):
            raise ValueError("Reusable checkpoint has different tensor names than RAW")
        for name in baseline:
            if current[name][1] != baseline[name][1]:
                raise ValueError(f"Reusable checkpoint shape mismatch: {name}")
            if checkpoint.get_slice(current[name][0]).get_dtype() != "F16":
                raise ValueError(f"Reusable checkpoint is not FP16: {name}")


def provenance_path(merged: Path) -> Path:
    return merged.with_name(merged.name + ".sources.json")


def source_identifiers(paths: tuple[Path, Path, Path]) -> list[dict]:
    """Cheap change detection, not a cryptographic hash or scientific verification."""
    return [
        {"path": str(path), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
        for path in paths
    ]


def check_provenance(merged: Path, paths: tuple[Path, Path, Path]) -> None:
    manifest = provenance_path(merged)
    if manifest.is_file():
        record = read_json(manifest)
        if record.get("version") != PROVENANCE_VERSION or record.get("sources") != source_identifiers(paths):
            raise ValueError(
                f"Existing checkpoint was made with different inputs: {merged}. "
                "Use a new --merged-checkpoint path."
            )
    else:
        warn("Merged checkpoint has no provenance record (.sources.json); its source identity "
             "cannot be confirmed. Only tensor compatibility was checked.")


def save_provenance(merged: Path, paths: tuple[Path, Path, Path]) -> None:
    manifest = provenance_path(merged)
    # An existing sidecar must not be silently overwritten.
    with manifest.open("x", encoding="utf-8") as stream:
        json.dump({"version": PROVENANCE_VERSION, "sources": source_identifiers(paths)}, stream, indent=2)
        stream.write("\n")


def export_transformer(merged: Path, raw_pipeline: Path, destination: Path) -> None:
    from diffusers import Krea2Transformer2DModel

    info("  Loading and converting the transformer (requires substantial RAM)")
    transformer = Krea2Transformer2DModel.from_single_file(
        str(merged), config=str(raw_pipeline), subfolder="transformer",
        dtype=torch.float16, low_cpu_mem_usage=True, local_files_only=True,
    )
    if any(parameter.is_meta for parameter in transformer.parameters()):
        raise ValueError("Diffusers conversion left uninitialized transformer parameters")
    transformer.save_pretrained(str(destination), safe_serialization=True, max_shard_size="5GB")
    validate_export(transformer, destination)
    del transformer


def validate_export(transformer, destination: Path) -> None:
    """Ensure serialized Diffusers shards contain all expected tensor names/shapes."""
    expected = transformer.state_dict()
    if not expected or any(t.is_meta for t in expected.values()):
        raise ValueError("Diffusers transformer contains uninitialized weights")
    shards = sorted(destination.glob("*.safetensors"))
    if not shards or not (destination / "config.json").is_file():
        raise ValueError("Diffusers export is missing weights or configuration")
    observed = set()
    for shard_path in shards:
        with open_checkpoint(shard_path) as shard:
            for name in shard.keys():
                if name in observed or name not in expected:
                    raise ValueError(f"Unexpected/duplicate exported tensor: {name}")
                observed.add(name)
                if tuple(shard.get_slice(name).get_shape()) != tuple(expected[name].shape):
                    raise ValueError(f"Exported tensor shape mismatch: {name}")
    if observed != set(expected):
        raise ValueError(f"Export incomplete: {len(set(expected) - observed)} tensors missing")
    if len(shards) > 1:
        manifest = destination / "diffusion_pytorch_model.safetensors.index.json"
        if not manifest.is_file() or set(read_json(manifest).get("weight_map", {})) != set(expected):
            raise ValueError("Diffusers shard index is missing/incomplete")
    info(f"  Transformer export: {len(observed):,} tensor names/shapes checked in {len(shards)} shard(s)")


def copy_raw_components(source: Path, destination: Path) -> None:
    shutil.copy2(source / "model_index.json", destination / "model_index.json")
    for name in RAW_COMPONENTS:
        shutil.copytree(source / name, destination / name)
    info("  Copied unchanged RAW components: " + ", ".join(RAW_COMPONENTS))


def build(original_turbo: Path, fine_tuned_turbo: Path, original_raw: Path,
          raw_pipeline: Path, output_directory: Path, merged_checkpoint: Path | None = None) -> None:
    info("Krea 2 RAW Diffusers builder")
    info("Transfer: RAW + (fine-tuned Turbo - original Turbo)")
    info("Transformer precision: FP16 (FP32 arithmetic)")
    step(1, "Check inputs")
    inputs = tuple(path.expanduser().resolve() for path in (original_turbo, fine_tuned_turbo, original_raw))
    raw_pipeline = raw_pipeline.expanduser().resolve()
    output_directory = output_directory.expanduser().resolve()
    merged_checkpoint = merged_checkpoint.expanduser().resolve() if merged_checkpoint else None

    for path in inputs:
        if not path.is_file() or path.suffix.lower() != ".safetensors":
            raise ValueError(f"Missing or invalid native .safetensors checkpoint: {path}")
    if len(set(inputs)) != len(inputs):
        raise ValueError("Original Turbo, fine-tuned Turbo and RAW must be different files")
    validate_raw_pipeline(raw_pipeline)
    info("  RAW pipeline configuration and support files found")

    if (output_directory == raw_pipeline or output_directory.is_relative_to(raw_pipeline)
            or raw_pipeline.is_relative_to(output_directory)):
        raise ValueError("Output directory must not overlap the RAW base directory")
    if output_directory.exists():
        raise FileExistsError(f"Output already exists: {output_directory}")
    staging = output_directory.with_name(output_directory.name + ".incomplete")
    if staging.exists():
        raise FileExistsError(f"Unfinished previous output exists: {staging}")

    if merged_checkpoint:
        if merged_checkpoint.suffix.lower() != ".safetensors":
            raise ValueError("--merged-checkpoint must end in .safetensors")
        if (merged_checkpoint in inputs or merged_checkpoint.is_relative_to(raw_pipeline)
                or merged_checkpoint.is_relative_to(output_directory)
                or merged_checkpoint.is_relative_to(staging)):
            raise ValueError("Merged checkpoint must be separate from input and output directories")
        if merged_checkpoint.exists():
            if not merged_checkpoint.is_file():
                raise ValueError(f"Not a regular merged checkpoint file: {merged_checkpoint}")
            check_provenance(merged_checkpoint, inputs)
            validate_reusable_checkpoint(merged_checkpoint, inputs[2])
            info(f"  Existing checkpoint is structurally compatible: {merged_checkpoint}")
        elif provenance_path(merged_checkpoint).exists():
            raise FileExistsError("A provenance record exists without its merged checkpoint; inspect it first")

    output_directory.parent.mkdir(parents=True, exist_ok=True)
    staging.mkdir()  # Atomic: refuse concurrent runs to the same destination.
    temporary_merged = staging / "merged_native.safetensors"
    merged = merged_checkpoint if merged_checkpoint else temporary_merged
    try:
        if merged.is_file():
            step(2, "Reuse native merged checkpoint")
            info(f"  Checkpoint: {merged}")
        else:
            step(2, "Merge transformer weights")
        if not merged.is_file():
            if merged_checkpoint:
                merged.parent.mkdir(parents=True, exist_ok=True)
                partial = merged.with_name(merged.name + ".partial")
                if partial.exists():
                    raise FileExistsError(f"Remove/inspect unfinished checkpoint first: {partial}")
                try:
                    make_native_checkpoint(*inputs, partial)
                    partial.replace(merged)
                    save_provenance(merged, inputs)
                except BaseException:
                    partial.unlink(missing_ok=True)
                    raise
                info(f"  Checkpoint retained: {merged}")
            else:
                make_native_checkpoint(*inputs, merged)

        step(3, "Convert transformer to Diffusers")
        warn("Krea 2 normalization modules are recommended to remain FP32 by Diffusers, "
             "but this workflow exports all transformer weights as FP16. "
             "Check inference quality before relying on the result.")
        export_transformer(merged, raw_pipeline, staging / "transformer")
        # Free temporary ~26 GB checkpoint before copying the unchanged components.
        if merged_checkpoint is None:
            temporary_merged.unlink()
        step(4, "Assemble and check model directory")
        copy_raw_components(raw_pipeline, staging)
        validate_raw_pipeline(staging)
        staging.rename(output_directory)
    except BaseException:
        try:
            shutil.rmtree(staging)
        except OSError:
            warn(f"Could not delete incomplete output: {staging}. Remove it manually after checking its contents.")
        raise

    info("\nSuccess: complete Krea 2 RAW Diffusers model created")
    info(f"  Model directory: {output_directory}")
    if merged_checkpoint:
        info(f"  Native checkpoint: {merged_checkpoint} (retained)")
    else:
        info("  Native checkpoint: temporary file removed")
    info("  Validation: tensor metadata and required RAW files checked")
    info("  Note: image quality and downstream training were not tested")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Transfer a Krea 2 Turbo fine-tune to a complete RAW Diffusers model.",
        epilog="All native checkpoints must be unquantized FP16/BF16/FP32 transformer Safetensors files.",
    )
    parser.add_argument("--turbo-original", "--turbo-base", dest="turbo_original", type=Path, required=True,
                        help="Original Turbo transformer .safetensors")
    parser.add_argument("--turbo-finetuned", "--turbo-ft", dest="turbo_finetuned", type=Path, required=True,
                        help="Your fine-tuned Turbo transformer .safetensors")
    parser.add_argument("--raw-original", "--raw", dest="raw_original", type=Path, required=True,
                        help="Original RAW native transformer .safetensors")
    parser.add_argument("--raw-diffusers", "--base", dest="raw_diffusers", type=Path, required=True,
                        help="Local Krea-2-Raw Diffusers pipeline folder")
    parser.add_argument("--output", type=Path, required=True, help="New complete Diffusers folder")
    parser.add_argument("--merged-checkpoint", "--native-merged", type=Path,
                        help="Keep this native FP16 checkpoint; reuse if already present")
    parser.add_argument("--debug", action="store_true",
                        help="Show the full Python traceback if an operation fails")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    try:
        build(args.turbo_original, args.turbo_finetuned, args.raw_original,
              args.raw_diffusers, args.output, args.merged_checkpoint)
    except KeyboardInterrupt:
        print("\nerror: Interrupted by user.", file=sys.stderr, flush=True)
        return 130
    except Exception as exc:
        if args.debug:
            raise
        print(f"error: {exc}", file=sys.stderr, flush=True)
        if isinstance(exc, FileExistsError):
            print("hint: Choose another output/checkpoint path or inspect and remove the existing file.",
                  file=sys.stderr, flush=True)
        elif isinstance(exc, OSError) and (getattr(exc, "winerror", None) == 1455 or "paging file" in str(exc).lower()):
            print("hint: Windows commit-memory limit reached. Increase available RAM/page file capacity.",
                  file=sys.stderr, flush=True)
        else:
            print("hint: Run again with --debug for the full traceback.", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
