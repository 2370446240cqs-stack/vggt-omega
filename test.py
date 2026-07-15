"""End-to-end smoke test for the hybrid VGGT-Omega aggregator.

Example:
    python test_hybrid.py \
        --checkpoint path/to/vggt_omega_1b_512.pt \
        --images path/to/image_a.jpg path/to/image_b.jpg
"""

from __future__ import annotations

import argparse
import gc
import time
from collections.abc import Mapping
from pathlib import Path

import torch

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.load_fn import load_and_preprocess_images


HYBRID_PREFIX_BLOCKS = 6
LOOP_STEPS = 18
SHARED_BLOCK_INIT_INDEX = 12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Smoke-test the hybrid VGGT-Omega model.")
    parser.add_argument("--checkpoint", type=Path, required=True, help="Path to a standard VGGT-Omega checkpoint.")
    parser.add_argument("--images", type=Path, nargs="+", required=True, help="Two or more input images.")
    parser.add_argument(
        "--image-resolution",
        type=int,
        default=512,
        help="Preprocessing resolution. Must be divisible by 16. Default: 512.",
    )
    return parser.parse_args()


def load_checkpoint(path: Path) -> dict[str, torch.Tensor]:
    if not path.is_file():
        raise FileNotFoundError(f"Checkpoint not found: {path}")

    try:
        checkpoint = torch.load(path, map_location="cpu", weights_only=True, mmap=True)
    except TypeError:
        checkpoint = torch.load(path, map_location="cpu")

    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Expected a state-dict-like checkpoint, got {type(checkpoint).__name__}")

    for wrapper_key in ("state_dict", "model"):
        wrapped = checkpoint.get(wrapper_key)
        if isinstance(wrapped, Mapping):
            checkpoint = wrapped
            break

    state_dict = dict(checkpoint)
    if state_dict and all(key.startswith("module.") for key in state_dict):
        state_dict = {key.removeprefix("module."): value for key, value in state_dict.items()}
    return state_dict


def assert_shared_block_initialization(
    model: VGGTOmega,
    original_state_dict: Mapping[str, torch.Tensor],
) -> None:
    aggregator = model.aggregator
    if aggregator.shared_frame_block is None or aggregator.shared_inter_frame_block is None:
        raise AssertionError("Hybrid shared blocks were not created")

    mappings = (
        ("frame_blocks", aggregator.shared_frame_block),
        ("inter_frame_blocks", aggregator.shared_inter_frame_block),
    )
    for source_module_name, shared_module in mappings:
        source_prefix = f"aggregator.{source_module_name}.{SHARED_BLOCK_INIT_INDEX}."
        for parameter_name, shared_value in shared_module.state_dict().items():
            source_key = f"{source_prefix}{parameter_name}"
            if source_key not in original_state_dict:
                raise KeyError(f"Checkpoint is missing the shared-block initialization tensor: {source_key}")
            torch.testing.assert_close(shared_value, original_state_dict[source_key])


def assert_residual_gates(model: VGGTOmega) -> None:
    expected_value = 1.0 / LOOP_STEPS
    for gate_name in ("shared_frame_gate", "shared_global_gate"):
        gate = getattr(model.aggregator, gate_name)
        if gate is None or gate.ndim != 0:
            raise AssertionError(f"{gate_name} is not a scalar parameter")
        if not gate.requires_grad:
            raise AssertionError(f"{gate_name} is not trainable")
        torch.testing.assert_close(
            gate.detach(),
            torch.tensor(expected_value, dtype=gate.dtype, device=gate.device),
        )


def register_loop_counters(model: VGGTOmega) -> tuple[dict[str, int], list[torch.utils.hooks.RemovableHandle]]:
    aggregator = model.aggregator
    if aggregator.shared_frame_block is None or aggregator.shared_inter_frame_block is None:
        raise AssertionError("Hybrid shared blocks were not created")

    counts = {"frame": 0, "global": 0}

    def count_frame(_module, _inputs, _output) -> None:
        counts["frame"] += 1

    def count_global(_module, _inputs, _output) -> None:
        counts["global"] += 1

    handles = [
        aggregator.shared_frame_block.register_forward_hook(count_frame),
        aggregator.shared_inter_frame_block.register_forward_hook(count_global),
    ]
    return counts, handles


def assert_predictions(predictions: Mapping[str, torch.Tensor], num_frames: int) -> None:
    required_keys = {"pose_enc", "depth", "depth_conf", "camera_and_register_tokens"}
    missing_keys = required_keys.difference(predictions)
    if missing_keys:
        raise AssertionError(f"Predictions are missing keys: {sorted(missing_keys)}")

    for name in required_keys:
        value = predictions[name]
        if not torch.isfinite(value).all():
            raise AssertionError(f"Prediction {name!r} contains NaN or Inf values")
        if value.shape[0] != 1 or value.shape[1] != num_frames:
            raise AssertionError(
                f"Prediction {name!r} has unexpected batch/frame dimensions {tuple(value.shape[:2])}; "
                f"expected (1, {num_frames})"
            )

    if predictions["pose_enc"].shape[-1] != 9:
        raise AssertionError(f"Expected 9D camera encoding, got {tuple(predictions['pose_enc'].shape)}")


def main() -> None:
    args = parse_args()
    if len(args.images) < 2:
        raise ValueError("Use at least two images so global inter-frame attention is exercised")
    missing_images = [str(path) for path in args.images if not path.is_file()]
    if missing_images:
        raise FileNotFoundError(f"Input images not found: {missing_images}")
    if args.image_resolution <= 0 or args.image_resolution % 16 != 0:
        raise ValueError("--image-resolution must be positive and divisible by 16")
    if not torch.cuda.is_available():
        raise RuntimeError("This full VGGT-Omega smoke test requires a CUDA GPU")

    print("Constructing hybrid model:")
    print(f"  hybrid_prefix_blocks={HYBRID_PREFIX_BLOCKS}")
    print(f"  loop_steps={LOOP_STEPS}")
    print(f"  shared_block_init_index={SHARED_BLOCK_INIT_INDEX}")
    model = VGGTOmega(
        hybrid_prefix_blocks=HYBRID_PREFIX_BLOCKS,
        loop_steps=LOOP_STEPS,
        shared_block_init_index=SHARED_BLOCK_INIT_INDEX,
    ).eval()

    state_dict = load_checkpoint(args.checkpoint)
    load_result = model.load_state_dict(state_dict, strict=True)
    if load_result.missing_keys or load_result.unexpected_keys:
        raise AssertionError(f"Checkpoint loading failed strict validation: {load_result}")
    assert_shared_block_initialization(model, state_dict)
    assert_residual_gates(model)
    print("Checkpoint remapping: passed")
    print(f"Residual scalar gates: passed (initial value={1.0 / LOOP_STEPS:.6f})")

    del state_dict
    gc.collect()
    model = model.to("cuda")
    images = load_and_preprocess_images(
        [str(path) for path in args.images],
        image_resolution=args.image_resolution,
    ).to("cuda")

    loop_counts, hook_handles = register_loop_counters(model)
    torch.cuda.reset_peak_memory_stats()
    torch.cuda.synchronize()
    start_time = time.perf_counter()
    try:
        with torch.inference_mode():
            predictions = model(images)
    finally:
        for handle in hook_handles:
            handle.remove()
    torch.cuda.synchronize()
    elapsed_seconds = time.perf_counter() - start_time

    if loop_counts != {"frame": LOOP_STEPS, "global": LOOP_STEPS}:
        raise AssertionError(
            f"Unexpected shared-block call counts: {loop_counts}; "
            f"expected frame/global to each run {LOOP_STEPS} times"
        )
    assert_predictions(predictions, num_frames=len(args.images))

    peak_memory_gib = torch.cuda.max_memory_allocated() / (1024**3)
    parameter_count = sum(parameter.numel() for parameter in model.parameters())
    print("Shared-block call counts: passed")
    print("Prediction validation: passed")
    print(f"Model parameters: {parameter_count / 1e6:.2f} M")
    print(f"Frame gate: {model.aggregator.shared_frame_gate.item():.6f}")
    print(f"Global gate: {model.aggregator.shared_global_gate.item():.6f}")
    print(f"Forward time: {elapsed_seconds:.3f} s")
    print(f"Peak CUDA tensor memory: {peak_memory_gib:.2f} GiB")
    for name, value in predictions.items():
        if isinstance(value, torch.Tensor):
            print(f"  {name}: shape={tuple(value.shape)}, dtype={value.dtype}")
    print("Hybrid VGGT-Omega smoke test passed")


if __name__ == "__main__":
    main()
