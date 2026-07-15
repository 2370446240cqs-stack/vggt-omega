"""Supervised multi-task training for VGGT-Omega.

The JSONL manifest contains one sequence per line. Paths are relative to the
manifest. Each ``data`` NPZ stores images plus any supervised targets::

    {"data": "sequences/000001.npz"}

Required arrays:
  images: [S,H,W,3] uint8 or [S,3,H,W] float32 in [0,1]
Optional arrays:
  depth: [S,H,W] or [S,H,W,1], valid depth is finite and > 0
  pose_enc: [S,9], or both extrinsics [S,3,4] and intrinsics [S,3,3]
  text_embedding: [D] (requires --enable-alignment)

Images and depth must already share a fixed, patch-aligned resolution. This
avoids silently invalidating camera intrinsics through image augmentation.
"""

from __future__ import annotations

import argparse
import json
import math
import random
from collections.abc import Mapping
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset

from vggt_omega.models import VGGTOmega
from vggt_omega.utils.pose_enc import extri_intri_to_pose_encoding


class SequenceDataset(Dataset):
    def __init__(self, manifest: Path, frames: int, patch_size: int = 16) -> None:
        self.root = manifest.resolve().parent
        self.frames = frames
        self.patch_size = patch_size
        with manifest.open("r", encoding="utf-8") as handle:
            self.items = [json.loads(line) for line in handle if line.strip()]
        if not self.items:
            raise ValueError(f"Empty manifest: {manifest}")

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.items[index]
        path = Path(item["data"])
        path = path if path.is_absolute() else self.root / path
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key].copy() for key in data.files}
        images = torch.from_numpy(arrays["images"])
        if images.ndim != 4:
            raise ValueError(f"{path}: images must have 4 dimensions")
        if images.shape[-1] == 3:
            images = images.permute(0, 3, 1, 2)
        if images.shape[1] != 3:
            raise ValueError(f"{path}: images must be RGB")
        images = images.float().div_(255.0) if images.dtype == torch.uint8 else images.float()
        height, width = images.shape[-2:]
        if height % self.patch_size or width % self.patch_size:
            raise ValueError(f"{path}: image size {(height, width)} is not divisible by {self.patch_size}")

        sequence_length = images.shape[0]
        if sequence_length < self.frames:
            raise ValueError(f"{path}: has {sequence_length} frames, need {self.frames}")
        start = random.randint(0, sequence_length - self.frames) if sequence_length > self.frames else 0
        sl = slice(start, start + self.frames)
        sample: dict[str, torch.Tensor] = {"images": images[sl]}

        if "depth" in arrays:
            depth = torch.from_numpy(arrays["depth"]).float()
            if depth.ndim == 3:
                depth = depth.unsqueeze(-1)
            if tuple(depth.shape[1:3]) != (height, width) or depth.shape[-1] != 1:
                raise ValueError(f"{path}: depth must have shape [S,H,W] or [S,H,W,1]")
            sample["depth"] = depth[sl]
        if "pose_enc" in arrays:
            sample["pose_enc"] = torch.from_numpy(arrays["pose_enc"])[sl].float()
        elif "extrinsics" in arrays and "intrinsics" in arrays:
            extrinsics = torch.from_numpy(arrays["extrinsics"])[sl].float().unsqueeze(0)
            intrinsics = torch.from_numpy(arrays["intrinsics"])[sl].float().unsqueeze(0)
            sample["pose_enc"] = extri_intri_to_pose_encoding(extrinsics, intrinsics, (height, width))[0]
        if "text_embedding" in arrays:
            sample["text_embedding"] = torch.from_numpy(arrays["text_embedding"]).float()
        if len(sample) == 1:
            raise ValueError(f"{path}: no supervision target found")
        return sample


def _masked_mean(value: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    return value.masked_select(mask).mean() if mask.any() else value.sum() * 0.0


def compute_losses(
    predictions: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor],
    camera_weight: float, depth_weight: float, text_weight: float,
) -> dict[str, torch.Tensor]:
    losses: dict[str, torch.Tensor] = {}
    if "depth" in batch:
        target = batch["depth"]
        prediction = predictions["depth"]
        valid = torch.isfinite(target) & (target > 0)
        # Log-depth is scale balanced; predicted confidence learns a Laplace NLL.
        error = (torch.log(prediction.clamp_min(1e-6)) - torch.log(target.clamp_min(1e-6))).abs()
        confidence = predictions["depth_conf"].unsqueeze(-1).clamp_min(1.0)
        losses["depth"] = _masked_mean(confidence * error - torch.log(confidence), valid)
    if "pose_enc" in batch:
        pred, target = predictions["pose_enc"], batch["pose_enc"]
        translation = F.smooth_l1_loss(pred[..., :3], target[..., :3])
        pred_q = F.normalize(pred[..., 3:7], dim=-1)
        target_q = F.normalize(target[..., 3:7], dim=-1)
        rotation = (1.0 - (pred_q * target_q).sum(-1).abs()).mean()
        fov = F.smooth_l1_loss(pred[..., 7:], target[..., 7:])
        losses["camera"] = translation + rotation + fov
    if "text_embedding" in batch:
        pred = predictions["text_alignment_embedding"]
        target = F.normalize(batch["text_embedding"], dim=-1)
        losses["text"] = (1.0 - (pred * target).sum(-1)).mean()
    weighted = [camera_weight * losses["camera"]] if "camera" in losses else []
    weighted += [depth_weight * losses["depth"]] if "depth" in losses else []
    weighted += [text_weight * losses["text"]] if "text" in losses else []
    if not weighted:
        raise ValueError("Batch contains no target enabled by the model")
    losses["total"] = torch.stack(weighted).sum()
    return losses


def load_state_dict(path: Path) -> dict[str, torch.Tensor]:
    checkpoint = torch.load(path, map_location="cpu", weights_only=True)
    if isinstance(checkpoint, Mapping):
        for key in ("model", "state_dict"):
            if isinstance(checkpoint.get(key), Mapping):
                checkpoint = checkpoint[key]
                break
    state = dict(checkpoint)
    return {key.removeprefix("module."): value for key, value in state.items()}


def save_checkpoint(path: Path, model, optimizer, scheduler, scaler, epoch: int, step: int, args) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                "scheduler": scheduler.state_dict(), "scaler": scaler.state_dict(),
                "epoch": epoch, "step": step, "args": vars(args)}, path)


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train/fine-tune VGGT-Omega")
    p.add_argument("--train-manifest", type=Path, required=True)
    p.add_argument("--output-dir", type=Path, required=True)
    p.add_argument("--checkpoint", type=Path, help="Released model state dict used for initialization")
    p.add_argument("--resume", type=Path, help="Training checkpoint to resume")
    p.add_argument("--epochs", type=int, default=10)
    p.add_argument("--frames", type=int, default=4)
    p.add_argument("--batch-size", type=int, default=1)
    p.add_argument("--workers", type=int, default=4)
    p.add_argument("--accumulation-steps", type=int, default=8)
    p.add_argument("--lr", type=float, default=2e-5)
    p.add_argument("--weight-decay", type=float, default=0.05)
    p.add_argument("--warmup-steps", type=int, default=100)
    p.add_argument("--max-grad-norm", type=float, default=1.0)
    p.add_argument("--camera-weight", type=float, default=1.0)
    p.add_argument("--depth-weight", type=float, default=1.0)
    p.add_argument("--text-weight", type=float, default=1.0)
    p.add_argument("--enable-alignment", action="store_true")
    p.add_argument("--hybrid-prefix-blocks", type=int)
    p.add_argument("--loop-steps", type=int)
    p.add_argument("--shared-block-init-index", type=int)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--save-every", type=int, default=500)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("VGGT-Omega training requires a CUDA GPU")
    if args.checkpoint and args.resume:
        raise ValueError("Use only one of --checkpoint and --resume")
    random.seed(args.seed); np.random.seed(args.seed); torch.manual_seed(args.seed)
    dataset = SequenceDataset(args.train_manifest, args.frames)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.workers,
                        pin_memory=True, drop_last=True, persistent_workers=args.workers > 0)
    model = VGGTOmega(enable_alignment=args.enable_alignment,
                      hybrid_prefix_blocks=args.hybrid_prefix_blocks, loop_steps=args.loop_steps,
                      shared_block_init_index=args.shared_block_init_index)
    if args.checkpoint:
        model.load_state_dict(load_state_dict(args.checkpoint), strict=True)
    model.cuda().train()
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    updates_per_epoch = math.ceil(len(loader) / args.accumulation_steps)
    total_steps = max(1, args.epochs * updates_per_epoch)
    def lr_factor(step: int) -> float:
        if step < args.warmup_steps:
            return (step + 1) / max(1, args.warmup_steps)
        progress = (step - args.warmup_steps) / max(1, total_steps - args.warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * min(progress, 1.0)))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_factor)
    scaler = torch.amp.GradScaler("cuda", enabled=not torch.cuda.is_bf16_supported())
    start_epoch = global_step = 0
    if args.resume:
        state = torch.load(args.resume, map_location="cpu", weights_only=False)
        model.load_state_dict(state["model"]); optimizer.load_state_dict(state["optimizer"])
        scheduler.load_state_dict(state["scheduler"]); scaler.load_state_dict(state["scaler"])
        start_epoch, global_step = state["epoch"], state["step"]

    optimizer.zero_grad(set_to_none=True)
    for epoch in range(start_epoch, args.epochs):
        for batch_index, batch in enumerate(loader):
            batch = {key: value.cuda(non_blocking=True) for key, value in batch.items()}
            predictions = model(batch["images"])
            losses = compute_losses(predictions, batch, args.camera_weight, args.depth_weight, args.text_weight)
            scaler.scale(losses["total"] / args.accumulation_steps).backward()
            should_update = (batch_index + 1) % args.accumulation_steps == 0 or batch_index + 1 == len(loader)
            if should_update:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), args.max_grad_norm)
                scaler.step(optimizer); scaler.update(); optimizer.zero_grad(set_to_none=True); scheduler.step()
                global_step += 1
                values = " ".join(f"{key}={value.detach().item():.5f}" for key, value in losses.items())
                print(f"epoch={epoch + 1}/{args.epochs} step={global_step}/{total_steps} {values}", flush=True)
                if args.save_every > 0 and global_step % args.save_every == 0:
                    save_checkpoint(args.output_dir / f"step_{global_step:08d}.pt", model, optimizer,
                                    scheduler, scaler, epoch, global_step, args)
        save_checkpoint(args.output_dir / "last.pt", model, optimizer, scheduler, scaler,
                        epoch + 1, global_step, args)


if __name__ == "__main__":
    main()
