# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import math

import torch
import torch.nn as nn

from vggt_omega.models.layers import Mlp, RopePositionEmbedding, SelfAttentionBlock
from vggt_omega.models.layers.vision_transformer import DinoVisionTransformer


_RESNET_MEAN = [0.485, 0.456, 0.406]
_RESNET_STD = [0.229, 0.224, 0.225]
_DEFAULT_CACHED_LAYER_INDICES = (4, 11, 17, 23)


class Aggregator(nn.Module):
    """Alternating-attention encoder over video frames."""

    def __init__(
        self,
        patch_size: int = 16,
        embed_dim: int = 1024,
        depth: int = 24,
        num_heads: int = 16,
        mlp_ratio: float = 4.0,
        num_register_tokens: int = 16,
        register_attention_block_indices: list[int] = [2, 6, 9, 14, 20],
        cached_layer_indices: tuple[int, ...] | None = None,
        hybrid_prefix_blocks: int | None = None,
        loop_steps: int | None = None,
        shared_block_init_index: int | None = None,
        loop_residual_gate_init: float | None = None,
    ) -> None:
        super().__init__()

        if depth <= 0:
            raise ValueError(f"depth must be positive, got {depth}")

        self.is_hybrid = hybrid_prefix_blocks is not None
        if self.is_hybrid:
            if hybrid_prefix_blocks is None or not 0 <= hybrid_prefix_blocks < depth:
                raise ValueError(
                    f"hybrid_prefix_blocks must be in [0, {depth - 1}], got {hybrid_prefix_blocks}"
                )
            if loop_steps is None or loop_steps <= 0:
                raise ValueError(f"loop_steps must be positive in hybrid mode, got {loop_steps}")
            prefix_depth = hybrid_prefix_blocks
            total_steps = prefix_depth + loop_steps
            if loop_residual_gate_init is None:
                loop_residual_gate_init = 1.0 / loop_steps
            if not math.isfinite(loop_residual_gate_init) or not 0.0 <= loop_residual_gate_init <= 1.0:
                raise ValueError(
                    "loop_residual_gate_init must be finite and within [0, 1], "
                    f"got {loop_residual_gate_init}"
                )
        else:
            if loop_steps is not None:
                raise ValueError("loop_steps requires hybrid_prefix_blocks")
            if shared_block_init_index is not None:
                raise ValueError("shared_block_init_index requires hybrid_prefix_blocks")
            if loop_residual_gate_init is not None:
                raise ValueError("loop_residual_gate_init requires hybrid_prefix_blocks")
            prefix_depth = depth
            total_steps = depth

        if cached_layer_indices is None:
            cached_layer_indices = _scale_cached_layer_indices(total_steps)
        _validate_cached_layer_indices(cached_layer_indices, total_steps)

        self.patch_embed = _build_patch_embed(patch_size=patch_size, embed_dim=embed_dim)
        self.rope_embed = RopePositionEmbedding(
            embed_dim=embed_dim,
            num_heads=num_heads,
            base=100,
            normalize_coords="max",
            dtype=torch.float32,
        )

        block_kwargs = {
            "dim": embed_dim,
            "num_heads": num_heads,
            "ffn_ratio": mlp_ratio,
            "qkv_bias": True,
            "proj_bias": True,
            "ffn_bias": True,
            "ffn_layer": Mlp,
            "init_values": 1e-5,
            "use_qk_norm": True,
            "mask_k_bias": True,
        }
        self.frame_blocks = nn.ModuleList([SelfAttentionBlock(**block_kwargs) for _ in range(prefix_depth)])
        self.inter_frame_blocks = nn.ModuleList([SelfAttentionBlock(**block_kwargs) for _ in range(prefix_depth)])

        if self.is_hybrid:
            self.shared_frame_block = SelfAttentionBlock(**block_kwargs)
            self.shared_inter_frame_block = SelfAttentionBlock(**block_kwargs)
            self.shared_frame_gate = nn.Parameter(torch.tensor(loop_residual_gate_init, dtype=torch.float32))
            self.shared_global_gate = nn.Parameter(torch.tensor(loop_residual_gate_init, dtype=torch.float32))
            if shared_block_init_index is None:
                register_indices = set(register_attention_block_indices)
                shared_block_init_index = next(
                    (idx for idx in range(prefix_depth, depth) if idx not in register_indices),
                    prefix_depth,
                )
            if not prefix_depth <= shared_block_init_index < depth:
                raise ValueError(
                    "shared_block_init_index must refer to one of the replaced blocks "
                    f"in [{prefix_depth}, {depth - 1}], got {shared_block_init_index}"
                )
        else:
            self.shared_frame_block = None
            self.shared_inter_frame_block = None
            self.shared_frame_gate = None
            self.shared_global_gate = None

        self.depth = depth
        self.prefix_depth = prefix_depth
        self.loop_steps = loop_steps if loop_steps is not None else 0
        self.total_steps = total_steps
        self.shared_block_init_index = shared_block_init_index
        self.patch_size = patch_size
        self.cached_layer_indices = tuple(cached_layer_indices)
        self._cached_layer_indices_set = set(cached_layer_indices)
        self.camera_token = nn.Parameter(torch.empty(1, 2, 1, embed_dim))
        self.register_token = nn.Parameter(torch.empty(1, 2, num_register_tokens, embed_dim))
        self.patch_token_start = 1 + num_register_tokens

        self.inter_frame_attention_types = ["global"] * depth
        for idx in register_attention_block_indices:
            if idx < 0 or idx >= depth:
                raise ValueError(f"register_attention_block_indices contains invalid block index {idx}")
            self.inter_frame_attention_types[idx] = "register"

        for name, value in (("_resnet_mean", _RESNET_MEAN), ("_resnet_std", _RESNET_STD)):
            self.register_buffer(name, torch.FloatTensor(value).view(1, 1, 3, 1, 1), persistent=False)

        self.init_weights()

    def _load_from_state_dict(
        self,
        state_dict,
        prefix,
        local_metadata,
        strict,
        missing_keys,
        unexpected_keys,
        error_msgs,
    ) -> None:
        """Remap a standard VGGT-Omega checkpoint when loading a hybrid model.

        Prefix blocks keep their original weights. The shared frame/global pair is
        initialized from ``shared_block_init_index`` and unused tail blocks are
        removed from the incoming state dictionary, so ``strict=True`` still works.
        """
        if self.is_hybrid:
            self._remap_standard_checkpoint_for_hybrid(state_dict, prefix)
        super()._load_from_state_dict(
            state_dict,
            prefix,
            local_metadata,
            strict,
            missing_keys,
            unexpected_keys,
            error_msgs,
        )

    def _remap_standard_checkpoint_for_hybrid(self, state_dict, prefix: str) -> None:
        if self.shared_block_init_index is None:
            raise RuntimeError("Hybrid aggregator has no shared block initialization index")

        for gate_name in ("shared_frame_gate", "shared_global_gate"):
            gate_key = f"{prefix}{gate_name}"
            if gate_key not in state_dict:
                gate = getattr(self, gate_name)
                if gate is None:
                    raise RuntimeError(f"Hybrid aggregator has no {gate_name}")
                state_dict[gate_key] = gate.detach().clone()

        block_mappings = (
            ("frame_blocks", "shared_frame_block"),
            ("inter_frame_blocks", "shared_inter_frame_block"),
        )
        for source_module, target_module in block_mappings:
            target_prefix = f"{prefix}{target_module}."
            if not any(key.startswith(target_prefix) for key in state_dict):
                source_prefix = f"{prefix}{source_module}.{self.shared_block_init_index}."
                source_keys = [key for key in state_dict if key.startswith(source_prefix)]
                for source_key in source_keys:
                    suffix = source_key[len(source_prefix) :]
                    state_dict[f"{target_prefix}{suffix}"] = state_dict[source_key]

            module_prefix = f"{prefix}{source_module}."
            for key in list(state_dict):
                if not key.startswith(module_prefix):
                    continue
                remainder = key[len(module_prefix) :]
                block_index_text = remainder.split(".", 1)[0]
                if block_index_text.isdigit() and int(block_index_text) >= self.prefix_depth:
                    del state_dict[key]

    def init_weights(self) -> None:
        nn.init.normal_(self.camera_token, std=1e-3)
        nn.init.normal_(self.register_token, std=1e-3)

    def forward(
        self,
        images: torch.Tensor,
    ) -> tuple[list[torch.Tensor | None], int]:
        batch_size, num_frames, num_channels, height, width = images.shape
        if num_channels != 3:
            raise ValueError(f"Expected 3 input channels, got {num_channels}")

        images = (images - self._resnet_mean) / self._resnet_std
        images = images.view(batch_size * num_frames, num_channels, height, width)

        camera_token = slice_expand_and_flatten(self.camera_token, batch_size, num_frames)
        register_token = slice_expand_and_flatten(self.register_token, batch_size, num_frames)

        patch_tokens = self.patch_embed(images)
        if isinstance(patch_tokens, dict):
            patch_tokens = patch_tokens["x_norm_patchtokens"]

        tokens = torch.cat([camera_token, register_token, patch_tokens], dim=1)
        _, num_tokens, embed_dim = tokens.shape

        patch_grid_size = (height // self.patch_size, width // self.patch_size)
        with torch.no_grad():
            rope_sin, rope_cos = self.rope_embed(H=patch_grid_size[0], W=patch_grid_size[1])
            frame_rope = (
                rope_sin.to(device=patch_tokens.device, dtype=torch.float32),
                rope_cos.to(device=patch_tokens.device, dtype=torch.float32),
            )

        outputs = []
        for block_idx in range(self.prefix_depth):
            tokens, frame_tokens = self._run_frame_block(
                tokens,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                self.frame_blocks[block_idx],
                frame_rope,
            )
            tokens = self._run_inter_frame_attention_block(
                tokens,
                batch_size,
                num_frames,
                num_tokens,
                embed_dim,
                self.inter_frame_blocks[block_idx],
                self.inter_frame_attention_types[block_idx],
            )
            if block_idx in self._cached_layer_indices_set:
                outputs.append(torch.cat([frame_tokens, tokens], dim=-1))
            else:
                outputs.append(None)

        if self.is_hybrid:
            if (
                self.shared_frame_block is None
                or self.shared_inter_frame_block is None
                or self.shared_frame_gate is None
                or self.shared_global_gate is None
            ):
                raise RuntimeError("Hybrid aggregator shared blocks are not initialized")
            for loop_idx in range(self.loop_steps):
                step_idx = self.prefix_depth + loop_idx
                tokens, frame_tokens = self._run_frame_block(
                    tokens,
                    batch_size,
                    num_frames,
                    num_tokens,
                    embed_dim,
                    self.shared_frame_block,
                    frame_rope,
                    self.shared_frame_gate,
                )
                tokens = self._run_inter_frame_attention_block(
                    tokens,
                    batch_size,
                    num_frames,
                    num_tokens,
                    embed_dim,
                    self.shared_inter_frame_block,
                    "global",
                    self.shared_global_gate,
                )
                if step_idx in self._cached_layer_indices_set:
                    outputs.append(torch.cat([frame_tokens, tokens], dim=-1))
                else:
                    outputs.append(None)

        return outputs, self.patch_token_start

    def _run_frame_block(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block: SelfAttentionBlock,
        rope_sincos: tuple[torch.Tensor, torch.Tensor],
        residual_gate: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        tokens = tokens.view(batch_size * num_frames, num_tokens, embed_dim)
        updated_tokens = block(tokens, rope_sincos)
        tokens = _apply_outer_residual_gate(tokens, updated_tokens, residual_gate)
        return tokens, tokens.view(batch_size, num_frames, num_tokens, embed_dim)

    def _run_inter_frame_attention_block(
        self,
        tokens: torch.Tensor,
        batch_size: int,
        num_frames: int,
        num_tokens: int,
        embed_dim: int,
        block: SelfAttentionBlock,
        attention_type: str,
        residual_gate: torch.Tensor | None = None,
    ) -> torch.Tensor:
        tokens = tokens.view(batch_size, num_frames, num_tokens, embed_dim)

        if attention_type == "global":
            tokens = tokens.view(batch_size, num_frames * num_tokens, embed_dim)
            updated_tokens = block(tokens, None)
            tokens = _apply_outer_residual_gate(tokens, updated_tokens, residual_gate)
            return tokens.view(batch_size, num_frames, num_tokens, embed_dim)

        if attention_type != "register":
            raise ValueError(f"Unknown inter-frame attention type: {attention_type}")

        patch_token_start = self.patch_token_start
        camera_and_register_tokens = tokens[:, :, :patch_token_start].reshape(
            batch_size,
            num_frames * patch_token_start,
            embed_dim,
        )
        patch_tokens = tokens[:, :, patch_token_start:].reshape(
            batch_size,
            num_frames * (num_tokens - patch_token_start),
            embed_dim,
        )

        updated_tokens = block(camera_and_register_tokens, None)
        camera_and_register_tokens = _apply_outer_residual_gate(camera_and_register_tokens, updated_tokens, residual_gate)
        tokens = torch.cat([camera_and_register_tokens, patch_tokens], dim=1)

        camera_and_register_tokens = tokens[:, : num_frames * patch_token_start].view(
            batch_size,
            num_frames,
            patch_token_start,
            embed_dim,
        )
        patch_tokens = tokens[:, num_frames * patch_token_start :].view(
            batch_size,
            num_frames,
            num_tokens - patch_token_start,
            embed_dim,
        )
        return torch.cat([camera_and_register_tokens, patch_tokens], dim=2)


def _apply_outer_residual_gate(
    input_tokens: torch.Tensor,
    updated_tokens: torch.Tensor,
    residual_gate: torch.Tensor | None,
) -> torch.Tensor:
    if residual_gate is None:
        return updated_tokens
    gate = residual_gate.to(dtype=updated_tokens.dtype, device=updated_tokens.device)
    return input_tokens + gate * (updated_tokens - input_tokens)


def _scale_cached_layer_indices(total_steps: int) -> tuple[int, ...]:
    if total_steps < len(_DEFAULT_CACHED_LAYER_INDICES):
        raise ValueError(
            f"Aggregator needs at least {len(_DEFAULT_CACHED_LAYER_INDICES)} total steps "
            "to provide the four DenseHead feature levels"
        )
    if total_steps == 24:
        return _DEFAULT_CACHED_LAYER_INDICES

    indices = tuple(
        round((layer_idx + 1) * total_steps / 24) - 1
        for layer_idx in _DEFAULT_CACHED_LAYER_INDICES
    )
    if len(set(indices)) != len(indices):
        indices = tuple(round((feature_idx + 1) * total_steps / 4) - 1 for feature_idx in range(4))
    return indices[:-1] + (total_steps - 1,)


def _validate_cached_layer_indices(cached_layer_indices: tuple[int, ...], total_steps: int) -> None:
    if len(cached_layer_indices) != 4:
        raise ValueError(f"DenseHead requires exactly four cached layers, got {cached_layer_indices}")
    if len(set(cached_layer_indices)) != len(cached_layer_indices):
        raise ValueError(f"cached_layer_indices must be unique, got {cached_layer_indices}")
    if tuple(sorted(cached_layer_indices)) != cached_layer_indices:
        raise ValueError(f"cached_layer_indices must be in increasing order, got {cached_layer_indices}")
    if cached_layer_indices[0] < 0 or cached_layer_indices[-1] >= total_steps:
        raise ValueError(
            f"cached_layer_indices must be within [0, {total_steps - 1}], got {cached_layer_indices}"
        )
    if cached_layer_indices[-1] != total_steps - 1:
        raise ValueError("cached_layer_indices must include the final execution step")


def _build_patch_embed(patch_size: int, embed_dim: int) -> DinoVisionTransformer:
    model = DinoVisionTransformer(
        img_size=224,
        patch_size=patch_size,
        in_chans=3,
        pos_embed_rope_base=100,
        pos_embed_rope_normalize_coords="max",
        pos_embed_rope_dtype="fp32",
        embed_dim=embed_dim,
        depth=24,
        num_heads=16,
        ffn_ratio=4,
        qkv_bias=True,
        drop_path_rate=0.0,
        layerscale_init=1.0e-5,
        norm_layer="layernormbf16",
        ffn_layer="mlp",
        ffn_bias=True,
        proj_bias=True,
        n_storage_tokens=4,
        mask_k_bias=True,
    )
    model.init_weights()
    return model


def slice_expand_and_flatten(token_tensor: torch.Tensor, batch_size: int, num_frames: int) -> torch.Tensor:
    first_frame_token = token_tensor[:, 0:1].expand(batch_size, 1, *token_tensor.shape[2:])
    other_frame_tokens = token_tensor[:, 1:].expand(batch_size, num_frames - 1, *token_tensor.shape[2:])
    tokens = torch.cat([first_frame_token, other_frame_tokens], dim=1)
    return tokens.view(batch_size * num_frames, *tokens.shape[2:])
