import math

import torch
import torch.nn as nn

from models.cubemap_geometry import CubemapPrecompute


class FaceRayAttentionAdapter(nn.Module):
    def __init__(
        self,
        embed_dim: int,
        adapter_dim: int,
        num_heads: int,
        ray_subdim: int,
        precompute: CubemapPrecompute,
        seq_offset: int,
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.adapter_dim = adapter_dim
        self.num_heads = num_heads
        self.head_dim = adapter_dim // num_heads
        self.ray_subdim = ray_subdim
        self.seq_offset = seq_offset
        self.scale = 1.0 / math.sqrt(self.head_dim)

        self.qkv = nn.Linear(embed_dim, adapter_dim * 3, bias=False)
        self.out_proj = nn.Linear(adapter_dim, embed_dim, bias=True)
        nn.init.zeros_(self.out_proj.weight)
        nn.init.zeros_(self.out_proj.bias)

        self.register_buffer(
            "boundary_face_ids", precompute.boundary_face_ids, persistent=False
        )
        self.register_buffer(
            "boundary_token_indices",
            precompute.boundary_token_indices,
            persistent=False,
        )
        self.register_buffer(
            "boundary_local_indices",
            precompute.boundary_local_indices,
            persistent=False,
        )
        self.register_buffer(
            "boundary_stage_ids", precompute.boundary_stage_ids, persistent=False
        )
        self.register_buffer(
            "neighbor_face_ids", precompute.neighbor_face_ids, persistent=False
        )
        self.register_buffer(
            "neighbor_token_indices",
            precompute.neighbor_token_indices,
            persistent=False,
        )
        self.register_buffer(
            "neighbor_local_indices",
            precompute.neighbor_local_indices,
            persistent=False,
        )
        self.register_buffer(
            "neighbor_stage_ids", precompute.neighbor_stage_ids, persistent=False
        )
        self.register_buffer(
            "neighbor_mask", precompute.neighbor_mask, persistent=False
        )
        self.register_buffer("d_ray", precompute.d_ray, persistent=False)

        self.register_buffer(
            "boundary_face_ids_last",
            precompute.boundary_face_ids_last,
            persistent=False,
        )
        self.register_buffer(
            "boundary_local_indices_last",
            precompute.boundary_local_indices_last,
            persistent=False,
        )
        self.register_buffer(
            "neighbor_face_ids_last",
            precompute.neighbor_face_ids_last,
            persistent=False,
        )
        self.register_buffer(
            "neighbor_local_indices_last",
            precompute.neighbor_local_indices_last,
            persistent=False,
        )
        self.register_buffer(
            "neighbor_mask_last", precompute.neighbor_mask_last, persistent=False
        )

    def forward(self, x: torch.Tensor, face_group: int = 1) -> torch.Tensor:
        if face_group <= 1:
            return torch.zeros_like(x)

        sampler_mode = False
        if x.dim() == 3:
            b, l, _ = x.shape
            if b % face_group == 0:
                b = b // face_group
                x_tokens = x.view(b, face_group, l, self.embed_dim)
            elif (l - 1) % face_group == 0:
                sampler_mode = True
                per_face = (l - 1) // face_group
                x_tokens = x[:, 1:, :].view(b, face_group, per_face, self.embed_dim)
            else:
                raise ValueError("Batch size must be divisible by face_group")
        elif x.dim() == 4:
            b, _, l, _ = x.shape
            x_tokens = x
        else:
            raise ValueError("Unexpected x shape for FaceRayAttentionAdapter")

        if self.boundary_face_ids.numel() == 0:
            return torch.zeros_like(x).view(-1, l, self.embed_dim)

        if sampler_mode:
            q_faces = self.boundary_face_ids_last
            q_indices = self.boundary_local_indices_last
            q_local = self.boundary_local_indices_last
            neighbor_faces = self.neighbor_face_ids_last
            neighbor_indices = self.neighbor_local_indices_last
            neighbor_local = self.neighbor_local_indices_last
            neighbor_mask = self.neighbor_mask_last
            stage_ids = torch.full_like(q_indices, self.d_ray.shape[0] - 1)
            neighbor_stage_ids = torch.full_like(
                neighbor_indices, self.d_ray.shape[0] - 1
            )
        else:
            q_faces = self.boundary_face_ids
            q_indices = self.boundary_token_indices
            q_local = self.boundary_local_indices
            neighbor_faces = self.neighbor_face_ids
            neighbor_indices = self.neighbor_token_indices
            neighbor_local = self.neighbor_local_indices
            neighbor_mask = self.neighbor_mask
            stage_ids = self.boundary_stage_ids
            neighbor_stage_ids = self.neighbor_stage_ids

        max_face = x_tokens.shape[1] - 1
        max_idx = x_tokens.shape[2] - 1
        q_faces_clamped = q_faces.clamp(min=0, max=max_face)
        q_indices_clamped = q_indices.clamp(min=0, max=max_idx)
        valid_q = (q_faces == q_faces_clamped) & (q_indices == q_indices_clamped)

        x_q = x_tokens[:, q_faces_clamped, q_indices_clamped, :]
        qkv_q = self.qkv(x_q)
        q, _, _ = qkv_q.chunk(3, dim=-1)

        safe_neighbor_faces = neighbor_faces.clamp(min=0, max=max_face)
        safe_neighbor_indices = neighbor_indices.clamp(min=0, max=max_idx)
        neighbor_valid = (neighbor_faces == safe_neighbor_faces) & (
            neighbor_indices == safe_neighbor_indices
        )
        neighbor_mask = neighbor_mask & neighbor_valid
        x_kv = x_tokens[:, safe_neighbor_faces, safe_neighbor_indices, :]
        qkv_kv = self.qkv(x_kv)
        _, k, v = qkv_kv.chunk(3, dim=-1)

        bsz = x_tokens.shape[0]
        nb = q.shape[1]
        max_n = k.shape[2]

        q = q.view(bsz, nb, self.num_heads, self.head_dim)
        k = k.view(bsz, nb, max_n, self.num_heads, self.head_dim)
        v = v.view(bsz, nb, max_n, self.num_heads, self.head_dim)

        q = self._apply_dray(
            q, self._query_dray(stage_ids, q_faces, q_local), inverse=False
        )
        k = self._apply_dray(
            k,
            self._key_dray(neighbor_stage_ids, neighbor_faces, neighbor_local),
            inverse=False,
        )
        v = self._apply_dray(
            v,
            self._key_dray(neighbor_stage_ids, neighbor_faces, neighbor_local),
            inverse=False,
        )

        attn = (q.unsqueeze(2) * k).sum(dim=-1) * self.scale
        attn = attn.masked_fill(
            ~neighbor_mask.unsqueeze(0).unsqueeze(-1), float("-inf")
        )
        attn = torch.softmax(attn, dim=2)
        out = (attn.unsqueeze(-1) * v).sum(dim=2)

        out = self._apply_dray(
            out, self._query_dray(stage_ids, q_faces, q_local), inverse=True
        )
        out = out.reshape(bsz, nb, self.adapter_dim)
        out = self.out_proj(out)

        out_full = torch.zeros_like(x_tokens)
        out = out * valid_q.unsqueeze(0).unsqueeze(-1).to(out.dtype)
        out_full[:, q_faces_clamped, q_indices_clamped, :] = out.to(out_full.dtype)

        if sampler_mode:
            out_seq = out_full.reshape(bsz, -1, self.embed_dim)
            out_seq = torch.cat(
                [
                    torch.zeros(
                        bsz,
                        1,
                        self.embed_dim,
                        device=out_seq.device,
                        dtype=out_seq.dtype,
                    ),
                    out_seq,
                ],
                dim=1,
            )
            return out_seq

        return out_full.view(-1, l, self.embed_dim)

    def _query_dray(
        self, stage_ids: torch.Tensor, face_ids: torch.Tensor, local_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.d_ray[stage_ids, face_ids, local_ids]

    def _key_dray(
        self, stage_ids: torch.Tensor, face_ids: torch.Tensor, local_ids: torch.Tensor
    ) -> torch.Tensor:
        return self.d_ray[
            stage_ids,
            face_ids.clamp(min=0),
            local_ids.clamp(min=0),
        ]

    def _apply_dray(
        self, x: torch.Tensor, d_ray: torch.Tensor, inverse: bool
    ) -> torch.Tensor:
        ray_dim = self.ray_subdim
        pass_dim = self.head_dim - ray_dim
        r = ray_dim // 3

        if x.dim() == 4:
            ray = x[..., :ray_dim].reshape(x.shape[0], x.shape[1], x.shape[2], 3, r)
            d = d_ray.unsqueeze(0).unsqueeze(2)
            if inverse:
                d = d.transpose(-2, -1)
            ray = torch.einsum("bnwij,bnwjr->bnwir", d, ray)
            ray = ray.reshape(x.shape[0], x.shape[1], x.shape[2], ray_dim)
            if pass_dim > 0:
                return torch.cat([ray, x[..., ray_dim:]], dim=-1)
            return ray

        if x.dim() == 5:
            ray = x[..., :ray_dim].reshape(
                x.shape[0], x.shape[1], x.shape[2], x.shape[3], 3, r
            )
            d = d_ray.unsqueeze(0).unsqueeze(3)
            if inverse:
                d = d.transpose(-2, -1)
            ray = torch.einsum("bnmwij,bnmwjr->bnmwir", d, ray)
            ray = ray.reshape(x.shape[0], x.shape[1], x.shape[2], x.shape[3], ray_dim)
            if pass_dim > 0:
                return torch.cat([ray, x[..., ray_dim:]], dim=-1)
            return ray

        raise ValueError("Unexpected tensor shape for D_ray application")
