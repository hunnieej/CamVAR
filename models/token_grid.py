"""
Token Grid Buffers for L=2240 token grid mapping.

Pre-computed constant buffers mapping token index i ∈ [0..2239] to (scale_id, u, v, p_s).
Row-major ordering per scale: flatten order is u*p+v, scales concatenated coarse→fine.
"""

import torch
import torch.nn as nn


class TokenGridBuffers(nn.Module):
    """
    Pre-computed constant buffers for L=2240 token grid mapping.

    patch_nums = [1, 2, 3, 4, 6, 9, 13, 18, 24, 32]  # 10 scales
    Total tokens L = 1 + 4 + 9 + 16 + 36 + 81 + 169 + 324 + 576 + 1024 = 2240

    For each token index i:
    - scale_id: which scale this token belongs to (0-9)
    - u, v: coordinate within that scale's grid
    - p_s: patch size for that scale
    """

    def __init__(self, patch_nums=None):
        super().__init__()

        if patch_nums is None:
            patch_nums = [1, 2, 3, 4, 6, 9, 13, 18, 24, 32]

        self.patch_nums = patch_nums
        self.num_scales = len(patch_nums)
        self.L = sum(pn**2 for pn in patch_nums)

        # Verify L=2240
        assert self.L == 2240, f"Expected L=2240, got {self.L}"

        # Build constant buffers
        scale_ids = []
        u_coords = []
        v_coords = []
        patch_sizes = []

        for scale_idx, pn in enumerate(patch_nums):
            # Row-major order: for each row u, iterate columns v
            for u in range(pn):
                for v in range(pn):
                    scale_ids.append(scale_idx)
                    u_coords.append(u)
                    v_coords.append(v)
                    patch_sizes.append(pn)

        # Register as buffers (won't be trained, will move with model.to(device))
        self.register_buffer("scale_ids", torch.tensor(scale_ids, dtype=torch.long))
        self.register_buffer("u_coords", torch.tensor(u_coords, dtype=torch.long))
        self.register_buffer("v_coords", torch.tensor(v_coords, dtype=torch.long))
        self.register_buffer("patch_sizes", torch.tensor(patch_sizes, dtype=torch.long))

        # Verify shapes
        assert self.scale_ids.shape == (self.L,)
        assert self.u_coords.shape == (self.L,)
        assert self.v_coords.shape == (self.L,)
        assert self.patch_sizes.shape == (self.L,)

        print(
            f"TokenGridBuffers initialized: L={self.L}, scales={self.num_scales}, patch_nums={patch_nums}"
        )

    def get_coords(self):
        """
        Returns the coordinate buffers.

        Returns:
            scale_ids: (L,) - scale index for each token
            u_coords: (L,) - u coordinate
            v_coords: (L,) - v coordinate
            patch_sizes: (L,) - patch size for each token
        """
        return self.scale_ids, self.u_coords, self.v_coords, self.patch_sizes

    def verify_buffer(self):
        """Verify buffer construction is correct."""
        begin_ends = []
        cur = 0
        for pn in self.patch_nums:
            begin_ends.append((cur, cur + pn**2))
            cur += pn**2

        # Check each scale's tokens
        for scale_idx, (begin, end) in enumerate(begin_ends):
            scale_tokens = self.scale_ids[begin:end]
            assert torch.all(scale_tokens == scale_idx), (
                f"Scale {scale_idx} has wrong scale_ids"
            )

            pn = self.patch_nums[scale_idx]
            u_scale = self.u_coords[begin:end]
            v_scale = self.v_coords[begin:end]
            p_scale = self.patch_sizes[begin:end]

            assert torch.all(p_scale == pn), f"Scale {scale_idx} has wrong patch_sizes"
            assert torch.max(u_scale) == pn - 1, f"Scale {scale_idx} u_coords max wrong"
            assert torch.max(v_scale) == pn - 1, f"Scale {scale_idx} v_coords max wrong"
            assert torch.min(u_scale) == 0, f"Scale {scale_idx} u_coords min wrong"
            assert torch.min(v_scale) == 0, f"Scale {scale_idx} v_coords min wrong"

        print("TokenGridBuffers verification passed!")


if __name__ == "__main__":
    # Test the token grid buffers
    grid = TokenGridBuffers()
    grid.verify_buffer()

    scale_ids, u_coords, v_coords, patch_sizes = grid.get_coords()
    print(f"\nFirst 10 tokens:")
    for i in range(10):
        print(
            f"  Token {i}: scale={scale_ids[i].item()}, u={u_coords[i].item()}, "
            f"v={v_coords[i].item()}, p={patch_sizes[i].item()}"
        )

    print(f"\nLast 10 tokens:")
    for i in range(2230, 2240):
        print(
            f"  Token {i}: scale={scale_ids[i].item()}, u={u_coords[i].item()}, "
            f"v={v_coords[i].item()}, p={patch_sizes[i].item()}"
        )
