"""
Camera System: CameraEmbedder, RayConstructor, and CamRoPE.

Implements:
1. CameraEmbedder: MLP to embed camera direction (B,3) → (B,64)
2. RayConstructor: Builds world-space ray directions for each token
3. CamRoPE: Camera-aware Rotary Position Encoding with shared theta mapping
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F


class CameraEmbedder(nn.Module):
    """
    MLP to embed camera direction into camera features.

    Input: cam_dir (B, 3) - unit camera direction vector
    Output: camera features (B, 64)
    """

    def __init__(self, input_dim=3, hidden_dim=64, output_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, output_dim),
        )

    def forward(self, cam_dir):
        """
        cam_dir: (B, 3) - camera direction (should be unit vectors)
        Returns: (B, 64) - camera features
        """
        # Validate input
        B = cam_dir.shape[0]
        assert cam_dir.shape == (B, 3), (
            f"Expected cam_dir shape (B,3), got {cam_dir.shape}"
        )

        # Check unit vectors (within tolerance)
        norms = torch.norm(cam_dir, dim=-1)
        assert torch.all(torch.abs(norms - 1.0) < 1e-3), (
            f"Camera directions must be unit vectors, got norms: {norms}"
        )

        return self.mlp(cam_dir)


class RayConstructor:
    """
    Constructs world-space ray directions for each token based on camera parameters.

    Given:
    - cam_dir: (B, 3) camera forward direction (unit vector)
    - token coordinates (u, v) and patch size p_s
    - FoV and patch_size parameters

    Computes:
    - r_world: (B, L, 3) world-space ray direction for each token
    """

    @staticmethod
    def build_camera_matrix(cam_dir, world_up=None, alt_up=None, parallel_thresh=0.99):
        """
        Build camera-to-world rotation matrix R_c2w.

        Args:
            cam_dir: (B, 3) - camera forward direction (unit vector)
            world_up: (3,) - world up vector, default [0, 1, 0]
            alt_up: (3,) - alternative up vector for parallel case, default [1, 0, 0]
            parallel_thresh: float - threshold for detecting parallel vectors

        Returns:
            R_c2w: (B, 3, 3) - rotation matrix, columns are [right, up, forward]
        """
        if world_up is None:
            world_up = torch.tensor(
                [0.0, 1.0, 0.0], device=cam_dir.device, dtype=cam_dir.dtype
            )
        if alt_up is None:
            alt_up = torch.tensor(
                [1.0, 0.0, 0.0], device=cam_dir.device, dtype=cam_dir.dtype
            )

        B = cam_dir.shape[0]

        # Normalize forward direction
        forward = F.normalize(cam_dir, dim=-1)  # (B, 3)

        # Check if forward is parallel to world_up
        world_up_expanded = world_up.unsqueeze(0).expand(B, -1)  # (B, 3)
        dot = torch.abs((forward * world_up_expanded).sum(dim=-1))  # (B,)

        # Use alt_up when parallel
        is_parallel = dot > parallel_thresh  # (B,)
        alt_up_expanded = alt_up.unsqueeze(0).expand(B, -1)  # (B, 3)

        up_vec = torch.where(
            is_parallel.unsqueeze(-1), alt_up_expanded, world_up_expanded
        )  # (B, 3)

        # Compute right = forward × up
        right = torch.cross(forward, up_vec, dim=-1)  # (B, 3)
        right = F.normalize(right, dim=-1)

        # Recompute up = right × forward (ensure orthogonality)
        up = torch.cross(right, forward, dim=-1)  # (B, 3)
        up = F.normalize(up, dim=-1)

        # Stack into rotation matrix: columns are [right, up, forward]
        R_c2w = torch.stack([right, up, forward], dim=-1)  # (B, 3, 3)

        return R_c2w

    @staticmethod
    def compute_rays(cam_dir, u_coords, v_coords, patch_sizes, fov_deg, patch_size):
        """
        Compute world-space ray directions for all tokens.

        Args:
            cam_dir: (B, 3) - camera direction (unit vector)
            u_coords: (L,) - u coordinates for all tokens
            v_coords: (L,) - v coordinates for all tokens
            patch_sizes: (L,) - patch size p_s for all tokens
            fov_deg: float - field of view in degrees
            patch_size: int - base patch size for normalization

        Returns:
            r_world: (B, L, 3) - world-space ray directions (unit vectors)
        """
        B = cam_dir.shape[0]
        L = u_coords.shape[0]
        device = cam_dir.device
        dtype = cam_dir.dtype

        # Convert coordinates to float
        u = u_coords.float().to(device)  # (L,)
        v = v_coords.float().to(device)  # (L,)
        p_s = patch_sizes.float().to(device)  # (L,)

        # Compute normalized coordinates in [-1, 1]
        # x_n = 2 * ((u + 0.5) / p_s - 0.5)
        x_n = 2.0 * ((u + 0.5) / p_s - 0.5)  # (L,)
        y_n = 2.0 * ((v + 0.5) / p_s - 0.5)  # (L,)

        # Apply FoV
        alpha = (fov_deg * math.pi / 180.0) / 2.0
        tan_alpha = math.tan(alpha)

        x_c = x_n * tan_alpha  # (L,)
        y_c = y_n * tan_alpha  # (L,)
        z_c = torch.ones_like(x_c)  # (L,)

        # Camera-space rays (before normalization)
        r_cam = torch.stack([x_c, y_c, z_c], dim=-1)  # (L, 3)
        r_cam = F.normalize(r_cam, dim=-1)  # (L, 3)

        # Build R_c2w
        R_c2w = RayConstructor.build_camera_matrix(cam_dir)  # (B, 3, 3)

        # Transform to world space: r_world = R_c2w @ r_cam
        # r_cam: (L, 3) -> (1, L, 3, 1)
        # R_c2w: (B, 3, 3) -> (B, 1, 3, 3)
        r_cam_expanded = r_cam.unsqueeze(0).unsqueeze(-1)  # (1, L, 3, 1)
        R_c2w_expanded = R_c2w.unsqueeze(1)  # (B, 1, 3, 3)

        # Matrix multiplication
        r_world = torch.matmul(R_c2w_expanded, r_cam_expanded)  # (B, L, 3, 1)
        r_world = r_world.squeeze(-1)  # (B, L, 3)

        # Normalize (should already be normalized, but ensure)
        r_world = F.normalize(r_world, dim=-1)

        return r_world


class CamRoPE(nn.Module):
    """
    Camera-aware Rotary Position Encoding.

    Uses shared theta mapping: Linear(3 → 16) producing theta (B,L,16)
    which is then broadcast to heads as (B,L,1,16) during rotary application.

    This is applied ONLY to Q and K_tokens (not memory K) in SA_cam.
    """

    def __init__(self, d_cam=32):
        super().__init__()

        # Validate d_cam is even
        assert d_cam % 2 == 0, f"d_cam must be even for rotary pairs, got {d_cam}"

        self.d_cam = d_cam
        self.rotary_pairs = d_cam // 2  # 16 pairs for d_cam=32

        # Shared theta mapping: Linear(3 → rotary_pairs)
        self.theta_linear = nn.Linear(3, self.rotary_pairs)

        print(f"CamRoPE initialized: d_cam={d_cam}, rotary_pairs={self.rotary_pairs}")

    def forward(self, r_world):
        """
        Compute theta values from world-space ray directions.

        Args:
            r_world: (B, L, 3) - world-space ray directions

        Returns:
            theta: (B, L, 16) - shared theta for all heads (NOT head-wise)
        """
        B, L, _ = r_world.shape
        assert r_world.shape[2] == 3, (
            f"Expected r_world last dim 3, got {r_world.shape}"
        )

        # Apply shared theta mapping
        theta = self.theta_linear(r_world)  # (B, L, 3) -> (B, L, 16)

        assert theta.shape == (B, L, self.rotary_pairs), (
            f"Expected theta shape ({B}, {L}, {self.rotary_pairs}), got {theta.shape}"
        )

        return theta

    @staticmethod
    def apply_rotary(q, k, theta, apply_to_k=True):
        """
        Apply rotary position encoding to Q and optionally K.

        Args:
            q: (B, L, num_heads, head_dim) - query
            k: (B, L_k, num_heads, head_dim) - key (L_k can be L or L+M)
            theta: (B, L, rotary_pairs) - theta values (shared across heads)
            apply_to_k: bool - whether to apply to K (False for memory segment)

        Returns:
            q_rot: (B, L, num_heads, head_dim) - rotated query
            k_rot: (B, L_k, num_heads, head_dim) - rotated key
        """
        B, L_q, num_heads, head_dim = q.shape
        B_k, L_k, num_heads_k, head_dim_k = k.shape

        assert num_heads == num_heads_k and head_dim == head_dim_k
        rotary_pairs = theta.shape[2]

        # Broadcast theta to heads: (B, L, rotary_pairs) -> (B, L, 1, rotary_pairs)
        theta_expanded = theta.unsqueeze(2)  # (B, L, 1, rotary_pairs)

        # Apply rotary to Q (all tokens)
        q_rot = CamRoPE._apply_rotary_single(q, theta_expanded, rotary_pairs)

        # Apply rotary to K (only if requested and only to token part)
        if apply_to_k and L_k == L_q:
            # K is tokens only, apply to all
            k_rot = CamRoPE._apply_rotary_single(k, theta_expanded, rotary_pairs)
        elif apply_to_k and L_k > L_q:
            # K includes memory, apply only to token part
            k_tokens = k[:, :L_q, :, :]  # (B, L, num_heads, head_dim)
            k_memory = k[:, L_q:, :, :]  # (B, M, num_heads, head_dim)

            k_tokens_rot = CamRoPE._apply_rotary_single(
                k_tokens, theta_expanded, rotary_pairs
            )
            k_rot = torch.cat(
                [k_tokens_rot, k_memory], dim=1
            )  # (B, L+M, num_heads, head_dim)
        else:
            k_rot = k

        return q_rot, k_rot

    @staticmethod
    def _apply_rotary_single(x, theta, rotary_pairs):
        """
        Apply rotary encoding to a single tensor.

        Args:
            x: (B, L, num_heads, head_dim) - input
            theta: (B, L, 1, rotary_pairs) - rotation angles

        Returns:
            x_rot: (B, L, num_heads, head_dim) - rotated output
        """
        B, L, num_heads, head_dim = x.shape

        # Only apply to first d_cam dimensions (rotary_pairs * 2)
        d_cam = rotary_pairs * 2
        if head_dim < d_cam:
            # If head_dim < d_cam, pad or only use available dims
            d_cam = head_dim
            rotary_pairs = d_cam // 2

        # Split into pairs
        x_pairs = x[:, :, :, : rotary_pairs * 2].reshape(
            B, L, num_heads, rotary_pairs, 2
        )  # (B,L,H,pairs,2)
        x_rest = x[:, :, :, rotary_pairs * 2 :]  # (B,L,H,remaining)

        # Compute cos and sin
        cos_theta = torch.cos(theta).unsqueeze(-1)  # (B, L, 1, rotary_pairs, 1)
        sin_theta = torch.sin(theta).unsqueeze(-1)  # (B, L, 1, rotary_pairs, 1)

        # Apply rotation matrix [[cos, -sin], [sin, cos]]
        x0 = x_pairs[..., 0:1]  # (B, L, H, rotary_pairs, 1)
        x1 = x_pairs[..., 1:2]  # (B, L, H, rotary_pairs, 1)

        x0_rot = x0 * cos_theta - x1 * sin_theta
        x1_rot = x0 * sin_theta + x1 * cos_theta

        x_pairs_rot = torch.cat([x0_rot, x1_rot], dim=-1)  # (B, L, H, rotary_pairs, 2)
        x_pairs_rot = x_pairs_rot.reshape(B, L, num_heads, rotary_pairs * 2)

        # Concatenate with rest
        if x_rest.numel() > 0:
            x_rot = torch.cat([x_pairs_rot, x_rest], dim=-1)
        else:
            x_rot = x_pairs_rot

        return x_rot


if __name__ == "__main__":
    # Test CameraEmbedder
    print("Testing CameraEmbedder...")
    cam_embedder = CameraEmbedder()
    cam_dir = torch.randn(2, 3)
    cam_dir = F.normalize(cam_dir, dim=-1)
    cam_feat = cam_embedder(cam_dir)
    print(f"  Input: {cam_dir.shape}, Output: {cam_feat.shape}")
    assert cam_feat.shape == (2, 64)

    # Test RayConstructor
    print("\nTesting RayConstructor...")
    from token_grid import TokenGridBuffers

    grid = TokenGridBuffers()
    scale_ids, u_coords, v_coords, patch_sizes = grid.get_coords()

    cam_dir = torch.tensor([[0.0, 0.0, 1.0], [1.0, 0.0, 0.0]])  # 2 cameras
    r_world = RayConstructor.compute_rays(
        cam_dir, u_coords, v_coords, patch_sizes, fov_deg=60.0, patch_size=16
    )
    print(f"  cam_dir: {cam_dir.shape}, r_world: {r_world.shape}")
    assert r_world.shape == (2, 2240, 3)

    # Test CamRoPE
    print("\nTesting CamRoPE...")
    cam_rope = CamRoPE(d_cam=32)
    theta = cam_rope(r_world)
    print(f"  r_world: {r_world.shape}, theta: {theta.shape}")
    assert theta.shape == (2, 2240, 16)

    # Test rotary application
    q = torch.randn(2, 2240, 4, 32)
    k = torch.randn(2, 2240, 4, 32)
    q_rot, k_rot = CamRoPE.apply_rotary(q, k, theta)
    print(f"  q: {q.shape} -> q_rot: {q_rot.shape}")
    print(f"  k: {k.shape} -> k_rot: {k_rot.shape}")

    # Test with memory
    k_with_mem = torch.randn(2, 2272, 4, 32)  # 2240 + 32 memory
    q_rot, k_rot_mem = CamRoPE.apply_rotary(q, k_with_mem, theta)
    print(f"  k_with_mem: {k_with_mem.shape} -> k_rot_mem: {k_rot_mem.shape}")
    assert k_rot_mem.shape == (2, 2272, 4, 32)

    print("\nAll camera system tests passed!")
