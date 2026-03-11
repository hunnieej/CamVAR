"""
RayAdapter: Ray-aware adaptation module with CameraAwareAttention.

Implements:
1. PerBlockGate: Per-block MLP gate with LayerNorm, sigmoid bounded output,
                 and soft-open initialization (Option B redesign)
2. CameraAwareAttention: Self-attention with CamRoPE and memory integration
3. RayAdapter: Complete adapter module with P_down, SA_cam, P_up
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from .camera_system import CamRoPE


class PerBlockGate(nn.Module):
    """
    Per-block adaptive gate — Option B redesign.

    Architecture:
        LayerNorm(c_t) → Linear(64→64) → SiLU → Linear(64→1) → gate_max * sigmoid(x / T)

    Initialization:
        - First layer: default PyTorch init (Kaiming uniform)
        - Final layer weights: small normal (std=0.01)
        - Final layer bias: logit(gate_init / gate_max) so initial output ≈ gate_init

    Input:  c_t  (B, 64) — camera features from CameraEmbedder
    Output: gate (B, 1, 1) — broadcast over L and C dimensions

    Args:
        camera_dim:      Input feature dimension (default 64)
        gate_max:        Upper bound on gate output (default 0.1)
        gate_init:       Target initial gate value (default 0.03)
        gate_temperature: Sigmoid temperature T (default 1.0)
    """

    def __init__(
        self,
        camera_dim: int = 64,
        gate_max: float = 0.1,
        gate_init: float = 0.03,
        gate_temperature: float = 1.0,
    ):
        super().__init__()

        self.gate_max = gate_max
        self.gate_temperature = gate_temperature

        # Normalize camera features before the MLP
        self.ln = nn.LayerNorm(camera_dim)

        # MLP: 64 → 64 → 1
        self.fc1 = nn.Linear(camera_dim, camera_dim)
        self.act = nn.SiLU()
        self.fc2 = nn.Linear(camera_dim, 1)

        # Soft-open initialization
        # Solve: gate_max * sigmoid(bias / T) = gate_init
        #   => sigmoid(bias / T) = gate_init / gate_max
        #   => bias = T * logit(gate_init / gate_max)
        ratio = gate_init / gate_max
        # clamp ratio away from 0 and 1 to keep logit finite
        ratio = max(1e-4, min(1 - 1e-4, ratio))
        init_bias = gate_temperature * math.log(ratio / (1.0 - ratio))

        # fc1: default PyTorch init (Kaiming), no change needed
        # fc2: small weights, bias set to achieve soft-open target
        nn.init.normal_(self.fc2.weight, mean=0.0, std=0.01)
        nn.init.constant_(self.fc2.bias, init_bias)

    def forward(self, c_t: torch.Tensor) -> torch.Tensor:
        """
        c_t: (B, 64) — camera features
        Returns: (B, 1, 1) — gate value in (0, gate_max)
        """
        B = c_t.shape[0]
        assert c_t.shape == (B, 64), f"Expected c_t shape (B, 64), got {c_t.shape}"

        x = self.ln(c_t)  # (B, 64) — normalize scale
        x = self.act(self.fc1(x))  # (B, 64)
        x = self.fc2(x)  # (B, 1)
        gate = self.gate_max * torch.sigmoid(
            x / self.gate_temperature
        )  # (B, 1) ∈ (0, gate_max)
        gate = gate.unsqueeze(2)  # (B, 1, 1)
        return gate


class CameraAwareAttention(nn.Module):
    """
    Self-attention with camera-aware positional encoding and memory integration.

    Key features:
    - Operates on bottleneck dimension (adapter_dim=128)
    - Q from tokens only (length L)
    - K,V from concat(tokens, memory) (length L+M)
    - CamRoPE applied to Q and K_tokens (not memory K)
    - Output length remains L
    """

    def __init__(self, dim=128, num_heads=4, head_dim=32):
        super().__init__()

        self.dim = dim
        self.num_heads = num_heads
        self.head_dim = head_dim
        # self.scale = 0.25 / (head_dim**0.5)
        self.scale = 1 / (head_dim**0.5)

        # Q projection (from tokens only)
        self.q_proj = nn.Linear(dim, num_heads * head_dim, bias=False)
        self.q_bias = nn.Parameter(torch.zeros(num_heads * head_dim))

        # KV projection (from tokens + memory)
        self.kv_proj = nn.Linear(dim, 2 * num_heads * head_dim, bias=False)
        self.k_bias = nn.Parameter(torch.zeros(num_heads * head_dim))
        self.v_bias = nn.Parameter(torch.zeros(num_heads * head_dim))

        # Output projection
        self.out_proj = nn.Linear(num_heads * head_dim, dim)

        # print(
        #     f"CameraAwareAttention initialized: dim={dim}, num_heads={num_heads}, head_dim={head_dim}"
        # )

    def forward(self, u_c, memory, theta):
        """
        Compute camera-aware self-attention with memory.

        Args:
            u_c: (B, L, 128) - token features in bottleneck space
            memory: (B, M, 128) - memory features
            theta: (B, L, 16) - shared theta from CamRoPE

        Returns:
            attn_out: (B, L, 128) - attended token features
        """
        B, L, C = u_c.shape
        M = memory.shape[1]

        assert u_c.shape == (B, L, self.dim), (
            f"Expected u_c shape ({B}, {L}, {self.dim}), got {u_c.shape}"
        )
        assert memory.shape == (B, M, self.dim), (
            f"Expected memory shape ({B}, {M}, {self.dim}), got {memory.shape}"
        )
        assert theta.shape == (B, L, 16), (
            f"Expected theta shape ({B}, {L}, 16), got {theta.shape}"
        )

        # Compute Q from tokens only
        q = F.linear(u_c, self.q_proj.weight, self.q_bias)  # (B, L, num_heads*head_dim)
        q = q.view(B, L, self.num_heads, self.head_dim)  # (B, L, num_heads, head_dim)

        # Compute KV from tokens + memory
        tokens_and_memory = torch.cat([u_c, memory], dim=1)  # (B, L+M, dim)
        kv = F.linear(
            tokens_and_memory,
            self.kv_proj.weight,
            torch.cat([self.k_bias, self.v_bias]),
        )  # (B, L+M, 2*num_heads*head_dim)
        kv = kv.view(
            B, L + M, 2, self.num_heads, self.head_dim
        )  # (B, L+M, 2, num_heads, head_dim)
        k, v = kv.unbind(dim=2)  # Each: (B, L+M, num_heads, head_dim)

        # Apply CamRoPE to Q and K_tokens (not memory K)
        q_rot, k_rot = CamRoPE.apply_rotary(q, k, theta, apply_to_k=True)

        # Compute attention: Q @ K^T
        q_rot = q_rot.transpose(1, 2)  # (B, num_heads, L, head_dim)
        k_rot = k_rot.transpose(1, 2)  # (B, num_heads, L+M, head_dim)
        v = v.transpose(1, 2)  # (B, num_heads, L+M, head_dim)

        attn_scores = (
            torch.matmul(q_rot, k_rot.transpose(-2, -1)) * self.scale
        )  # (B, num_heads, L, L+M)
        attn_weights = F.softmax(attn_scores, dim=-1)

        # Apply attention to V
        attn_output = torch.matmul(attn_weights, v)  # (B, num_heads, L, head_dim)
        attn_output = attn_output.transpose(
            1, 2
        ).contiguous()  # (B, L, num_heads, head_dim)
        attn_output = attn_output.view(
            B, L, self.num_heads * self.head_dim
        )  # (B, L, num_heads*head_dim)

        # Output projection
        out = self.out_proj(attn_output)  # (B, L, dim)

        return out


class RayAdapter(nn.Module):
    """
    Ray-aware adaptation module with CORRECT initialization.

    Architecture:
    - P_down: Linear(1920 → 128) with NORMAL init (NOT zero)
    - SA_cam: CameraAwareAttention
    - P_up: Linear(128 → 1920) with NORMAL init (NOT zero)

    CRITICAL: P_down and P_up use default PyTorch initialization (NOT zero)
    to enable gradient flow. Only the gate is zero-initialized.
    """

    def __init__(self, embed_dim=1920, adapter_dim=128, num_heads=4, head_dim=32):
        super().__init__()

        self.embed_dim = embed_dim
        self.adapter_dim = adapter_dim

        # NORMAL initialization (NOT zero) - enables gradient flow
        self.p_down = nn.Linear(embed_dim, adapter_dim)
        self.p_up = nn.Linear(adapter_dim, embed_dim)

        # Camera-aware self-attention
        self.sa_cam = CameraAwareAttention(
            dim=adapter_dim, num_heads=num_heads, head_dim=head_dim
        )

        # print(
        #     f"RayAdapter initialized: embed_dim={embed_dim}, adapter_dim={adapter_dim}"
        # )
        # print(f"  P_down/P_up use NORMAL initialization (NOT zero)")

    def forward(self, u, memory, theta):
        """
        Process pre-normalized input with camera-aware attention.

        Args:
            u: (B, L, 1920) - pre-norm input from block
            memory: (B, M, 128) - shared model-level memory
            theta: (B, L, 16) - theta from CamRoPE

        Returns:
            delta_x_ray: (B, L, 1920) - ray adaptation residual (gating applied externally)

        Internal flow:
        1. u_c = P_down(u) → (B, L, 128)
        2. attn_out = SA_cam(u_c, memory, theta) → (B, L, 128)
        3. delta_x_ray = P_up(attn_out) → (B, L, 1920)
        """
        B, L, C = u.shape
        assert u.shape == (B, L, self.embed_dim), (
            f"Expected u shape ({B}, {L}, {self.embed_dim}), got {u.shape}"
        )

        # Project down to bottleneck
        u_c = self.p_down(u)  # (B, L, 1920) -> (B, L, 128)

        # Camera-aware self-attention with memory
        attn_out = self.sa_cam(u_c, memory, theta)  # (B, L, 128)

        # Project back to embedding dimension
        delta_x_ray = self.p_up(attn_out)  # (B, L, 128) -> (B, L, 1920)

        return delta_x_ray

    def verify_normal_init(self):
        """Verify that P_down and P_up are NOT zero-initialized."""
        p_down_std = self.p_down.weight.std().item()
        p_up_std = self.p_up.weight.std().item()

        assert p_down_std > 0.001, (
            f"P_down should have normal init, got std={p_down_std}"
        )
        assert p_up_std > 0.001, f"P_up should have normal init, got std={p_up_std}"

        print(
            f"✓ RayAdapter initialization verified: P_down std={p_down_std:.4f}, P_up std={p_up_std:.4f}"
        )


if __name__ == "__main__":
    print("Testing PerBlockGate (Option B)...")
    gate = PerBlockGate(gate_max=0.1, gate_init=0.03, gate_temperature=1.0)

    c_t = torch.randn(2, 64)
    gate_val = gate(c_t)
    print(f"  Input: {c_t.shape}, Output: {gate_val.shape}")
    assert gate_val.shape == (2, 1, 1)
    assert gate_val.min().item() > 0, "Gate should be > 0 (sigmoid bounded)"
    assert gate_val.max().item() < 0.1, "Gate should be < gate_max=0.1"
    print(f"  Gate output at init (should be ~0.03): {gate_val.mean().item():.4f}")
    print(f"  Gate range: [{gate_val.min().item():.4f}, {gate_val.max().item():.4f}]")

    print("\nTesting CameraAwareAttention...")
    sa_cam = CameraAwareAttention()
    u_c = torch.randn(2, 2240, 128)
    memory = torch.randn(2, 32, 128)
    theta = torch.randn(2, 2240, 16)

    attn_out = sa_cam(u_c, memory, theta)
    print(f"  u_c: {u_c.shape}, memory: {memory.shape}, theta: {theta.shape}")
    print(f"  Output: {attn_out.shape}")
    assert attn_out.shape == (2, 2240, 128)

    print("\nTesting RayAdapter...")
    adapter = RayAdapter()
    adapter.verify_normal_init()

    u = torch.randn(2, 2240, 1920)
    delta_x_ray = adapter(u, memory, theta)
    print(f"  u: {u.shape}, memory: {memory.shape}, theta: {theta.shape}")
    print(f"  delta_x_ray: {delta_x_ray.shape}")
    assert delta_x_ray.shape == (2, 2240, 1920)

    # Verify delta_x_ray is non-zero (from normal init)
    assert delta_x_ray.abs().max() > 0.01, (
        "delta_x_ray should be non-zero from normal init"
    )
    print(
        f"  delta_x_ray magnitude: {delta_x_ray.abs().max().item():.4f} (should be non-zero)"
    )

    print("\nAll RayAdapter tests passed!")
