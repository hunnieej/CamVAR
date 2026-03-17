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
import dist as _dist_dbg


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

    def __init__(
        self,
        dim=128,
        num_heads=4,
        head_dim=32,
        theta_gain_value: float = 12.0,
        temp_gain_value: float = 12.0,
        warm_start_steps: int = 0,
        warm_theta_gain_value: float = 12.0,
        warm_temp_gain_value: float = 12.0,
        warm_unfreeze: bool = True,
    ):
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

        # Gains (default fixed), optional warm-start scheduling
        tg_init = float(theta_gain_value)
        tmpg_init = float(temp_gain_value)
        self.theta_gain_param = nn.Parameter(torch.tensor(tg_init), requires_grad=False)
        self.temp_gain_param = nn.Parameter(
            torch.tensor(tmpg_init), requires_grad=False
        )
        self.register_buffer("warm_start_steps", torch.tensor(int(warm_start_steps)))
        self.register_buffer(
            "warm_theta_gain_value", torch.tensor(float(warm_theta_gain_value))
        )
        self.register_buffer(
            "warm_temp_gain_value", torch.tensor(float(warm_temp_gain_value))
        )
        self.warm_unfreeze = warm_unfreeze

        # print(
        #     f"CameraAwareAttention initialized: dim={dim}, num_heads={num_heads}, head_dim={head_dim}"
        # )

    def forward(
        self,
        u_c,
        memory,
        theta,
        disable_memory_kv=False,
        attn_debug=False,
        attn_len_override=None,
        g_it=None,
    ):
        """
        Compute camera-aware self-attention with memory.

        Args:
            u_c:    (B, L, 128)  - token features in bottleneck space
            memory: (B, M, 128)  - memory features
            theta:  (B, L, 16)   - shared theta from CamRoPE
            disable_memory_kv: bool - ablation flag; when True, K/V come from
                               tokens only (memory is excluded), effectively
                               M=0. Theta/CamRoPE still applied normally.

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

        # Ablation (1): Memory-off — K/V from tokens only, no memory concat
        if disable_memory_kv:
            tokens_and_memory = u_c  # (B, L, dim)  — memory excluded
            M_eff = 0
        else:
            tokens_and_memory = torch.cat([u_c, memory], dim=1)  # (B, L+M, dim)
            M_eff = M

        kv = F.linear(
            tokens_and_memory,
            self.kv_proj.weight,
            torch.cat([self.k_bias, self.v_bias]),
        )  # (B, L+M_eff, 2*num_heads*head_dim)
        kv = kv.view(
            B, L + M_eff, 2, self.num_heads, self.head_dim
        )  # (B, L+M_eff, 2, num_heads, head_dim)
        k, v = kv.unbind(dim=2)  # Each: (B, L+M_eff, num_heads, head_dim)

        # Warm-start gain scheduling
        warm_steps = int(self.warm_start_steps.item())
        use_warm = warm_steps > 0 and g_it is not None and g_it < warm_steps
        theta_gain = self.warm_theta_gain_value if use_warm else self.theta_gain_param
        temp_gain = self.warm_temp_gain_value if use_warm else self.temp_gain_param
        if self.warm_unfreeze and not use_warm:
            if not self.theta_gain_param.requires_grad:
                self.theta_gain_param.requires_grad_(True)
            if not self.temp_gain_param.requires_grad:
                self.temp_gain_param.requires_grad_(True)

        # Apply CamRoPE to Q and K_tokens (not memory K)
        theta_eff = theta_gain * theta
        q_rot, k_rot = CamRoPE.apply_rotary(q, k, theta_eff, apply_to_k=True)

        # Compute attention: Q @ K^T
        q_rot = q_rot.transpose(1, 2)  # (B, num_heads, L, head_dim)
        k_rot = k_rot.transpose(1, 2)  # (B, num_heads, L+M_eff, head_dim)
        v = v.transpose(1, 2)  # (B, num_heads, L+M_eff, head_dim)

        # temp_gain sharpens logits before softmax to combat attention collapse
        attn_scores = (
            torch.matmul(q_rot, k_rot.transpose(-2, -1)) * self.scale * temp_gain
        )  # (B, num_heads, L, L+M_eff)
        attn_weights = F.softmax(attn_scores, dim=-1)

        attn_debug_metrics = None

        # Optional lightweight debug for attention collapse detection
        if attn_debug:
            if not hasattr(self, "_attn_debug_calls"):
                self._attn_debug_calls = 0
            if self._attn_debug_calls < 5:
                with torch.no_grad():
                    N = (
                        attn_len_override
                        if attn_len_override is not None
                        else k_rot.shape[-2]
                    )
                    logN = math.log(max(N, 1))
                    invN = 1.0 / max(N, 1)

                    scores_std = attn_scores.std().item()
                    weights_entropy = (
                        (-attn_weights * attn_weights.clamp_min(1e-12).log())
                        .sum(dim=-1)
                        .mean()
                        .item()
                    )
                    weights_max_mean = attn_weights.max(dim=-1).values.mean().item()

                    entropy_gap = weights_entropy - logN
                    max_gap = weights_max_mean - invN

                    attn_debug_metrics = {
                        "theta_gain": theta_gain.item(),
                        "temp_gain": temp_gain.item(),
                        "entropy_gap": entropy_gap,
                        "max_gap": max_gap,
                        "weights_entropy": weights_entropy,
                        "weights_max_mean": weights_max_mean,
                    }

                    # if _dist_dbg.is_master():
                    #     print(
                    #         f"[sa_cam debug] call={self._attn_debug_calls} "
                    #         f"theta_gain={theta_gain.item():.4f} "
                    #         f"temp_gain={temp_gain.item():.4f} "
                    #         f"N_active={N} "
                    #         f"scores_std={scores_std:.4f} "
                    #         f"weights_entropy={weights_entropy:.4f} "
                    #         f"entropy_gap={entropy_gap:.4f} "
                    #         f"weights_max_mean={weights_max_mean:.4f} "
                    #         f"max_gap={max_gap:.6f}"
                    #     )
                self._attn_debug_calls += 1

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

        if attn_debug_metrics is not None:
            return out, attn_debug_metrics
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

    def __init__(
        self,
        embed_dim=1920,
        adapter_dim=128,
        num_heads=4,
        head_dim=32,
        theta_gain_value=12.0,
        temp_gain_value=12.0,
        warm_start_steps: int = 0,
        warm_theta_gain_value: float = 12.0,
        warm_temp_gain_value: float = 12.0,
        warm_unfreeze: bool = True,
    ):
        super().__init__()

        self.embed_dim = embed_dim
        self.adapter_dim = adapter_dim

        # NORMAL initialization (NOT zero) - enables gradient flow
        self.p_down = nn.Linear(embed_dim, adapter_dim)
        self.p_up = nn.Linear(adapter_dim, embed_dim)

        # Camera-aware self-attention
        self.sa_cam = CameraAwareAttention(
            dim=adapter_dim,
            num_heads=num_heads,
            head_dim=head_dim,
            theta_gain_value=theta_gain_value,
            temp_gain_value=temp_gain_value,
            warm_start_steps=warm_start_steps,
            warm_theta_gain_value=warm_theta_gain_value,
            warm_temp_gain_value=warm_temp_gain_value,
            warm_unfreeze=warm_unfreeze,
        )

        # Normalization layers for residualized bottleneck (active path)
        self.norm_pre = nn.LayerNorm(adapter_dim)
        self.norm_post = nn.LayerNorm(adapter_dim)

        # print(
        #     f"RayAdapter initialized: embed_dim={embed_dim}, adapter_dim={adapter_dim}"
        # )
        # print(f"  P_down/P_up use NORMAL initialization (NOT zero)")

    def forward(
        self,
        u,
        memory,
        theta,
        disable_memory_kv=False,
        attn_debug=False,
        scale_mask=None,
        g_it=None,
    ):
        """
        Process pre-normalized input with camera-aware attention.

        Args:
            u:      (B, L, 1920) - pre-norm input from block
            memory: (B, M, 128)  - shared model-level memory
            theta:  (B, L, 16)   - theta from CamRoPE
            disable_memory_kv: bool - ablation flag passed to SA_cam

        Returns:
            delta_x_ray: (B, L, 1920) - ray adaptation residual (gating applied externally)

        Internal flow:
        1. u_c        = P_down(u)                          → (B, L, 128)
        2. attn_out   = SA_cam(u_c, memory, theta, ...)    → (B, L, 128)
        3. delta_x_ray = P_up(attn_out)                    → (B, L, 1920)
        """
        B, L, C = u.shape
        assert u.shape == (B, L, self.embed_dim), (
            f"Expected u shape ({B}, {L}, {self.embed_dim}), got {u.shape}"
        )

        # Project down to bottleneck
        u_c = self.p_down(u)  # (B, L, 1920) -> (B, L, 128)

        # Option B: restrict attention to active tokens only
        if scale_mask is not None:
            active_mask = scale_mask.squeeze(-1).squeeze(0).bool()  # (L,)
        else:
            active_mask = None

        attn_metrics = None

        if active_mask is not None and active_mask.any():
            u_c_active = u_c[:, active_mask, :]  # (B, L_active, 128)
            theta_active = theta[:, active_mask, :]  # (B, L_active, 16)

            # Pre-attn normalization
            u_c_active_norm = self.norm_pre(u_c_active)

            # Disable memory for this experiment to avoid confounds
            B = u.shape[0]
            memory_empty = memory.new_zeros(B, 0, self.adapter_dim)

            attn_out_active = self.sa_cam(
                u_c_active_norm,
                memory_empty,
                theta_active,
                disable_memory_kv=True,
                attn_debug=attn_debug,
                attn_len_override=u_c_active.shape[1],
                g_it=g_it,
            )

            attn_metrics = None
            if attn_debug and isinstance(attn_out_active, tuple):
                attn_out_active, attn_metrics = attn_out_active

            # Residual + post normalization in bottleneck
            h_active = u_c_active + attn_out_active
            h_active = self.norm_post(h_active)

            # Project and scatter back to full length (zeros elsewhere)
            delta_active = self.p_up(h_active)  # (B, L_active, 1920)
            delta_x_ray = torch.zeros(
                u.shape[0],
                u.shape[1],
                self.embed_dim,
                device=u.device,
                dtype=delta_active.dtype,
            )
            delta_x_ray[:, active_mask, :] = delta_active
        elif active_mask is not None and not active_mask.any():
            # Mask provided but no active tokens
            delta_x_ray = torch.zeros(
                u.shape[0], u.shape[1], self.embed_dim, device=u.device, dtype=u.dtype
            )
        else:
            # No mask provided — fallback to full-sequence attention (previous behavior)
            attn_out = self.sa_cam(
                u_c,
                memory,
                theta,
                disable_memory_kv=disable_memory_kv,
                attn_debug=attn_debug,
                g_it=g_it,
            )  # (B, L, 128)
            attn_metrics = None
            if attn_debug and isinstance(attn_out, tuple):
                attn_out, attn_metrics = attn_out
            delta_x_ray = self.p_up(attn_out)  # (B, L, 1920)

        if attn_debug and attn_metrics is not None:
            return delta_x_ray, attn_metrics
        return delta_x_ray

    def set_fixed_gains(self, theta_gain: float, temp_gain: float):
        """Utility to override fixed gains for contrast ablations.

        These gains are buffers (not learnable). Call before running inference/training
        to test alternative fixed contrasts (e.g., 12.0 / 12.0).
        """
        with torch.no_grad():
            self.theta_gain.fill_(theta_gain)
            self.temp_gain.fill_(temp_gain)

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
