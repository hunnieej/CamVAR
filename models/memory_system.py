"""
Memory System: Model-level memory management.

Implements MemoryUpdater for scene-level memory bank that:
- Is shared across all transformer blocks (read-only in blocks)
- Updated once per forward pass OUTSIDE the transformer loop
- Phase 1: Reset to zeros each forward
- Phase 2+: Can persist across views
"""

import torch
import torch.nn as nn


class MemoryUpdater(nn.Module):
    """
    Model-level memory updater (NOT per-block).

    Memory is a scene-level state: there is ONE memory tensor m per sample/view-step,
    shared across all blocks during forward. Memory update happens once per view-step
    outside the blocks.

    Update rule: memory_out = memory_in + alpha * Δm
    where alpha is a learned scalar initialized to 0 (for safety).
    """

    def __init__(self, adapter_dim=128, mem_size=32):
        super().__init__()

        self.adapter_dim = adapter_dim
        self.mem_size = mem_size

        # Projection from pooled features to memory update
        self.update_proj = nn.Linear(adapter_dim, mem_size * adapter_dim)

        # Learned alpha initialized to 0 for "do no harm" start
        self.alpha = nn.Parameter(torch.zeros(1))

        print(
            f"MemoryUpdater initialized: adapter_dim={adapter_dim}, mem_size={mem_size}, alpha=0"
        )

    def reset_memory(self, batch_size, device, dtype=torch.float32):
        """
        Reset memory to zeros (Phase 1 behavior).

        Args:
            batch_size: int
            device: torch.device
            dtype: torch.dtype

        Returns:
            memory: (B, M, adapter_dim) - zero-initialized memory
        """
        return torch.zeros(
            batch_size, self.mem_size, self.adapter_dim, device=device, dtype=dtype
        )

    def update_memory(self, u_c_pooled, memory):
        """
        Update memory once per forward pass.

        Args:
            u_c_pooled: (B, adapter_dim) - pooled representation from final layer
            memory: (B, M, adapter_dim) - current memory state

        Returns:
            memory_updated: (B, M, adapter_dim) - updated memory
        """
        B = u_c_pooled.shape[0]
        assert u_c_pooled.shape == (B, self.adapter_dim), (
            f"Expected u_c_pooled shape ({B}, {self.adapter_dim}), got {u_c_pooled.shape}"
        )
        assert memory.shape == (B, self.mem_size, self.adapter_dim), (
            f"Expected memory shape ({B}, {self.mem_size}, {self.adapter_dim}), got {memory.shape}"
        )

        # Compute memory update
        delta_m = self.update_proj(u_c_pooled)  # (B, adapter_dim) -> (B, M*adapter_dim)
        delta_m = delta_m.view(
            B, self.mem_size, self.adapter_dim
        )  # (B, M, adapter_dim)

        # Apply learned alpha
        memory_updated = memory + self.alpha * delta_m

        return memory_updated

    def verify_alpha_init(self):
        """Verify that alpha is initialized to 0."""
        assert torch.allclose(self.alpha, torch.zeros_like(self.alpha)), (
            f"Alpha should be initialized to 0, got {self.alpha.item()}"
        )
        print("✓ Alpha initialization verified: alpha=0")


if __name__ == "__main__":
    print("Testing MemoryUpdater...")

    # Create updater
    mem_updater = MemoryUpdater(adapter_dim=128, mem_size=32)
    mem_updater.verify_alpha_init()

    # Test reset
    batch_size = 2
    device = torch.device("cpu")
    memory = mem_updater.reset_memory(batch_size, device)
    print(f"  Reset memory shape: {memory.shape}")
    assert memory.shape == (2, 32, 128)
    assert torch.all(memory == 0)

    # Test update
    u_c_pooled = torch.randn(2, 128)
    memory_updated = mem_updater.update_memory(u_c_pooled, memory)
    print(f"  Updated memory shape: {memory_updated.shape}")
    assert memory_updated.shape == (2, 32, 128)

    # At initialization, with alpha=0, memory should not change
    print(
        f"  Memory change (should be near 0 with alpha=0): {torch.abs(memory_updated - memory).max().item():.6f}"
    )

    # Simulate training: manually set alpha to test update
    with torch.no_grad():
        mem_updater.alpha.fill_(0.1)

    memory_updated2 = mem_updater.update_memory(u_c_pooled, memory)
    print(
        f"  Memory change with alpha=0.1: {torch.abs(memory_updated2 - memory).max().item():.6f}"
    )
    assert not torch.allclose(memory_updated2, memory), (
        "Memory should change with non-zero alpha"
    )

    print("\nMemoryUpdater tests passed!")
