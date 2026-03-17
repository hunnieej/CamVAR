"""
Discrete face-ID embedding for cubemap conditioning.

Provides a learnable embedding for each of the 6 canonical cubemap faces,
indexed by their canonical integer IDs:
    0 = front, 1 = right, 2 = back, 3 = left, 4 = up, 5 = down

This is the primary conditioning signal for the cubemap training path.
It is NOT a continuous trajectory — face IDs are discrete semantic labels.

Usage in trainer:
    embed = FaceIdEmbedding(embed_dim=3)   # embed_dim matches cam_dir shape
    face_id_tensor = torch.tensor([0, 2, 3, 1], ...)  # e.g. set1
    cam_dir_override = embed(face_id_tensor)  # [B, 3]
    # pass as cam_dir to trainer.train_step(...)
"""

import torch
import torch.nn as nn

from utils.cubemap_groups import CANONICAL_FACES, FACE_TO_IDX

NUM_CANONICAL_FACES: int = len(CANONICAL_FACES)  # 6


class FaceIdEmbedding(nn.Module):
    """
    Learnable discrete embedding for the 6 canonical cubemap faces.

    Each face gets a separate learnable vector of dimension `embed_dim`.
    The embedding is initialized with a small normal distribution so that
    the model starts without a strong directional prior.

    Args:
        embed_dim: Dimension of the embedding vector.  Should match the
                   dimensionality expected by cam_dir in the ray adapter
                   (typically 3 for a unit direction vector, but can be
                   larger for a learned feature space).
        init_std: Standard deviation for weight initialization (default 0.02).
    """

    def __init__(self, embed_dim: int = 3, init_std: float = 0.02):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_faces = NUM_CANONICAL_FACES
        # One embedding per canonical face
        self.embedding = nn.Embedding(self.num_faces, embed_dim)
        nn.init.normal_(self.embedding.weight, mean=0.0, std=init_std)

    def forward(self, face_ids: torch.LongTensor) -> torch.FloatTensor:
        """
        Map face IDs to embedding vectors.

        Args:
            face_ids: LongTensor of any shape containing values in [0, 5].

        Returns:
            FloatTensor of shape (*face_ids.shape, embed_dim).
        """
        return self.embedding(face_ids)

    def face_name_to_embedding(self, face_name: str) -> torch.FloatTensor:
        """Convenience: look up embedding by face name string."""
        if face_name not in FACE_TO_IDX:
            raise ValueError(
                f"Unknown face name '{face_name}'. "
                f"Valid names: {list(FACE_TO_IDX.keys())}"
            )
        idx = torch.tensor([FACE_TO_IDX[face_name]], dtype=torch.long)
        return self.forward(idx).squeeze(0)

    def extra_repr(self) -> str:
        return (
            f"num_faces={self.num_faces}, embed_dim={self.embed_dim}, "
            f"faces={CANONICAL_FACES}"
        )
