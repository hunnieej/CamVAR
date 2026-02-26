import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset


FACE_ORDER = ("F", "R", "B", "L", "U", "D")


def _load_image(path: Path) -> torch.Tensor:
    img = Image.open(path).convert("RGB")
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = np.transpose(arr, (2, 0, 1))
    return torch.from_numpy(arr)


class CubemapSceneDataset(Dataset):
    def __init__(self, root: str):
        self.root = Path(root)
        self.scene_dirs = sorted([p for p in self.root.iterdir() if p.is_dir()])
        if not self.scene_dirs:
            raise FileNotFoundError(f"No scenes found in {self.root}")

    def __len__(self) -> int:
        return len(self.scene_dirs)

    def __getitem__(self, idx: int) -> Dict[str, object]:
        scene_dir = self.scene_dirs[idx]
        faces_dir = scene_dir / "faces"
        prompt_path = scene_dir / "prompts.json"

        faces: List[torch.Tensor] = []
        for face_id in FACE_ORDER:
            img_path = faces_dir / f"{face_id}.png"
            faces.append(_load_image(img_path))
        faces = torch.stack(faces, dim=0)

        prompt = ""
        if prompt_path.exists():
            payload = json.loads(prompt_path.read_text(encoding="utf-8"))
            if isinstance(payload, dict):
                prompt = payload.get("scene", "")

        return {
            "faces": faces,
            "prompt": prompt,
            "scene_id": scene_dir.name,
        }
