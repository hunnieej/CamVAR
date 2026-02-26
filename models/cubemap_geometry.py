from dataclasses import dataclass
from typing import Dict, List, Tuple, Union

import torch


FACE_ORDER = ("F", "R", "B", "L", "U", "D")


def get_face_frames() -> torch.Tensor:
    n = torch.tensor(
        [
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, -1.0, 0.0],
        ]
    )
    a = torch.tensor(
        [
            [1.0, 0.0, 0.0],
            [0.0, 0.0, -1.0],
            [-1.0, 0.0, 0.0],
            [0.0, 0.0, 1.0],
            [1.0, 0.0, 0.0],
            [1.0, 0.0, 0.0],
        ]
    )
    b = torch.tensor(
        [
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, -1.0],
            [0.0, 0.0, 1.0],
        ]
    )
    return torch.stack([n, a, b], dim=1)


def token_centers(h_s: int, w_s: int) -> Tuple[torch.Tensor, torch.Tensor]:
    ii = torch.arange(h_s)
    jj = torch.arange(w_s)
    ii, jj = torch.meshgrid(ii, jj, indexing="ij")
    u = 2.0 * (jj + 0.5) / w_s - 1.0
    v = 1.0 - 2.0 * (ii + 0.5) / h_s
    return u, v


def normalize(v: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return v / (v.norm(dim=-1, keepdim=True) + eps)


def build_ray(
    n: torch.Tensor,
    a: torch.Tensor,
    b: torch.Tensor,
    u: Union[torch.Tensor, float],
    v: Union[torch.Tensor, float],
) -> torch.Tensor:
    u = torch.as_tensor(u, device=n.device, dtype=n.dtype)
    v = torch.as_tensor(v, device=n.device, dtype=n.dtype)
    d = n + u[..., None] * a + v[..., None] * b
    return normalize(d)


def build_ray_frame(d: torch.Tensor, a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    e_z = d
    proj = (b * d).sum(dim=-1, keepdim=True) * d
    e_y = b - proj
    degenerate = e_y.norm(dim=-1, keepdim=True) < 1e-6
    if degenerate.any():
        e_y = torch.where(degenerate, a.expand_as(e_y), e_y)
    e_y = normalize(e_y)
    e_x = torch.cross(e_y, e_z, dim=-1)
    e_x = normalize(e_x)
    r_ray = torch.stack([e_x, e_y, e_z], dim=-1)
    return r_ray.transpose(-2, -1)


def project_to_face(d: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    x, y, z = d.unbind(dim=-1)
    ax, ay, az = x.abs(), y.abs(), z.abs()
    use_x = (ax >= ay) & (ax >= az)
    use_y = (ay > ax) & (ay >= az)
    use_z = ~(use_x | use_y)

    face = torch.zeros_like(x, dtype=torch.long)
    u = torch.zeros_like(x)
    v = torch.zeros_like(x)

    face_x_pos = use_x & (x > 0)
    face_x_neg = use_x & (x < 0)
    face_y_pos = use_y & (y > 0)
    face_y_neg = use_y & (y < 0)
    face_z_pos = use_z & (z > 0)
    face_z_neg = use_z & (z < 0)

    face[face_z_pos] = 0
    u[face_z_pos] = x[face_z_pos] / az[face_z_pos]
    v[face_z_pos] = y[face_z_pos] / az[face_z_pos]

    face[face_z_neg] = 2
    u[face_z_neg] = -x[face_z_neg] / az[face_z_neg]
    v[face_z_neg] = y[face_z_neg] / az[face_z_neg]

    face[face_x_pos] = 1
    u[face_x_pos] = -z[face_x_pos] / ax[face_x_pos]
    v[face_x_pos] = y[face_x_pos] / ax[face_x_pos]

    face[face_x_neg] = 3
    u[face_x_neg] = z[face_x_neg] / ax[face_x_neg]
    v[face_x_neg] = y[face_x_neg] / ax[face_x_neg]

    face[face_y_pos] = 4
    u[face_y_pos] = x[face_y_pos] / ay[face_y_pos]
    v[face_y_pos] = -z[face_y_pos] / ay[face_y_pos]

    face[face_y_neg] = 5
    u[face_y_neg] = x[face_y_neg] / ay[face_y_neg]
    v[face_y_neg] = z[face_y_neg] / ay[face_y_neg]

    return face, u, v


def uv_to_ij(
    u: torch.Tensor, v: torch.Tensor, h_s: int, w_s: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    j = torch.floor((u + 1.0) * 0.5 * w_s).long()
    i = torch.floor((1.0 - v) * 0.5 * h_s).long()
    j = j.clamp(min=0, max=w_s - 1)
    i = i.clamp(min=0, max=h_s - 1)
    return i, j


@dataclass
class CubemapPrecompute:
    boundary_face_ids: torch.Tensor
    boundary_token_indices: torch.Tensor
    boundary_local_indices: torch.Tensor
    boundary_stage_ids: torch.Tensor
    neighbor_face_ids: torch.Tensor
    neighbor_token_indices: torch.Tensor
    neighbor_local_indices: torch.Tensor
    neighbor_stage_ids: torch.Tensor
    neighbor_mask: torch.Tensor
    d_ray: torch.Tensor
    boundary_face_ids_last: torch.Tensor
    boundary_local_indices_last: torch.Tensor
    neighbor_face_ids_last: torch.Tensor
    neighbor_local_indices_last: torch.Tensor
    neighbor_mask_last: torch.Tensor
    boundary_count_per_stage: List[int]
    face_order: Tuple[str, ...] = FACE_ORDER


class CubemapGeometry:
    def __init__(
        self,
        patch_nums: Tuple[int, ...],
        begin_ends: List[Tuple[int, int]],
        strip_w: int = 2,
        k: int = 2,
    ):
        self.patch_nums = patch_nums
        self.begin_ends = begin_ends
        self.strip_w = strip_w
        self.k = k
        self.frames = get_face_frames()

    def precompute(self) -> CubemapPrecompute:
        boundary_face_ids = []
        boundary_token_indices = []
        boundary_local_indices = []
        boundary_stage_ids = []
        neighbor_face_ids = []
        neighbor_token_indices = []
        neighbor_local_indices = []
        neighbor_stage_ids = []
        boundary_count_per_stage = []

        boundary_face_ids_last = []
        boundary_local_indices_last = []
        neighbor_face_ids_last = []
        neighbor_local_indices_last = []

        max_tokens = max(pn * pn for pn in self.patch_nums)
        d_ray_all = torch.zeros(
            (len(self.patch_nums), len(FACE_ORDER), max_tokens, 3, 3)
        )

        for stage_id, pn in enumerate(self.patch_nums):
            h_s = w_s = pn
            u_grid, v_grid = token_centers(h_s, w_s)
            token_count = h_s * w_s
            seq_offset = self.begin_ends[stage_id][0]
            stage_boundary_count = 0

            for face_id in range(len(FACE_ORDER)):
                n, a, b = self.frames[face_id]
                d = build_ray(n, a, b, u_grid, v_grid)
                d_ray = build_ray_frame(d, a, b)
                d_ray_all[stage_id, face_id, :token_count] = d_ray.view(
                    token_count, 3, 3
                )

                for i in range(h_s):
                    for j in range(w_s):
                        is_top = i < self.strip_w
                        is_bottom = i >= h_s - self.strip_w
                        is_left = j < self.strip_w
                        is_right = j >= w_s - self.strip_w
                        if not (is_top or is_bottom or is_left or is_right):
                            continue

                        local_idx = i * w_s + j
                        token_idx = seq_offset + local_idx

                        u = u_grid[i, j]
                        v = v_grid[i, j]
                        d_main = build_ray(n, a, b, u, v)
                        neighbor_set = self._neighbors_from_ray(
                            d_main, h_s, w_s, is_left, is_right, is_top, is_bottom
                        )

                        if (is_left or is_right) and (is_top or is_bottom):
                            eps = 1e-4
                            u_edge = -1.0 + eps if is_left else 1.0 - eps
                            v_edge = 1.0 - eps if is_top else -1.0 + eps
                            d_h = build_ray(
                                n, a, b, torch.tensor(u_edge), torch.tensor(0.0)
                            )
                            d_v = build_ray(
                                n, a, b, torch.tensor(0.0), torch.tensor(v_edge)
                            )
                            neighbor_set |= self._neighbors_from_ray(
                                d_h, h_s, w_s, True, True, False, False
                            )
                            neighbor_set |= self._neighbors_from_ray(
                                d_v, h_s, w_s, False, False, True, True
                            )

                        if not neighbor_set:
                            continue

                        neighbor_face_ids.append([item[0] for item in neighbor_set])
                        neighbor_local_indices.append(
                            [item[1] for item in neighbor_set]
                        )
                        neighbor_token_indices.append(
                            [seq_offset + item[1] for item in neighbor_set]
                        )
                        neighbor_stage_ids.append([stage_id for _ in neighbor_set])
                        boundary_face_ids.append(face_id)
                        boundary_local_indices.append(local_idx)
                        boundary_token_indices.append(token_idx)
                        boundary_stage_ids.append(stage_id)
                        stage_boundary_count += 1

                        if stage_id == len(self.patch_nums) - 1:
                            neighbor_face_ids_last.append(
                                [item[0] for item in neighbor_set]
                            )
                            neighbor_local_indices_last.append(
                                [item[1] for item in neighbor_set]
                            )
                            boundary_face_ids_last.append(face_id)
                            boundary_local_indices_last.append(local_idx)

            boundary_count_per_stage.append(stage_boundary_count)

        max_neighbors = (
            max(len(n) for n in neighbor_face_ids) if neighbor_face_ids else 0
        )
        neighbor_face_ids = self._pad_list(neighbor_face_ids, max_neighbors, -1)
        neighbor_token_indices = self._pad_list(
            neighbor_token_indices, max_neighbors, -1
        )
        neighbor_local_indices = self._pad_list(
            neighbor_local_indices, max_neighbors, -1
        )
        neighbor_stage_ids = self._pad_list(neighbor_stage_ids, max_neighbors, -1)
        neighbor_mask = torch.tensor(neighbor_face_ids) != -1

        max_neighbors_last = (
            max(len(n) for n in neighbor_face_ids_last) if neighbor_face_ids_last else 0
        )
        neighbor_face_ids_last = self._pad_list(
            neighbor_face_ids_last, max_neighbors_last, -1
        )
        neighbor_local_indices_last = self._pad_list(
            neighbor_local_indices_last, max_neighbors_last, -1
        )
        neighbor_mask_last = torch.tensor(neighbor_face_ids_last) != -1

        return CubemapPrecompute(
            boundary_face_ids=torch.tensor(boundary_face_ids, dtype=torch.long),
            boundary_token_indices=torch.tensor(
                boundary_token_indices, dtype=torch.long
            ),
            boundary_local_indices=torch.tensor(
                boundary_local_indices, dtype=torch.long
            ),
            boundary_stage_ids=torch.tensor(boundary_stage_ids, dtype=torch.long),
            neighbor_face_ids=torch.tensor(neighbor_face_ids, dtype=torch.long),
            neighbor_token_indices=torch.tensor(
                neighbor_token_indices, dtype=torch.long
            ),
            neighbor_local_indices=torch.tensor(
                neighbor_local_indices, dtype=torch.long
            ),
            neighbor_stage_ids=torch.tensor(neighbor_stage_ids, dtype=torch.long),
            neighbor_mask=neighbor_mask,
            d_ray=d_ray_all,
            boundary_face_ids_last=torch.tensor(
                boundary_face_ids_last, dtype=torch.long
            ),
            boundary_local_indices_last=torch.tensor(
                boundary_local_indices_last, dtype=torch.long
            ),
            neighbor_face_ids_last=torch.tensor(
                neighbor_face_ids_last, dtype=torch.long
            ),
            neighbor_local_indices_last=torch.tensor(
                neighbor_local_indices_last, dtype=torch.long
            ),
            neighbor_mask_last=neighbor_mask_last,
            boundary_count_per_stage=boundary_count_per_stage,
        )

    def _neighbors_from_ray(
        self,
        d: torch.Tensor,
        h_s: int,
        w_s: int,
        allow_horizontal: bool,
        allow_vertical: bool,
        allow_top: bool,
        allow_bottom: bool,
    ) -> set:
        face, u_p, v_p = project_to_face(d.unsqueeze(0))
        face = face.item()
        u_p = u_p.item()
        v_p = v_p.item()
        i_p, j_p = uv_to_ij(torch.tensor(u_p), torch.tensor(v_p), h_s, w_s)
        i_p = i_p.item()
        j_p = j_p.item()

        neighbors = set()
        if allow_horizontal:
            for delta in range(-self.k, self.k + 1):
                ii = i_p + delta
                if 0 <= ii < h_s:
                    neighbors.add((face, ii * w_s + j_p))
        if allow_vertical:
            for delta in range(-self.k, self.k + 1):
                jj = j_p + delta
                if 0 <= jj < w_s:
                    neighbors.add((face, i_p * w_s + jj))

        if allow_top or allow_bottom:
            neighbors.add((face, i_p * w_s + j_p))
        return neighbors

    @staticmethod
    def _pad_list(
        items: List[List[int]], max_len: int, pad_value: int
    ) -> List[List[int]]:
        padded = []
        for item in items:
            item = item + [pad_value] * (max_len - len(item))
            padded.append(item)
        return padded

    def projection_consistency_check(self, sample_count: int = 256) -> Dict[str, float]:
        pn = self.patch_nums[-1]
        h_s = w_s = pn
        u_grid, v_grid = token_centers(h_s, w_s)
        stats: Dict[str, float] = {
            "u_out_of_range": 0.0,
            "v_out_of_range": 0.0,
            "ij_out_of_range": 0.0,
            "mapped_same_face": 0.0,
            "total": 0.0,
        }

        rng = torch.Generator().manual_seed(0)
        face_ids = torch.randint(0, len(FACE_ORDER), (sample_count,), generator=rng)
        ii = torch.randint(0, h_s, (sample_count,), generator=rng)
        jj = torch.randint(0, w_s, (sample_count,), generator=rng)

        for face_id, i, j in zip(face_ids.tolist(), ii.tolist(), jj.tolist()):
            is_boundary = (
                i < self.strip_w
                or i >= h_s - self.strip_w
                or j < self.strip_w
                or j >= w_s - self.strip_w
            )
            if not is_boundary:
                continue
            stats["total"] += 1.0

            n, a, b = self.frames[face_id]
            eps = 1e-4
            is_top = i < self.strip_w
            is_bottom = i >= h_s - self.strip_w
            is_left = j < self.strip_w
            is_right = j >= w_s - self.strip_w

            candidate_faces = []
            if is_left or is_right:
                u_edge = -1.0 - eps if is_left else 1.0 + eps
                d_h = build_ray(n, a, b, u_edge, 0.0)
                candidate_faces.append(d_h)
            if is_top or is_bottom:
                v_edge = 1.0 + eps if is_top else -1.0 - eps
                d_v = build_ray(n, a, b, 0.0, v_edge)
                candidate_faces.append(d_v)

            if not candidate_faces:
                continue

            mapped_same = True
            for d in candidate_faces:
                tgt_face, u_p, v_p = project_to_face(d.unsqueeze(0))
                tgt_face = tgt_face.item()
                u_p = u_p.item()
                v_p = v_p.item()
                if not (-1.0 <= u_p <= 1.0):
                    stats["u_out_of_range"] += 1.0
                if not (-1.0 <= v_p <= 1.0):
                    stats["v_out_of_range"] += 1.0
                i_p, j_p = uv_to_ij(torch.tensor(u_p), torch.tensor(v_p), h_s, w_s)
                if not (0 <= i_p.item() < h_s and 0 <= j_p.item() < w_s):
                    stats["ij_out_of_range"] += 1.0
                if tgt_face != face_id:
                    mapped_same = False
            if mapped_same:
                stats["mapped_same_face"] += 1.0

        for key in (
            "u_out_of_range",
            "v_out_of_range",
            "ij_out_of_range",
            "mapped_same_face",
        ):
            stats[key] = stats[key] / max(stats["total"], 1.0)
        return stats

    def orthonormality_check(
        self, precompute: CubemapPrecompute, sample_count: int = 256
    ) -> float:
        rng = torch.Generator().manual_seed(0)
        boundary_count = precompute.boundary_face_ids.shape[0]
        if boundary_count == 0:
            return 0.0
        idx = torch.randint(
            0, boundary_count, (min(sample_count, boundary_count),), generator=rng
        )
        face_ids = precompute.boundary_face_ids[idx]
        local_idx = precompute.boundary_local_indices[idx]
        stage_ids = precompute.boundary_stage_ids[idx]
        d_ray = precompute.d_ray[stage_ids, face_ids, local_idx]
        ident = torch.eye(3).to(d_ray)
        diff = torch.matmul(d_ray.transpose(-2, -1), d_ray) - ident
        return diff.abs().max().item()
