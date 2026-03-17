import numpy as np
import torch
from torch.utils.data.sampler import Sampler
from typing import Optional


class EvalDistributedSampler(Sampler):
    def __init__(self, dataset, num_replicas, rank):
        seps = np.linspace(0, len(dataset), num_replicas + 1, dtype=int)
        beg, end = seps[:-1], seps[1:]
        beg, end = beg[rank], end[rank]
        self.indices = tuple(range(beg, end))

    def __iter__(self):
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)


class InfiniteBatchSampler(Sampler):
    def __init__(
        self,
        dataset_len,
        batch_size,
        seed_for_all_rank=0,
        fill_last=False,
        shuffle=True,
        drop_last=False,
        start_ep=0,
        start_it=0,
    ):
        self.dataset_len = dataset_len
        self.batch_size = batch_size
        self.iters_per_ep = (
            dataset_len // batch_size
            if drop_last
            else (dataset_len + batch_size - 1) // batch_size
        )
        self.max_p = self.iters_per_ep * batch_size
        self.fill_last = fill_last
        self.shuffle = shuffle
        self.epoch = start_ep
        self.same_seed_for_all_ranks = seed_for_all_rank
        self.indices = self.gener_indices()
        self.start_ep, self.start_it = start_ep, start_it

    def gener_indices(self):
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.same_seed_for_all_ranks)
            indices = torch.randperm(self.dataset_len, generator=g).numpy()
        else:
            indices = torch.arange(self.dataset_len).numpy()

        tails = self.batch_size - (self.dataset_len % self.batch_size)
        if tails != self.batch_size and self.fill_last:
            tails = indices[:tails]
            np.random.shuffle(indices)
            indices = np.concatenate((indices, tails))

        # built-in list/tuple is faster than np.ndarray (when collating the data via a for-loop)
        # noinspection PyTypeChecker
        return tuple(indices.tolist())

    def __iter__(self):
        self.epoch = self.start_ep
        while True:
            self.epoch += 1
            p = (self.start_it * self.batch_size) if self.epoch == self.start_ep else 0
            while p < self.max_p:
                q = p + self.batch_size
                yield self.indices[p:q]
                p = q
            if self.shuffle:
                self.indices = self.gener_indices()

    def __len__(self):
        return self.iters_per_ep


class SceneViewBatchSampler(Sampler):
    """
    Rank-local sampler to build batches with a fixed number of scenes and views per scene.
    Intended for ERP training where each panorama has `views_per_pano` distinct views.
    """

    def __init__(
        self,
        num_panos: int,
        views_per_pano: int,
        batch_size: int,
        scenes_per_batch: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        views_per_scene: Optional[int] = None,
    ):
        # Fixed to single scene per batch for ERP grouping
        scenes_per_batch = 1
        assert batch_size > 0, "batch_size must be positive"

        # If views_per_scene is provided, it defines the trajectory length per scene;
        # otherwise fall back to batch_size (legacy behavior).
        self.views_per_scene = views_per_scene or batch_size

        assert views_per_pano >= self.views_per_scene, (
            "views_per_scene must not exceed views_per_pano"
        )
        assert batch_size <= self.views_per_scene, (
            "batch_size must not exceed views_per_scene"
        )

        self.num_panos = num_panos
        self.views_per_pano = views_per_pano
        self.scenes_per_batch = scenes_per_batch
        self.shuffle = shuffle
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

    def __iter__(self):
        dbg_count = 0
        while True:
            # pick one scene
            if self.shuffle:
                pano_id = torch.randint(
                    low=0, high=self.num_panos, size=(1,), generator=self.rng
                ).item()
            else:
                pano_id = 0

            # Return views in canonical order (0, 1, ..., views_per_scene-1) so that
            # make_view_groups() receives a deterministic, position-stable list.
            # Previously used torch.randperm which broke group-position semantics.
            view_ids = list(range(self.views_per_scene))
            batch_indices = [pano_id * self.views_per_pano + vid for vid in view_ids]

            if dbg_count < 5:
                print(
                    f"[SceneViewBatchSampler] scene={pano_id} views={view_ids} batch_indices={batch_indices}"
                )
                dbg_count += 1

            yield batch_indices

    def __len__(self):
        return max(
            1, (self.num_panos + self.scenes_per_batch - 1) // self.scenes_per_batch
        )


class DistSceneViewBatchSampler(Sampler):
    """
    Distributed sampler to form a global batch of 1 scene with distinct views across ranks.
    Assumes local batch size is provided; global views per batch = local_batch * world_size.
    """

    def __init__(
        self,
        num_panos: int,
        views_per_pano: int,
        batch_size: int,
        scenes_per_global_batch: int = 1,
        shuffle: bool = True,
        seed: int = 0,
        views_per_scene: int = None,
    ):
        import torch.distributed as dist

        self.world_size = (
            dist.get_world_size()
            if dist.is_available() and dist.is_initialized()
            else 1
        )
        self.rank = (
            dist.get_rank() if dist.is_available() and dist.is_initialized() else 0
        )
        self.num_panos = num_panos
        self.views_per_pano = views_per_pano
        self.scenes_per_batch = scenes_per_global_batch
        self.local_batch = batch_size
        self.views_per_scene = views_per_scene or (batch_size * self.world_size)
        assert batch_size > 0
        assert self.scenes_per_batch == 1, "only single-scene global batches supported"
        self.shuffle = shuffle
        self.rng = torch.Generator()
        self.rng.manual_seed(seed)

    def __iter__(self):
        import torch.distributed as dist
        import math

        while True:
            obj = [0, [0 for _ in range(self.views_per_scene)]]
            if self.rank == 0:
                pano_id = int(
                    torch.randint(
                        low=0, high=self.num_panos, size=(1,), generator=self.rng
                    ).item()
                )
                view_ids = torch.randperm(self.views_per_pano, generator=self.rng)[
                    : self.views_per_scene
                ].tolist()
                obj = [pano_id, view_ids]

            if self.world_size > 1 and dist.is_available() and dist.is_initialized():
                dist.broadcast_object_list(obj, src=0)

            pano_id, view_ids = obj
            if view_ids is None or len(view_ids) < self.views_per_scene:
                raise RuntimeError("Invalid view_ids in DistSceneViewBatchSampler")
            # Evenly split view_ids across ranks (may differ by at most 1)
            splits = torch.tensor_split(torch.tensor(view_ids), self.world_size, dim=0)
            view_ids_rank = splits[self.rank].tolist()

            # Debug sanity (rank 0 only, first few iterations)
            if self.rank == 0:
                setattr(self, "_dbg_count", getattr(self, "_dbg_count", 0) + 1)
                if getattr(self, "_dbg_count", 0) <= 5:
                    print(
                        f"[DistSceneViewBatchSampler] scene={pano_id} world={self.world_size} view_ids={view_ids} per_rank={[s.tolist() for s in splits]}",
                        flush=True,
                    )

            batch_indices = [
                pano_id * self.views_per_pano + vid for vid in view_ids_rank
            ]
            yield batch_indices

    def __len__(self):
        return max(
            1, (self.num_panos + self.scenes_per_batch - 1) // self.scenes_per_batch
        )


class DistInfiniteBatchSampler(InfiniteBatchSampler):
    def __init__(
        self,
        world_size,
        rank,
        dataset_len,
        glb_batch_size,
        same_seed_for_all_ranks=0,
        repeated_aug=0,
        fill_last=False,
        shuffle=True,
        start_ep=0,
        start_it=0,
    ):
        assert glb_batch_size % world_size == 0
        self.world_size, self.rank = world_size, rank
        self.dataset_len = dataset_len
        self.glb_batch_size = glb_batch_size
        self.batch_size = glb_batch_size // world_size

        self.iters_per_ep = (dataset_len + glb_batch_size - 1) // glb_batch_size
        self.fill_last = fill_last
        self.shuffle = shuffle
        self.repeated_aug = repeated_aug
        self.epoch = start_ep
        self.same_seed_for_all_ranks = same_seed_for_all_ranks
        self.indices = self.gener_indices()
        self.start_ep, self.start_it = start_ep, start_it

    def gener_indices(self):
        global_max_p = (
            self.iters_per_ep * self.glb_batch_size
        )  # global_max_p % world_size must be 0 cuz glb_batch_size % world_size == 0
        # print(f'global_max_p = iters_per_ep({self.iters_per_ep}) * glb_batch_size({self.glb_batch_size}) = {global_max_p}')
        if self.shuffle:
            g = torch.Generator()
            g.manual_seed(self.epoch + self.same_seed_for_all_ranks)
            global_indices = torch.randperm(self.dataset_len, generator=g)
            if self.repeated_aug > 1:
                global_indices = global_indices[
                    : (self.dataset_len + self.repeated_aug - 1) // self.repeated_aug
                ].repeat_interleave(self.repeated_aug, dim=0)[:global_max_p]
        else:
            global_indices = torch.arange(self.dataset_len)
        filling = global_max_p - global_indices.shape[0]
        if filling > 0 and self.fill_last:
            global_indices = torch.cat((global_indices, global_indices[:filling]))
        # global_indices = tuple(global_indices.numpy().tolist())

        seps = torch.linspace(
            0, global_indices.shape[0], self.world_size + 1, dtype=torch.int
        )
        local_indices = global_indices[
            seps[self.rank].item() : seps[self.rank + 1].item()
        ].tolist()
        self.max_p = len(local_indices)
        return tuple(local_indices)
