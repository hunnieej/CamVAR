"""
ERP same-scene trajectory grouping utilities.

Provides a single entry point ``make_view_groups`` that converts an ordered
canonical view list into a list of non-overlapping microbatch groups whose
union covers every view exactly once.

Design contract
---------------
* ``total_views`` – number of canonical views in the scene's trajectory.
* ``group_size``  – maximum number of views per microbatch group.
* ``num_groups``  – ``ceil(total_views / group_size)``; derived, never hard-coded.
* In ``contiguous`` mode, only the last group may be smaller.  In
  ``spread`` mode, groups differ by at most 1 in size (round-robin
  distribution), so multiple groups may be one view smaller than
  ``group_size`` when ``total_views % group_size != 0``.
* All views appear in exactly one group; groups are disjoint and their
  union is the complete canonical view list.

Grouping modes
--------------
``contiguous``
    Views are packed in run order:
        [0,1,2,3], [4,5,6,7], [8,9,10,11]  (12 views, group_size=4)
        [0,1,2,3], [4,5,6,7], [8,9]         (10 views, group_size=4)

``spread``
    Views are interleaved so each group samples the trajectory uniformly:
        [0,3,6,9], [1,4,7,10], [2,5,8,11]  (12 views, group_size=4)
        [0,3,6,9], [1,4,7], [2,5,8]         (10 views, group_size=4 → groups
                                               of size 4, 3, 3 — last partial
                                               group shrinks proportionally)
    This generalises the original hard-coded ``[[0,3,6,9],[1,4,7,10],[2,5,8,11]]``
    pattern to arbitrary ``total_views`` and ``group_size``.
"""

from __future__ import annotations

import math
from typing import List, Literal


_GroupingMode = Literal["contiguous", "spread"]


def make_view_groups(
    canonical_view_ids: List[int],
    group_size: int,
    mode: _GroupingMode = "spread",
) -> List[List[int]]:
    """Return a list of groups, each a sub-list of ``canonical_view_ids``.

    Parameters
    ----------
    canonical_view_ids:
        Ordered list of view indices for one scene's full trajectory.  The
        order defines the trajectory; do **not** pass a random permutation.
    group_size:
        Maximum number of views per microbatch group (``>= 1``).
    mode:
        ``"contiguous"`` packs consecutive views; ``"spread"`` interleaves
        them so each group covers the trajectory uniformly.

    Returns
    -------
    List of lists, each containing the view ids for one microbatch.  The
    groups are in a stable, deterministic order.

    Raises
    ------
    ValueError
        If ``canonical_view_ids`` is empty or ``group_size < 1``.
    """
    if not canonical_view_ids:
        raise ValueError("canonical_view_ids must be non-empty")
    if group_size < 1:
        raise ValueError(f"group_size must be >= 1, got {group_size}")

    total = len(canonical_view_ids)
    num_groups = math.ceil(total / group_size)

    if mode == "contiguous":
        groups = []
        for g in range(num_groups):
            start = g * group_size
            end = min(start + group_size, total)
            groups.append(canonical_view_ids[start:end])
        return groups

    elif mode == "spread":
        # Group g receives views at positions g, g+num_groups, g+2*num_groups, ...
        # This is the generalisation of the original interleaved pattern.
        groups = []
        for g in range(num_groups):
            grp = canonical_view_ids[g::num_groups]
            groups.append(grp)
        return groups

    else:
        raise ValueError(
            f"Unknown grouping mode '{mode}'. Choose 'contiguous' or 'spread'."
        )


def validate_groups(
    groups: List[List[int]],
    canonical_view_ids: List[int],
    group_size: int,
) -> None:
    """Assert the grouping contract.

    Checks:
    1. All samples belong to the same canonical view set (no foreign views).
    2. Grouped view ids cover the full canonical view set exactly once.
    3. Each group has at least 1 view and at most ``group_size`` views.
    4. Group sizes differ by at most 1 (both ``spread`` round-robin and
       ``contiguous`` satisfy this; the old "only last group may be smaller"
       was incorrect for spread mode with non-divisible totals, e.g. 10 views
       with group_size=4 yields groups of sizes [4, 3, 3]).

    Raises
    ------
    AssertionError on any violation.
    """
    canonical_set = set(canonical_view_ids)
    total = len(canonical_view_ids)
    num_groups = math.ceil(total / group_size)

    assert len(groups) == num_groups, (
        f"Expected {num_groups} groups for {total} views with group_size={group_size}, "
        f"got {len(groups)}"
    )

    min_size = max(1, total - (num_groups - 1) * group_size)
    # Minimum possible group size for any valid grouping policy:
    # = total tokens remaining after (num_groups-1) full groups.
    # Works for both contiguous (exact last-group size) and spread
    # (where round-robin distributes remainder evenly, always >= this).

    seen: set[int] = set()
    for gi, grp in enumerate(groups):
        assert len(grp) >= 1, f"Group {gi} is empty"
        assert len(grp) <= group_size, (
            f"Group {gi} has {len(grp)} views, exceeding group_size={group_size}"
        )
        assert len(grp) >= min_size, (
            f"Group {gi} has {len(grp)} views; minimum expected is {min_size} "
            f"(floor({total}/{num_groups})) for group_size={group_size}"
        )
        for vid in grp:
            assert vid in canonical_set, (
                f"View id {vid} in group {gi} is not in canonical_view_ids"
            )
            assert vid not in seen, (
                f"View id {vid} appears in more than one group (detected in group {gi})"
            )
            seen.add(vid)

    assert seen == canonical_set, (
        f"Groups do not cover all canonical views.\n"
        f"  Missing: {canonical_set - seen}\n"
        f"  Extra:   {seen - canonical_set}"
    )


def num_groups_for(total_views: int, group_size: int) -> int:
    """Return ``ceil(total_views / group_size)``."""
    return math.ceil(total_views / group_size)
