"""
Cubemap face grouping for anchor-weighted gradient accumulation.

Each effective training update consists of exactly 2 microbatches:
  set1 = [front, back, left, right]   (horizontal belt)
  set2 = [front, back, up, down]      (vertical cross)

front and back appear in BOTH sets — this is INTENTIONAL anchor weighting.
Effective per-sample weights per update (8 total slots):
  front : 2/8 = 0.25
  back  : 2/8 = 0.25
  left  : 1/8 = 0.125
  right : 1/8 = 0.125
  up    : 1/8 = 0.125
  down  : 1/8 = 0.125

This is NOT a bug. front/back are the canonical viewing anchors for
panoramic consistency and receive double gradient signal per update.
"""

from typing import Dict, List

# ── Canonical face ordering ─────────────────────────────────────────────────
CANONICAL_FACES: List[str] = ["front", "right", "back", "left", "up", "down"]
FACE_TO_IDX: Dict[str, int] = {f: i for i, f in enumerate(CANONICAL_FACES)}

# ── Fixed 2-group semantics ──────────────────────────────────────────────────
# set1: horizontal belt (front/back/left/right)
CUBEMAP_SET1: List[str] = ["front", "back", "left", "right"]
# set2: vertical cross (front/back/up/down)
CUBEMAP_SET2: List[str] = ["front", "back", "up", "down"]

# Total slots per effective update: 4 + 4 = 8
_TOTAL_SLOTS: int = len(CUBEMAP_SET1) + len(CUBEMAP_SET2)  # 8


def make_cubemap_groups(
    canonical_faces: List[str] = CANONICAL_FACES,
) -> List[List[str]]:
    """Return the two fixed face groups as lists of face names.

    The returned list is always [SET1, SET2] in that order.
    Front and back appear in both groups — intentional anchor weighting.

    Args:
        canonical_faces: Must equal CANONICAL_FACES (validated internally).

    Returns:
        [[front, back, left, right], [front, back, up, down]]
    """
    assert list(canonical_faces) == CANONICAL_FACES, (
        f"canonical_faces must be {CANONICAL_FACES}, got {canonical_faces}"
    )
    return [list(CUBEMAP_SET1), list(CUBEMAP_SET2)]


def validate_cubemap_groups(
    groups: List[List[str]],
    canonical_faces: List[str] = CANONICAL_FACES,
) -> None:
    """Assert all design invariants for cubemap 2-set grouping.

    Invariants:
    1. Exactly 2 groups.
    2. Each group has exactly 4 faces.
    3. set1 == CUBEMAP_SET1 (order-insensitive).
    4. set2 == CUBEMAP_SET2 (order-insensitive).
    5. front and back appear in both groups (anchor weighting).
    6. All face names are in CANONICAL_FACES.
    7. Canonical face list matches exactly.

    Raises:
        AssertionError on any violation.
    """
    assert list(canonical_faces) == CANONICAL_FACES, (
        f"canonical_faces must be {CANONICAL_FACES}, got {canonical_faces}"
    )
    assert len(groups) == 2, (
        f"Cubemap grouping requires exactly 2 groups, got {len(groups)}"
    )
    set1, set2 = groups
    assert len(set1) == 4, f"set1 must have 4 faces, got {len(set1)}: {set1}"
    assert len(set2) == 4, f"set2 must have 4 faces, got {len(set2)}: {set2}"

    assert set(set1) == set(CUBEMAP_SET1), (
        f"set1 must contain {set(CUBEMAP_SET1)}, got {set(set1)}"
    )
    assert set(set2) == set(CUBEMAP_SET2), (
        f"set2 must contain {set(CUBEMAP_SET2)}, got {set(set2)}"
    )

    # Verify anchor weighting: front and back must appear in both groups
    for anchor in ("front", "back"):
        assert anchor in set1, (
            f"Anchor '{anchor}' missing from set1 — breaks anchor weighting"
        )
        assert anchor in set2, (
            f"Anchor '{anchor}' missing from set2 — breaks anchor weighting"
        )

    # All face names valid
    for f in set1 + set2:
        assert f in FACE_TO_IDX, f"Unknown face name '{f}' in groups"


def cubemap_loss_divisors(total_slots: int = _TOTAL_SLOTS) -> List[float]:
    """Return per-group loss divisors for sample-aware loss weighting.

    With total_slots=8 and group_size=4:
        divisor_i = total_slots / group_size_i = 8/4 = 2.0

    The trainer divides loss by divisor, so:
        loss / divisor = loss * (group_size / total_slots)
    Summing across groups gives a full-update sample average.

    Returns:
        [2.0, 2.0]  — one per group
    """
    groups = make_cubemap_groups()
    divisors = []
    for g in groups:
        divisors.append(total_slots / float(len(g)))
    return divisors


def face_effective_weights(total_slots: int = _TOTAL_SLOTS) -> Dict[str, float]:
    """Return the effective gradient weight per face per update.

    Computed as: (number of times face appears across all groups) / total_slots

    Returns:
        {
            'front': 0.25,   # appears in set1 + set2
            'back':  0.25,   # appears in set1 + set2
            'left':  0.125,  # appears in set1 only
            'right': 0.125,  # appears in set1 only
            'up':    0.125,  # appears in set2 only
            'down':  0.125,  # appears in set2 only
        }
    """
    counts: Dict[str, int] = {f: 0 for f in CANONICAL_FACES}
    for g in make_cubemap_groups():
        for f in g:
            counts[f] += 1
    return {f: counts[f] / total_slots for f in CANONICAL_FACES}


def face_ids_for_group(group: List[str]) -> List[int]:
    """Convert a list of face names to their canonical integer IDs.

    Args:
        group: List of face name strings from CANONICAL_FACES.

    Returns:
        List of integer IDs (0=front, 1=right, 2=back, 3=left, 4=up, 5=down).
    """
    return [FACE_TO_IDX[f] for f in group]


# ── Duplication summary for metadata / logging ───────────────────────────────
DUPLICATION_SUMMARY: Dict = {
    "anchor_faces": ["front", "back"],
    "anchor_appears_in_sets": 2,
    "non_anchor_appears_in_sets": 1,
    "total_slots_per_update": _TOTAL_SLOTS,
    "effective_weights": face_effective_weights(),
    "note": (
        "front and back appear in both set1 and set2. "
        "This is intentional anchor weighting, not a data duplication bug."
    ),
}
