from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Iterable

from new.executor import Grid, objects, shape


Mask = tuple[tuple[int, ...], ...]
Tile = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class Motif:
    color: int
    bbox: tuple[int, int, int, int]
    mask: Mask


@dataclass(frozen=True)
class ObjectRelation:
    left: int
    right: int
    kind: str
    detail: str


@dataclass(frozen=True)
class SceneStructure:
    background: int
    motifs: tuple[Motif, ...]
    relations: tuple[ObjectRelation, ...]


@dataclass(frozen=True)
class PeriodicTemplate:
    period: tuple[int, int]
    tile: Tile


def perceive_scene(grid: Grid) -> SceneStructure:
    """
    Build a structural description instead of treating the grid as raw pixels.

    The representation deliberately separates:
      object identity / shape
      position
      relationships between objects

    This is the ARC equivalent of saying "eye", "cheekbone", and
    "their relative placement" instead of memorizing face pixels.
    """
    background = dominant_color(grid)
    scene_objects = objects(grid, background=background, diagonal=False)

    motifs: list[Motif] = []
    for obj in scene_objects:
        motifs.append(
            Motif(
                color=int(obj["color"]),
                bbox=tuple(obj["bbox"]),
                mask=normalized_mask(obj["cells"]),
            )
        )

    relations = infer_object_relations(scene_objects)

    return SceneStructure(
        background=background,
        motifs=tuple(motifs),
        relations=tuple(relations),
    )


def dominant_color(grid: Grid) -> int:
    counts = Counter(value for row in grid for value in row)
    return counts.most_common(1)[0][0]


def minority_motif(grid: Grid) -> Motif | None:
    """
    Return the non-background color pattern as one normalized motif.

    Unlike connected-component extraction, this keeps disconnected pieces
    of the same minority color together. That is useful when a symbol is
    made of several separated marks.
    """
    background = dominant_color(grid)
    counts = Counter(value for row in grid for value in row)

    candidates = [
        (count, color)
        for color, count in counts.items()
        if color != background
    ]
    if not candidates:
        return None

    _, color = min(candidates)
    cells = {
        (r, c)
        for r, row in enumerate(grid)
        for c, value in enumerate(row)
        if value == color
    }

    top = min(r for r, _ in cells)
    bottom = max(r for r, _ in cells)
    left = min(c for _, c in cells)
    right = max(c for _, c in cells)

    return Motif(
        color=color,
        bbox=(top, left, bottom, right),
        mask=normalized_mask(cells),
    )


def normalized_mask(cells: Iterable[tuple[int, int]]) -> Mask:
    cells = set(cells)
    if not cells:
        return ()

    top = min(r for r, _ in cells)
    bottom = max(r for r, _ in cells)
    left = min(c for _, c in cells)
    right = max(c for _, c in cells)

    return tuple(
        tuple(
            1 if (r, c) in cells else 0
            for c in range(left, right + 1)
        )
        for r in range(top, bottom + 1)
    )


def mask_signature(mask: Mask) -> str:
    if not mask:
        return "empty"
    return "/".join(
        "".join(str(value) for value in row)
        for row in mask
    )


def color_bbox(grid: Grid, color: int) -> tuple[int, int, int, int] | None:
    cells = [
        (r, c)
        for r, row in enumerate(grid)
        for c, value in enumerate(row)
        if value == color
    ]
    if not cells:
        return None

    return (
        min(r for r, _ in cells),
        min(c for _, c in cells),
        max(r for r, _ in cells),
        max(c for _, c in cells),
    )


def color_mask_in_bbox(
    grid: Grid,
    color: int,
    bbox: tuple[int, int, int, int],
) -> Mask:
    top, left, bottom, right = bbox
    return tuple(
        tuple(
            1 if grid[r][c] == color else 0
            for c in range(left, right + 1)
        )
        for r in range(top, bottom + 1)
    )


def tile_mask(mask: Mask, row_repeats: int, col_repeats: int) -> Mask:
    if not mask:
        return ()

    tiled_rows: list[tuple[int, ...]] = []
    for _ in range(row_repeats):
        for row in mask:
            tiled_rows.append(
                tuple(
                    value
                    for _ in range(col_repeats)
                    for value in row
                )
            )
    return tuple(tiled_rows)


def infer_periodic_template(
    grid: Grid,
    ignore_color: int | None = None,
    max_period: int = 8,
) -> PeriodicTemplate | None:
    """
    Find the smallest exact row/column period, allowing one overlay color to
    be ignored. Every visible cell with the same residue must agree.
    """
    height, width = shape(grid)
    if not height or not width:
        return None

    best: PeriodicTemplate | None = None

    for pr in range(1, min(max_period, height) + 1):
        for pc in range(1, min(max_period, width) + 1):
            residue_values: dict[tuple[int, int], set[int]] = {}

            for r in range(height):
                for c in range(width):
                    value = grid[r][c]
                    if ignore_color is not None and value == ignore_color:
                        continue

                    residue_values.setdefault((r % pr, c % pc), set()).add(value)

            if not residue_values:
                continue

            if any(len(values) != 1 for values in residue_values.values()):
                continue

            # Require every residue position to have at least one visible
            # supporting cell. Otherwise an overlay could hide an entire
            # residue and produce a fake period.
            if len(residue_values) != pr * pc:
                continue

            tile = tuple(
                tuple(
                    next(iter(residue_values[(r, c)]))
                    for c in range(pc)
                )
                for r in range(pr)
            )
            candidate = PeriodicTemplate(period=(pr, pc), tile=tile)

            if best is None:
                best = candidate
                continue

            old_area = best.period[0] * best.period[1]
            new_area = pr * pc
            if (new_area, pr, pc) < (old_area, best.period[0], best.period[1]):
                best = candidate

    return best


def render_periodic(
    tile: Tile,
    height: int,
    width: int,
) -> Grid:
    if not tile:
        raise ValueError("cannot render an empty tile")

    th = len(tile)
    tw = len(tile[0])

    return [
        [tile[r % th][c % tw] for c in range(width)]
        for r in range(height)
    ]


def infer_object_relations(
    scene_objects: list[dict],
) -> list[ObjectRelation]:
    relations: list[ObjectRelation] = []

    for i, left in enumerate(scene_objects):
        for j in range(i + 1, len(scene_objects)):
            right = scene_objects[j]

            ltop, lleft, lbottom, lright = left["bbox"]
            rtop, rleft, rbottom, rright = right["bbox"]

            if lright < rleft:
                relations.append(ObjectRelation(i, j, "left_of", "bbox"))
            elif rright < lleft:
                relations.append(ObjectRelation(j, i, "left_of", "bbox"))

            if lbottom < rtop:
                relations.append(ObjectRelation(i, j, "above", "bbox"))
            elif rbottom < ltop:
                relations.append(ObjectRelation(j, i, "above", "bbox"))

            if ltop == rtop:
                relations.append(ObjectRelation(i, j, "top_aligned", "bbox"))
            if lbottom == rbottom:
                relations.append(ObjectRelation(i, j, "bottom_aligned", "bbox"))
            if lleft == rleft:
                relations.append(ObjectRelation(i, j, "left_aligned", "bbox"))
            if lright == rright:
                relations.append(ObjectRelation(i, j, "right_aligned", "bbox"))

            if left["size"] == right["size"]:
                relations.append(ObjectRelation(i, j, "same_size", "cell_count"))

            if normalized_mask(left["cells"]) == normalized_mask(right["cells"]):
                relations.append(ObjectRelation(i, j, "same_shape", "normalized"))

    return relations
