from __future__ import annotations

from collections import Counter, deque
from dataclasses import dataclass
from typing import Any

Grid = list[list[int]]
Coord = tuple[int, int]


@dataclass(frozen=True)
class TaskFeatures:
    values: dict[str, float]


RELATION_KEYS = (
    "has_solid_component",
    "output_matches_solid_component_size",
    "output_matches_rotated_solid_component_size",
    "matching_solid_color_removed",
    "rotated_matching_solid_color_removed",
    "output_equals_input_crop",
    "output_equals_transformed_input_crop",
)


def _shape(grid: Grid) -> tuple[int, int]:
    if not grid:
        return 0, 0
    return len(grid), len(grid[0])


def _color_counts(grid: Grid) -> Counter[int]:
    return Counter(value for row in grid for value in row)


def _largest_solid_rectangle_ratio(grid: Grid) -> float:
    if not grid or not grid[0]:
        return 0.0

    positions: dict[int, list[Coord]] = {}
    for r, row in enumerate(grid):
        for c, value in enumerate(row):
            positions.setdefault(value, []).append((r, c))

    best = 0.0
    for cells in positions.values():
        top = min(r for r, _ in cells)
        bottom = max(r for r, _ in cells)
        left = min(c for _, c in cells)
        right = max(c for _, c in cells)
        area = (bottom - top + 1) * (right - left + 1)
        if area:
            best = max(best, len(cells) / area)

    return best


def _components_by_color(grid: Grid) -> list[dict[str, Any]]:
    if not grid or not grid[0]:
        return []

    height = len(grid)
    width = len(grid[0])
    seen: set[Coord] = set()
    components: list[dict[str, Any]] = []

    for r in range(height):
        for c in range(width):
            if (r, c) in seen:
                continue

            color = grid[r][c]
            queue = deque([(r, c)])
            seen.add((r, c))
            cells: list[Coord] = []

            while queue:
                cr, cc = queue.popleft()
                cells.append((cr, cc))

                for nr, nc in (
                    (cr - 1, cc),
                    (cr + 1, cc),
                    (cr, cc - 1),
                    (cr, cc + 1),
                ):
                    if not (0 <= nr < height and 0 <= nc < width):
                        continue
                    if (nr, nc) in seen:
                        continue
                    if grid[nr][nc] != color:
                        continue
                    seen.add((nr, nc))
                    queue.append((nr, nc))

            top = min(rr for rr, _ in cells)
            bottom = max(rr for rr, _ in cells)
            left = min(cc for _, cc in cells)
            right = max(cc for _, cc in cells)
            box_height = bottom - top + 1
            box_width = right - left + 1
            box_area = box_height * box_width

            components.append(
                {
                    "color": color,
                    "cells": cells,
                    "cell_count": len(cells),
                    "top": top,
                    "bottom": bottom,
                    "left": left,
                    "right": right,
                    "height": box_height,
                    "width": box_width,
                    "area": box_area,
                    "solid": len(cells) == box_area,
                }
            )

    return components


def _crop(grid: Grid, top: int, left: int, height: int, width: int) -> Grid:
    return [row[left:left + width] for row in grid[top:top + height]]


def _rotate_90(grid: Grid) -> Grid:
    if not grid:
        return []
    return [list(row) for row in zip(*grid[::-1])]


def _rotate_180(grid: Grid) -> Grid:
    return [row[::-1] for row in grid[::-1]]


def _rotate_270(grid: Grid) -> Grid:
    if not grid:
        return []
    return [list(row) for row in zip(*grid)][::-1]


def _mirror_horizontal(grid: Grid) -> Grid:
    return grid[::-1]


def _mirror_vertical(grid: Grid) -> Grid:
    return [row[::-1] for row in grid]


def _output_equals_any_crop(input_grid: Grid, output_grid: Grid) -> bool:
    ih, iw = _shape(input_grid)
    oh, ow = _shape(output_grid)

    if oh == 0 or ow == 0 or oh > ih or ow > iw:
        return False

    for top in range(ih - oh + 1):
        for left in range(iw - ow + 1):
            if _crop(input_grid, top, left, oh, ow) == output_grid:
                return True

    return False


def _output_equals_transformed_crop(input_grid: Grid, output_grid: Grid) -> bool:
    ih, iw = _shape(input_grid)
    oh, ow = _shape(output_grid)

    for height, width in {(oh, ow), (ow, oh)}:
        if height <= 0 or width <= 0 or height > ih or width > iw:
            continue

        for top in range(ih - height + 1):
            for left in range(iw - width + 1):
                crop = _crop(input_grid, top, left, height, width)
                transforms = (
                    _rotate_90(crop),
                    _rotate_180(crop),
                    _rotate_270(crop),
                    _mirror_horizontal(crop),
                    _mirror_vertical(crop),
                )
                if any(candidate == output_grid for candidate in transforms):
                    return True

    return False


def _relation_features(input_grid: Grid, output_grid: Grid) -> dict[str, float]:
    output_height, output_width = _shape(output_grid)
    output_colors = set(_color_counts(output_grid))

    components = _components_by_color(input_grid)
    solid_components = [
        component
        for component in components
        if component["solid"] and component["cell_count"] >= 4
    ]

    matching_size = [
        component
        for component in solid_components
        if (
            component["height"] == output_height
            and component["width"] == output_width
        )
    ]

    matching_rotated_size = [
        component
        for component in solid_components
        if (
            component["height"] == output_width
            and component["width"] == output_height
        )
    ]

    matching_removed = [
        component
        for component in matching_size
        if component["color"] not in output_colors
    ]

    rotated_matching_removed = [
        component
        for component in matching_rotated_size
        if component["color"] not in output_colors
    ]

    largest_solid_area = max(
        (component["area"] for component in solid_components),
        default=0,
    )
    output_area = output_height * output_width

    return {
        "has_solid_component": float(bool(solid_components)),
        "solid_component_count": float(len(solid_components)),
        "output_matches_solid_component_size": float(bool(matching_size)),
        "output_matches_rotated_solid_component_size": float(
            bool(matching_rotated_size)
        ),
        "matching_solid_color_removed": float(bool(matching_removed)),
        "rotated_matching_solid_color_removed": float(
            bool(rotated_matching_removed)
        ),
        "largest_solid_component_area_ratio_to_output": (
            float(largest_solid_area) / output_area if output_area else 0.0
        ),
        "output_equals_input_crop": float(
            _output_equals_any_crop(input_grid, output_grid)
        ),
        "output_equals_transformed_input_crop": float(
            _output_equals_transformed_crop(input_grid, output_grid)
        ),
    }


def extract_pair_features(pair: dict[str, Any]) -> dict[str, float]:
    input_grid: Grid = pair["input"]
    output_grid: Grid = pair["output"]

    ih, iw = _shape(input_grid)
    oh, ow = _shape(output_grid)

    input_counts = _color_counts(input_grid)
    output_counts = _color_counts(output_grid)

    input_colors = set(input_counts)
    output_colors = set(output_counts)

    input_area = ih * iw
    output_area = oh * ow

    features = {
        "same_shape": float((ih, iw) == (oh, ow)),
        "input_height": float(ih),
        "input_width": float(iw),
        "output_height": float(oh),
        "output_width": float(ow),
        "input_area": float(input_area),
        "output_area": float(output_area),
        "area_ratio": float(output_area) / input_area if input_area else 0.0,
        "input_color_count": float(len(input_colors)),
        "output_color_count": float(len(output_colors)),
        "colors_removed": float(len(input_colors - output_colors)),
        "colors_added": float(len(output_colors - input_colors)),
        "largest_solid_rectangle_ratio": _largest_solid_rectangle_ratio(input_grid),
    }

    features.update(_relation_features(input_grid, output_grid))
    return features


def extract_task_features(task: dict[str, Any]) -> TaskFeatures:
    pairs = task.get("train", [])
    if not pairs:
        return TaskFeatures(values={})

    per_pair = [extract_pair_features(pair) for pair in pairs]
    keys = sorted(per_pair[0])

    values: dict[str, float] = {}
    for key in keys:
        series = [features[key] for features in per_pair]
        values[f"mean_{key}"] = sum(series) / len(series)
        values[f"range_{key}"] = max(series) - min(series)

    values["train_pair_count"] = float(len(pairs))
    return TaskFeatures(values=values)


def extract_task_rule_features(task: dict[str, Any]) -> TaskFeatures:
    """
    Stage 3 representation.

    Instead of averaging a task into one blob, preserve whether the SAME
    relationship holds across every training pair. These are candidate
    "rules" or invariants, not mere visual statistics.
    """
    pairs = task.get("train", [])
    if not pairs:
        return TaskFeatures(values={})

    per_pair = [extract_pair_features(pair) for pair in pairs]
    values: dict[str, float] = {}

    for key in RELATION_KEYS:
        series = [pair_features[key] for pair_features in per_pair]
        values[f"all_{key}"] = float(all(value >= 0.5 for value in series))
        values[f"any_{key}"] = float(any(value >= 0.5 for value in series))
        values[f"fraction_{key}"] = sum(series) / len(series)
        values[f"consistent_{key}"] = float(
            max(series) - min(series) < 1e-9
        )

    # A few structural invariants that matter across pairs without caring
    # about the literal dimensions or colors.
    same_shape_series = [pair_features["same_shape"] for pair_features in per_pair]
    shrink_series = [
        float(pair_features["area_ratio"] < 1.0)
        for pair_features in per_pair
    ]
    color_removed_series = [
        float(pair_features["colors_removed"] > 0.0)
        for pair_features in per_pair
    ]

    values["all_same_shape"] = float(all(same_shape_series))
    values["all_output_smaller"] = float(all(shrink_series))
    values["all_some_color_removed"] = float(all(color_removed_series))
    values["train_pair_count"] = float(len(pairs))

    return TaskFeatures(values=values)



def extract_task_router_features(task: dict[str, Any]) -> TaskFeatures:
    """
    Rich router vocabulary.

    The coarse router uses rule invariants. Specialist group scorers use this
    larger vocabulary so each human group can learn its own most useful
    questions from both task appearance and input/output relationships.
    """
    broad = extract_task_features(task).values
    rules = extract_task_rule_features(task).values

    values = dict(broad)
    values.update(rules)
    return TaskFeatures(values=values)
