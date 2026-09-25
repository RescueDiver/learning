from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Any

Grid = list[list[int]]


@dataclass(frozen=True)
class TaskFeatures:
    values: dict[str, float]


def _shape(grid: Grid) -> tuple[int, int]:
    if not grid:
        return 0, 0
    return len(grid), len(grid[0])


def _color_counts(grid: Grid) -> Counter[int]:
    return Counter(value for row in grid for value in row)


def _largest_solid_rectangle_ratio(grid: Grid) -> float:
    if not grid or not grid[0]:
        return 0.0

    positions: dict[int, list[tuple[int, int]]] = {}
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


def _pair_features(pair: dict[str, Any]) -> dict[str, float]:
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

    return {
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


def extract_task_features(task: dict[str, Any]) -> TaskFeatures:
    pairs = task.get("train", [])
    if not pairs:
        return TaskFeatures(values={})

    per_pair = [_pair_features(pair) for pair in pairs]
    keys = sorted(per_pair[0])

    values: dict[str, float] = {}
    for key in keys:
        series = [features[key] for features in per_pair]
        values[f"mean_{key}"] = sum(series) / len(series)
        values[f"range_{key}"] = max(series) - min(series)

    values["train_pair_count"] = float(len(pairs))
    return TaskFeatures(values=values)
