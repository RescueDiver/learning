from __future__ import annotations

from collections import Counter, deque
from copy import deepcopy
from typing import Any, Callable


Grid = list[list[int]]
Coord = tuple[int, int]
ObjectCells = set[Coord]


def copy_grid(grid: Grid) -> Grid:
    return [row[:] for row in grid]


def shape(grid: Grid) -> tuple[int, int]:
    return (len(grid), len(grid[0]) if grid else 0)


def palette(grid: Grid) -> set[int]:
    return {value for row in grid for value in row}


def mostcolor(grid: Grid) -> int:
    counts = Counter(value for row in grid for value in row)
    return counts.most_common(1)[0][0]


def leastcolor(grid: Grid) -> int:
    counts = Counter(value for row in grid for value in row)
    return min(counts, key=lambda value: (counts[value], value))


def rot90(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid[::-1])] if grid else []


def rot180(grid: Grid) -> Grid:
    return [row[::-1] for row in grid[::-1]]


def rot270(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid)][::-1] if grid else []


def hmirror(grid: Grid) -> Grid:
    return [row[:] for row in grid[::-1]]


def vmirror(grid: Grid) -> Grid:
    return [row[::-1] for row in grid]


def crop(grid: Grid, top: int, left: int, height: int, width: int) -> Grid:
    return [row[left:left + width] for row in grid[top:top + height]]


def canvas(color: int, height: int, width: int) -> Grid:
    return [[color for _ in range(width)] for _ in range(height)]


def replace(grid: Grid, source: int, target: int) -> Grid:
    return [
        [target if value == source else value for value in row]
        for row in grid
    ]


def switch(grid: Grid, a: int, b: int) -> Grid:
    result = copy_grid(grid)
    for r, row in enumerate(result):
        for c, value in enumerate(row):
            if value == a:
                result[r][c] = b
            elif value == b:
                result[r][c] = a
    return result


def fill(grid: Grid, color: int, cells: list[Coord] | set[Coord]) -> Grid:
    result = copy_grid(grid)
    h, w = shape(result)
    for r, c in cells:
        if 0 <= r < h and 0 <= c < w:
            result[r][c] = color
    return result


def upscale(grid: Grid, row_factor: int, col_factor: int | None = None) -> Grid:
    if col_factor is None:
        col_factor = row_factor
    result: Grid = []
    for row in grid:
        expanded = [
            value
            for value in row
            for _ in range(col_factor)
        ]
        for _ in range(row_factor):
            result.append(expanded[:])
    return result


def downscale(grid: Grid, factor: int) -> Grid:
    if factor <= 0:
        raise ValueError("factor must be positive")
    return [
        [grid[r][c] for c in range(0, len(grid[0]), factor)]
        for r in range(0, len(grid), factor)
    ]


def hconcat(a: Grid, b: Grid) -> Grid:
    if len(a) != len(b):
        raise ValueError("grids must have same height")
    return [ra + rb for ra, rb in zip(a, b)]


def vconcat(a: Grid, b: Grid) -> Grid:
    if a and b and len(a[0]) != len(b[0]):
        raise ValueError("grids must have same width")
    return copy_grid(a) + copy_grid(b)


def trim(grid: Grid, background: int | None = None) -> Grid:
    if not grid:
        return []
    if background is None:
        background = mostcolor(grid)

    cells = [
        (r, c)
        for r, row in enumerate(grid)
        for c, value in enumerate(row)
        if value != background
    ]
    if not cells:
        return [[]]

    top = min(r for r, _ in cells)
    bottom = max(r for r, _ in cells)
    left = min(c for _, c in cells)
    right = max(c for _, c in cells)
    return crop(grid, top, left, bottom - top + 1, right - left + 1)


def objects(
    grid: Grid,
    background: int | None = None,
    diagonal: bool = False,
) -> list[dict[str, Any]]:
    if not grid:
        return []
    if background is None:
        background = mostcolor(grid)

    h, w = shape(grid)
    seen: set[Coord] = set()
    result: list[dict[str, Any]] = []

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if diagonal:
        directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for r in range(h):
        for c in range(w):
            if (r, c) in seen or grid[r][c] == background:
                continue

            color = grid[r][c]
            queue = deque([(r, c)])
            seen.add((r, c))
            cells: set[Coord] = set()

            while queue:
                cr, cc = queue.popleft()
                cells.add((cr, cc))
                for dr, dc in directions:
                    nr, nc = cr + dr, cc + dc
                    if not (0 <= nr < h and 0 <= nc < w):
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
            result.append(
                {
                    "color": color,
                    "cells": cells,
                    "size": len(cells),
                    "bbox": (top, left, bottom, right),
                    "height": bottom - top + 1,
                    "width": right - left + 1,
                }
            )

    return result


def shift_object(
    obj: dict[str, Any],
    dr: int,
    dc: int,
) -> dict[str, Any]:
    moved = deepcopy(obj)
    moved["cells"] = {(r + dr, c + dc) for r, c in obj["cells"]}
    top, left, bottom, right = obj["bbox"]
    moved["bbox"] = (top + dr, left + dc, bottom + dr, right + dc)
    return moved


def paint_object(grid: Grid, obj: dict[str, Any]) -> Grid:
    return fill(grid, int(obj["color"]), obj["cells"])


def cover_object(
    grid: Grid,
    obj: dict[str, Any],
    background: int | None = None,
) -> Grid:
    if background is None:
        background = mostcolor(grid)
    return fill(grid, background, obj["cells"])


def box_cells(top: int, left: int, bottom: int, right: int) -> set[Coord]:
    cells: set[Coord] = set()
    for c in range(left, right + 1):
        cells.add((top, c))
        cells.add((bottom, c))
    for r in range(top, bottom + 1):
        cells.add((r, left))
        cells.add((r, right))
    return cells


def rectangle_cells(top: int, left: int, bottom: int, right: int) -> set[Coord]:
    return {
        (r, c)
        for r in range(top, bottom + 1)
        for c in range(left, right + 1)
    }


def connect(a: Coord, b: Coord) -> set[Coord]:
    ar, ac = a
    br, bc = b

    if ar == br:
        lo, hi = sorted((ac, bc))
        return {(ar, c) for c in range(lo, hi + 1)}

    if ac == bc:
        lo, hi = sorted((ar, br))
        return {(r, ac) for r in range(lo, hi + 1)}

    if abs(ar - br) == abs(ac - bc):
        steps = abs(ar - br)
        dr = 1 if br > ar else -1
        dc = 1 if bc > ac else -1
        return {(ar + i * dr, ac + i * dc) for i in range(steps + 1)}

    return set()


def normalize_object(obj: dict[str, Any]) -> dict[str, Any]:
    top, left, _, _ = obj["bbox"]
    return shift_object(obj, -top, -left)


def execute_word(name: str, value: Any, **args: Any) -> Any:
    registry: dict[str, Callable[..., Any]] = {
        "identity": lambda x, **_: x,
        "rot90": lambda x, **_: rot90(x),
        "rot180": lambda x, **_: rot180(x),
        "rot270": lambda x, **_: rot270(x),
        "hmirror": lambda x, **_: hmirror(x),
        "vmirror": lambda x, **_: vmirror(x),
        "crop": lambda x, **kw: crop(x, kw["top"], kw["left"], kw["height"], kw["width"]),
        "replace": lambda x, **kw: replace(x, kw["source"], kw["target"]),
        "switch": lambda x, **kw: switch(x, kw["a"], kw["b"]),
        "upscale": lambda x, **kw: upscale(x, kw["row_factor"], kw.get("col_factor")),
        "downscale": lambda x, **kw: downscale(x, kw["factor"]),
        "trim": lambda x, **kw: trim(x, kw.get("background")),
    }

    if name not in registry:
        raise KeyError(f"word is known but not executable yet: {name}")

    return registry[name](value, **args)
