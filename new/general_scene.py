from __future__ import annotations

from collections import Counter, defaultdict, deque
from dataclasses import dataclass, asdict
from hashlib import sha1
from typing import Any, Iterable

Grid = list[list[int]]
Coord = tuple[int, int]
Mask = tuple[tuple[int, ...], ...]


@dataclass(frozen=True)
class SceneObject:
    object_id: str
    color: int | None
    cells: frozenset[Coord]
    bbox: tuple[int, int, int, int]
    height: int
    width: int
    area: int
    density: float
    touches_border: bool
    is_rectangle: bool
    is_square: bool
    is_line: bool
    has_hole: bool
    centroid: tuple[float, float]
    mask: Mask
    canonical_shape: str


@dataclass(frozen=True)
class Relation:
    left: str
    kind: str
    right: str
    value: int | float | str | None = None


@dataclass(frozen=True)
class SceneView:
    name: str
    background: int
    objects: tuple[SceneObject, ...]
    relations: tuple[Relation, ...]


@dataclass(frozen=True)
class SceneDescription:
    height: int
    width: int
    palette: tuple[int, ...]
    color_counts: tuple[tuple[int, int], ...]
    dominant_color: int
    horizontal_symmetric: bool
    vertical_symmetric: bool
    diagonal_symmetric: bool
    anti_diagonal_symmetric: bool
    row_period: int | None
    col_period: int | None
    views: tuple[SceneView, ...]


@dataclass(frozen=True)
class ObjectMatch:
    input_id: str
    output_id: str
    score: float
    same_color: bool
    same_shape: bool
    same_area: bool
    displacement: tuple[int, int]
    shape_relation: str


@dataclass(frozen=True)
class PairDescription:
    input_scene: SceneDescription
    output_scene: SceneDescription
    matches: tuple[ObjectMatch, ...]
    created_output_ids: tuple[str, ...]
    deleted_input_ids: tuple[str, ...]
    tokens: tuple[str, ...]
    details: tuple[str, ...]


def describe_scene(grid: Grid) -> SceneDescription:
    h, w = _shape(grid)
    counts = Counter(value for row in grid for value in row)
    palette = tuple(sorted(counts))
    dominant = counts.most_common(1)[0][0] if counts else 0

    views = (
        _make_view(grid, "color4", dominant, connectivity=4, respect_color=True),
        _make_view(grid, "color8", dominant, connectivity=8, respect_color=True),
        _make_view(grid, "foreground4", dominant, connectivity=4, respect_color=False),
        _make_view(grid, "foreground8", dominant, connectivity=8, respect_color=False),
    )

    return SceneDescription(
        height=h,
        width=w,
        palette=palette,
        color_counts=tuple(sorted(counts.items())),
        dominant_color=dominant,
        horizontal_symmetric=_hmirror(grid) == grid,
        vertical_symmetric=_vmirror(grid) == grid,
        diagonal_symmetric=h == w and _transpose(grid) == grid,
        anti_diagonal_symmetric=h == w and _anti_transpose(grid) == grid,
        row_period=_find_row_period(grid),
        col_period=_find_col_period(grid),
        views=views,
    )


def describe_pair(input_grid: Grid, output_grid: Grid) -> PairDescription:
    before = describe_scene(input_grid)
    after = describe_scene(output_grid)

    before_view = _view(before, "color4")
    after_view = _view(after, "color4")
    matches, deleted, created = _match_objects(before_view.objects, after_view.objects)

    tokens: set[str] = set()
    details: list[str] = []

    if (before.height, before.width) == (after.height, after.width):
        tokens.add("grid:size_preserved")
    else:
        tokens.add("grid:size_changed")
        if before.height and before.width:
            if after.height % before.height == 0 and after.width % before.width == 0:
                tokens.add("grid:integer_upscale")
        details.append(
            f"grid {before.height}x{before.width} -> {after.height}x{after.width}"
        )

    in_colors = set(before.palette)
    out_colors = set(after.palette)
    added_colors = sorted(out_colors - in_colors)
    removed_colors = sorted(in_colors - out_colors)

    if added_colors:
        tokens.add("color:added")
        details.append(f"colors added={added_colors}")
    if removed_colors:
        tokens.add("color:removed")
        details.append(f"colors removed={removed_colors}")
    if before.palette == after.palette:
        tokens.add("color:palette_preserved")
    if before.dominant_color == after.dominant_color:
        tokens.add("color:dominant_preserved")
    else:
        tokens.add("color:dominant_changed")

    for name, before_value, after_value in (
        ("horizontal", before.horizontal_symmetric, after.horizontal_symmetric),
        ("vertical", before.vertical_symmetric, after.vertical_symmetric),
        ("diagonal", before.diagonal_symmetric, after.diagonal_symmetric),
        ("anti_diagonal", before.anti_diagonal_symmetric, after.anti_diagonal_symmetric),
    ):
        if before_value and after_value:
            tokens.add(f"symmetry:{name}_preserved")
        elif not before_value and after_value:
            tokens.add(f"symmetry:{name}_created")
        elif before_value and not after_value:
            tokens.add(f"symmetry:{name}_broken")

    if before.row_period is not None:
        tokens.add("input:row_periodic")
    if before.col_period is not None:
        tokens.add("input:col_periodic")
    if after.row_period is not None:
        tokens.add("output:row_periodic")
    if after.col_period is not None:
        tokens.add("output:col_periodic")

    in_count = len(before_view.objects)
    out_count = len(after_view.objects)
    if in_count == out_count:
        tokens.add("objects:count_preserved")
    elif out_count > in_count:
        tokens.add("objects:created")
    else:
        tokens.add("objects:deleted")

    if created:
        details.append(f"created objects={len(created)}")
    if deleted:
        details.append(f"deleted objects={len(deleted)}")

    displacement_counts: Counter[tuple[int, int]] = Counter()
    recolored = 0
    shape_changed = 0
    same_shape = 0

    for match in matches:
        if match.same_shape:
            same_shape += 1
            tokens.add("objects:shape_preserved")
        else:
            shape_changed += 1
            tokens.add("objects:shape_changed")

        if not match.same_color:
            recolored += 1
            tokens.add("objects:recolored")
        else:
            tokens.add("objects:color_preserved")

        if match.displacement != (0, 0):
            displacement_counts[match.displacement] += 1
            tokens.add("objects:moved")
        else:
            tokens.add("objects:position_preserved")

        if match.shape_relation != "same":
            tokens.add(f"objects:{match.shape_relation}")

    if matches and len(matches) == in_count == out_count:
        tokens.add("objects:all_matched")

    if displacement_counts:
        if len(displacement_counts) == 1:
            tokens.add("movement:uniform_displacement")
            details.append(
                f"uniform displacement={next(iter(displacement_counts))}"
            )
        else:
            tokens.add("movement:mixed_displacements")

    if recolored and recolored == len(matches):
        tokens.add("objects:all_recolored")
    if same_shape and same_shape == len(matches):
        tokens.add("objects:all_shapes_preserved")
    if shape_changed:
        details.append(f"shape-changed matches={shape_changed}")

    _add_global_transform_tokens(input_grid, output_grid, tokens)
    _add_color_map_tokens(input_grid, output_grid, tokens, details)
    _add_relation_change_tokens(before_view, after_view, matches, tokens)

    return PairDescription(
        input_scene=before,
        output_scene=after,
        matches=matches,
        created_output_ids=tuple(created),
        deleted_input_ids=tuple(deleted),
        tokens=tuple(sorted(tokens)),
        details=tuple(details),
    )


def task_signature(train_pairs: list[dict[str, Any]]) -> dict[str, Any]:
    pair_descriptions = [
        describe_pair(pair["input"], pair["output"])
        for pair in train_pairs
    ]

    if not pair_descriptions:
        return {
            "pair_count": 0,
            "invariant_tokens": [],
            "any_tokens": [],
            "pair_tokens": [],
            "structural_facts": {},
        }

    token_sets = [set(pair.tokens) for pair in pair_descriptions]
    invariant = set.intersection(*token_sets)
    any_tokens = set.union(*token_sets)

    facts = _task_structural_facts(pair_descriptions)

    return {
        "pair_count": len(pair_descriptions),
        "invariant_tokens": sorted(invariant),
        "any_tokens": sorted(any_tokens),
        "pair_tokens": [list(pair.tokens) for pair in pair_descriptions],
        "pair_details": [list(pair.details) for pair in pair_descriptions],
        "structural_facts": facts,
    }


def compact_scene_dict(scene: SceneDescription) -> dict[str, Any]:
    return {
        "shape": [scene.height, scene.width],
        "palette": list(scene.palette),
        "dominant_color": scene.dominant_color,
        "symmetry": {
            "horizontal": scene.horizontal_symmetric,
            "vertical": scene.vertical_symmetric,
            "diagonal": scene.diagonal_symmetric,
            "anti_diagonal": scene.anti_diagonal_symmetric,
        },
        "period": {
            "rows": scene.row_period,
            "cols": scene.col_period,
        },
        "views": {
            view.name: {
                "object_count": len(view.objects),
                "relation_count": len(view.relations),
                "objects": [
                    {
                        "id": obj.object_id,
                        "color": obj.color,
                        "area": obj.area,
                        "bbox": list(obj.bbox),
                        "density": round(obj.density, 3),
                        "touches_border": obj.touches_border,
                        "rectangle": obj.is_rectangle,
                        "square": obj.is_square,
                        "line": obj.is_line,
                        "hole": obj.has_hole,
                        "shape": obj.canonical_shape,
                    }
                    for obj in view.objects
                ],
            }
            for view in scene.views
        },
    }


def _make_view(
    grid: Grid,
    name: str,
    background: int,
    connectivity: int,
    respect_color: bool,
) -> SceneView:
    components = _components(
        grid,
        background=background,
        connectivity=connectivity,
        respect_color=respect_color,
    )

    objects = tuple(
        _make_object(
            grid,
            cells,
            object_id=f"{name}:{index}",
            color=color,
        )
        for index, (cells, color) in enumerate(components)
    )

    return SceneView(
        name=name,
        background=background,
        objects=objects,
        relations=tuple(_relations(objects)),
    )


def _components(
    grid: Grid,
    background: int,
    connectivity: int,
    respect_color: bool,
) -> list[tuple[set[Coord], int | None]]:
    h, w = _shape(grid)
    seen: set[Coord] = set()
    result: list[tuple[set[Coord], int | None]] = []

    directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
    if connectivity == 8:
        directions += [(-1, -1), (-1, 1), (1, -1), (1, 1)]

    for r in range(h):
        for c in range(w):
            if (r, c) in seen or grid[r][c] == background:
                continue

            start_color = grid[r][c]
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
                    if grid[nr][nc] == background:
                        continue
                    if respect_color and grid[nr][nc] != start_color:
                        continue

                    seen.add((nr, nc))
                    queue.append((nr, nc))

            color: int | None = start_color if respect_color else None
            result.append((cells, color))

    result.sort(
        key=lambda item: (
            min(r for r, _ in item[0]),
            min(c for _, c in item[0]),
            len(item[0]),
            item[1] if item[1] is not None else -1,
        )
    )
    return result


def _make_object(
    grid: Grid,
    cells: set[Coord],
    object_id: str,
    color: int | None,
) -> SceneObject:
    h, w = _shape(grid)
    top = min(r for r, _ in cells)
    bottom = max(r for r, _ in cells)
    left = min(c for _, c in cells)
    right = max(c for _, c in cells)
    height = bottom - top + 1
    width = right - left + 1
    area = len(cells)
    mask = _normalized_mask(cells)
    density = area / (height * width)

    row_values = {r for r, _ in cells}
    col_values = {c for _, c in cells}
    is_line = len(row_values) == 1 or len(col_values) == 1
    is_rectangle = area == height * width

    centroid = (
        sum(r for r, _ in cells) / area,
        sum(c for _, c in cells) / area,
    )

    return SceneObject(
        object_id=object_id,
        color=color,
        cells=frozenset(cells),
        bbox=(top, left, bottom, right),
        height=height,
        width=width,
        area=area,
        density=density,
        touches_border=any(
            r == 0 or c == 0 or r == h - 1 or c == w - 1
            for r, c in cells
        ),
        is_rectangle=is_rectangle,
        is_square=is_rectangle and height == width,
        is_line=is_line,
        has_hole=_has_hole(mask),
        centroid=centroid,
        mask=mask,
        canonical_shape=_canonical_shape(mask),
    )


def _relations(objects: tuple[SceneObject, ...]) -> list[Relation]:
    relations: list[Relation] = []

    for i, a in enumerate(objects):
        for b in objects[i + 1:]:
            at, al, ab, ar = a.bbox
            bt, bl, bb, br = b.bbox

            if ar < bl:
                relations.append(Relation(a.object_id, "left_of", b.object_id, bl - ar - 1))
            elif br < al:
                relations.append(Relation(b.object_id, "left_of", a.object_id, al - br - 1))

            if ab < bt:
                relations.append(Relation(a.object_id, "above", b.object_id, bt - ab - 1))
            elif bb < at:
                relations.append(Relation(b.object_id, "above", a.object_id, at - bb - 1))

            if at == bt:
                relations.append(Relation(a.object_id, "top_aligned", b.object_id))
            if ab == bb:
                relations.append(Relation(a.object_id, "bottom_aligned", b.object_id))
            if al == bl:
                relations.append(Relation(a.object_id, "left_aligned", b.object_id))
            if ar == br:
                relations.append(Relation(a.object_id, "right_aligned", b.object_id))

            if abs(a.centroid[0] - b.centroid[0]) < 1e-9:
                relations.append(Relation(a.object_id, "center_row_aligned", b.object_id))
            if abs(a.centroid[1] - b.centroid[1]) < 1e-9:
                relations.append(Relation(a.object_id, "center_col_aligned", b.object_id))

            if a.area == b.area:
                relations.append(Relation(a.object_id, "same_area", b.object_id))
            if a.canonical_shape == b.canonical_shape:
                relations.append(Relation(a.object_id, "same_shape_family", b.object_id))
            if a.mask == b.mask:
                relations.append(Relation(a.object_id, "same_shape", b.object_id))
            if a.color is not None and a.color == b.color:
                relations.append(Relation(a.object_id, "same_color", b.object_id))

            if _touching(a.cells, b.cells):
                relations.append(Relation(a.object_id, "touching", b.object_id))
            elif _adjacent(a.cells, b.cells):
                relations.append(Relation(a.object_id, "adjacent", b.object_id))

            if _bbox_contains(a.bbox, b.bbox):
                relations.append(Relation(a.object_id, "contains_bbox", b.object_id))
            elif _bbox_contains(b.bbox, a.bbox):
                relations.append(Relation(b.object_id, "contains_bbox", a.object_id))

            distance = _manhattan_between(a.cells, b.cells)
            relations.append(Relation(a.object_id, "manhattan_gap", b.object_id, distance))

    return relations


def _match_objects(
    inputs: tuple[SceneObject, ...],
    outputs: tuple[SceneObject, ...],
) -> tuple[tuple[ObjectMatch, ...], list[str], list[str]]:
    candidates: list[tuple[float, int, int, ObjectMatch]] = []

    for i, a in enumerate(inputs):
        for j, b in enumerate(outputs):
            relation = _shape_relation(a.mask, b.mask)
            same_shape = relation != "different"
            same_area = a.area == b.area
            same_color = a.color == b.color
            displacement = (
                b.bbox[0] - a.bbox[0],
                b.bbox[1] - a.bbox[1],
            )

            score = 0.0
            if same_shape:
                score += 5.0
            if relation == "same":
                score += 2.0
            if same_area:
                score += 2.0
            if same_color:
                score += 1.0

            size_penalty = abs(a.area - b.area) / max(a.area, b.area, 1)
            distance_penalty = (
                abs(a.centroid[0] - b.centroid[0])
                + abs(a.centroid[1] - b.centroid[1])
            ) / 20.0
            score -= size_penalty + distance_penalty

            candidates.append(
                (
                    score,
                    i,
                    j,
                    ObjectMatch(
                        input_id=a.object_id,
                        output_id=b.object_id,
                        score=score,
                        same_color=same_color,
                        same_shape=same_shape,
                        same_area=same_area,
                        displacement=displacement,
                        shape_relation=relation,
                    ),
                )
            )

    candidates.sort(
        key=lambda item: (
            -item[0],
            item[1],
            item[2],
        )
    )

    used_in: set[int] = set()
    used_out: set[int] = set()
    matches: list[ObjectMatch] = []

    for score, i, j, match in candidates:
        if i in used_in or j in used_out:
            continue

        # Avoid forcing a nonsensical match merely because two objects exist.
        if score < 1.0:
            continue

        used_in.add(i)
        used_out.add(j)
        matches.append(match)

    deleted = [
        obj.object_id
        for i, obj in enumerate(inputs)
        if i not in used_in
    ]
    created = [
        obj.object_id
        for j, obj in enumerate(outputs)
        if j not in used_out
    ]

    matches.sort(key=lambda match: match.input_id)
    return tuple(matches), deleted, created


def _add_global_transform_tokens(
    input_grid: Grid,
    output_grid: Grid,
    tokens: set[str],
) -> None:
    transforms = {
        "identity": input_grid,
        "rot90": _rot90(input_grid),
        "rot180": _rot180(input_grid),
        "rot270": _rot270(input_grid),
        "mirror_h": _hmirror(input_grid),
        "mirror_v": _vmirror(input_grid),
    }

    for name, candidate in transforms.items():
        if candidate == output_grid:
            tokens.add(f"global:{name}")


def _add_color_map_tokens(
    input_grid: Grid,
    output_grid: Grid,
    tokens: set[str],
    details: list[str],
) -> None:
    if _shape(input_grid) != _shape(output_grid):
        return

    mapping: dict[int, set[int]] = defaultdict(set)
    reverse: dict[int, set[int]] = defaultdict(set)

    for in_row, out_row in zip(input_grid, output_grid):
        for a, b in zip(in_row, out_row):
            mapping[a].add(b)
            reverse[b].add(a)

    if all(len(values) == 1 for values in mapping.values()):
        tokens.add("color:deterministic_cell_map")
        pairs = {
            source: next(iter(targets))
            for source, targets in mapping.items()
        }
        if all(source == target for source, target in pairs.items()):
            tokens.add("color:cell_identity")
        else:
            details.append(f"cell color map={pairs}")

    if (
        mapping
        and all(len(values) == 1 for values in mapping.values())
        and all(len(values) == 1 for values in reverse.values())
    ):
        tokens.add("color:bijective_cell_map")


def _add_relation_change_tokens(
    before: SceneView,
    after: SceneView,
    matches: tuple[ObjectMatch, ...],
    tokens: set[str],
) -> None:
    if len(matches) < 2:
        return

    in_to_out = {
        match.input_id: match.output_id
        for match in matches
    }

    before_relations = {
        (rel.left, rel.kind, rel.right)
        for rel in before.relations
        if rel.left in in_to_out and rel.right in in_to_out
        and rel.kind != "manhattan_gap"
    }
    mapped_before = {
        (in_to_out[left], kind, in_to_out[right])
        for left, kind, right in before_relations
    }
    after_relations = {
        (rel.left, rel.kind, rel.right)
        for rel in after.relations
        if rel.left in set(in_to_out.values())
        and rel.right in set(in_to_out.values())
        and rel.kind != "manhattan_gap"
    }

    if mapped_before and mapped_before == after_relations:
        tokens.add("relations:preserved")
    else:
        if mapped_before & after_relations:
            tokens.add("relations:partially_preserved")
        if mapped_before - after_relations:
            tokens.add("relations:removed")
        if after_relations - mapped_before:
            tokens.add("relations:added")


def _task_structural_facts(
    pairs: list[PairDescription],
) -> dict[str, Any]:
    input_object_counts = [
        len(_view(pair.input_scene, "color4").objects)
        for pair in pairs
    ]
    output_object_counts = [
        len(_view(pair.output_scene, "color4").objects)
        for pair in pairs
    ]
    match_counts = [len(pair.matches) for pair in pairs]

    return {
        "input_object_counts": input_object_counts,
        "output_object_counts": output_object_counts,
        "match_counts": match_counts,
        "input_shapes": [
            [pair.input_scene.height, pair.input_scene.width]
            for pair in pairs
        ],
        "output_shapes": [
            [pair.output_scene.height, pair.output_scene.width]
            for pair in pairs
        ],
        "stable_input_object_count": len(set(input_object_counts)) == 1,
        "stable_output_object_count": len(set(output_object_counts)) == 1,
        "all_pairs_have_matches": all(count > 0 for count in match_counts),
    }


def _view(scene: SceneDescription, name: str) -> SceneView:
    return next(view for view in scene.views if view.name == name)


def _shape(grid: Grid) -> tuple[int, int]:
    return len(grid), len(grid[0]) if grid else 0


def _normalized_mask(cells: Iterable[Coord]) -> Mask:
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


def _canonical_shape(mask: Mask) -> str:
    forms = _shape_forms(mask)
    serialized = ["/".join("".join(map(str, row)) for row in form) for form in forms]
    return min(serialized) if serialized else ""


def _shape_relation(a: Mask, b: Mask) -> str:
    if a == b:
        return "same"

    candidates = {
        "rot90": _mask_rot90(a),
        "rot180": _mask_rot180(a),
        "rot270": _mask_rot270(a),
        "mirror_h": _mask_hmirror(a),
        "mirror_v": _mask_vmirror(a),
    }

    for name, candidate in candidates.items():
        if candidate == b:
            return name

    if _canonical_shape(a) == _canonical_shape(b):
        return "dihedral_equivalent"

    return "different"


def _shape_forms(mask: Mask) -> list[Mask]:
    if not mask:
        return [()]

    r0 = mask
    r1 = _mask_rot90(r0)
    r2 = _mask_rot90(r1)
    r3 = _mask_rot90(r2)

    return [
        r0, r1, r2, r3,
        _mask_hmirror(r0),
        _mask_hmirror(r1),
        _mask_hmirror(r2),
        _mask_hmirror(r3),
    ]


def _has_hole(mask: Mask) -> bool:
    if not mask:
        return False

    h = len(mask)
    w = len(mask[0])
    outside: set[Coord] = set()
    queue: deque[Coord] = deque()

    for r in range(h):
        for c in (0, w - 1):
            if mask[r][c] == 0 and (r, c) not in outside:
                outside.add((r, c))
                queue.append((r, c))

    for c in range(w):
        for r in (0, h - 1):
            if mask[r][c] == 0 and (r, c) not in outside:
                outside.add((r, c))
                queue.append((r, c))

    while queue:
        r, c = queue.popleft()
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            nr, nc = r + dr, c + dc
            if not (0 <= nr < h and 0 <= nc < w):
                continue
            if mask[nr][nc] != 0 or (nr, nc) in outside:
                continue
            outside.add((nr, nc))
            queue.append((nr, nc))

    return any(
        mask[r][c] == 0 and (r, c) not in outside
        for r in range(h)
        for c in range(w)
    )


def _touching(a: frozenset[Coord], b: frozenset[Coord]) -> bool:
    return bool(a & b)


def _adjacent(a: frozenset[Coord], b: frozenset[Coord]) -> bool:
    for r, c in a:
        for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
            if (r + dr, c + dc) in b:
                return True
    return False


def _bbox_contains(
    outer: tuple[int, int, int, int],
    inner: tuple[int, int, int, int],
) -> bool:
    ot, ol, ob, oright = outer
    it, il, ib, iright = inner
    return (
        outer != inner
        and ot <= it
        and ol <= il
        and ob >= ib
        and oright >= iright
    )


def _manhattan_between(
    a: frozenset[Coord],
    b: frozenset[Coord],
) -> int:
    return min(
        abs(ar - br) + abs(ac - bc)
        for ar, ac in a
        for br, bc in b
    )


def _find_row_period(grid: Grid) -> int | None:
    h, _ = _shape(grid)
    for period in range(1, h):
        if all(grid[r] == grid[r % period] for r in range(h)):
            return period
    return None


def _find_col_period(grid: Grid) -> int | None:
    h, w = _shape(grid)
    if not h:
        return None

    for period in range(1, w):
        if all(
            grid[r][c] == grid[r][c % period]
            for r in range(h)
            for c in range(w)
        ):
            return period
    return None


def _hmirror(grid: Grid) -> Grid:
    return [row[:] for row in reversed(grid)]


def _vmirror(grid: Grid) -> Grid:
    return [list(reversed(row)) for row in grid]


def _transpose(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid)] if grid else []


def _anti_transpose(grid: Grid) -> Grid:
    return _rot180(_transpose(grid))


def _rot90(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid[::-1])] if grid else []


def _rot180(grid: Grid) -> Grid:
    return [list(reversed(row)) for row in reversed(grid)]


def _rot270(grid: Grid) -> Grid:
    return [list(row) for row in zip(*grid)][::-1] if grid else []


def _mask_rot90(mask: Mask) -> Mask:
    if not mask:
        return ()
    return tuple(
        tuple(row[c] for row in reversed(mask))
        for c in range(len(mask[0]))
    )


def _mask_rot180(mask: Mask) -> Mask:
    return _mask_rot90(_mask_rot90(mask))


def _mask_rot270(mask: Mask) -> Mask:
    return _mask_rot90(_mask_rot180(mask))


def _mask_hmirror(mask: Mask) -> Mask:
    return tuple(reversed(mask))


def _mask_vmirror(mask: Mask) -> Mask:
    return tuple(tuple(reversed(row)) for row in mask)
