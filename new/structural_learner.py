from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from new.executor import Grid, copy_grid, palette, shape
from new.structural_perception import (
    Mask,
    Tile,
    color_bbox,
    color_mask_in_bbox,
    infer_periodic_template,
    mask_signature,
    minority_motif,
    render_periodic,
    tile_mask,
)


@dataclass(frozen=True)
class StructuralProgram:
    output_shape: tuple[int, int]
    period: tuple[int, int]
    overlay_color: int
    overlay_bbox: tuple[int, int, int, int]
    overlay_repeat: tuple[int, int]
    motif_to_tile: dict[str, Tile]

    def apply(self, input_grid: Grid) -> Grid | None:
        motif = minority_motif(input_grid)
        if motif is None:
            return None

        key = mask_signature(motif.mask)
        tile = self.motif_to_tile.get(key)
        if tile is None:
            # Important: do not pretend to understand a new structural shape.
            # A later relation learner can replace this lookup with a true rule.
            return None

        out_h, out_w = self.output_shape
        result = render_periodic(tile, out_h, out_w)

        top, left, bottom, right = self.overlay_bbox
        overlay_mask = tile_mask(
            motif.mask,
            self.overlay_repeat[0],
            self.overlay_repeat[1],
        )

        expected_h = bottom - top + 1
        expected_w = right - left + 1
        if len(overlay_mask) != expected_h:
            return None
        if overlay_mask and len(overlay_mask[0]) != expected_w:
            return None

        for r in range(expected_h):
            for c in range(expected_w):
                if overlay_mask[r][c]:
                    result[top + r][left + c] = self.overlay_color

        return result


@dataclass(frozen=True)
class StructuralLearningResult:
    program: StructuralProgram | None
    exact_train: bool
    evidence: tuple[str, ...]
    unresolved: tuple[str, ...]


def learn_periodic_motif_program(
    pairs: list[dict[str, Any]],
) -> StructuralLearningResult:
    """
    Learn a reusable structural decomposition:

        input motif
            -> normalized shape
            -> repeated overlay mask

        output
            -> periodic base template
            + overlay

    This is deliberately relationship-first. It learns WHERE a motif goes
    and HOW its shape is reused, rather than searching raw pixel edits.
    """
    if not pairs:
        return StructuralLearningResult(None, False, (), ("no training pairs",))

    output_shapes = {shape(pair["output"]) for pair in pairs}
    if len(output_shapes) != 1:
        return StructuralLearningResult(
            None,
            False,
            (),
            ("output shape is not constant",),
        )
    output_shape = next(iter(output_shapes))

    added_sets: list[set[int]] = []
    for pair in pairs:
        input_colors = palette(pair["input"])
        output_colors = palette(pair["output"])
        added_sets.append(output_colors - input_colors)

    common_added = set.intersection(*added_sets) if added_sets else set()
    if len(common_added) != 1:
        return StructuralLearningResult(
            None,
            False,
            (),
            ("could not identify one common output overlay color",),
        )
    overlay_color = next(iter(common_added))

    overlay_bboxes = [
        color_bbox(pair["output"], overlay_color)
        for pair in pairs
    ]
    if any(bbox is None for bbox in overlay_bboxes):
        return StructuralLearningResult(
            None,
            False,
            (),
            ("overlay color missing from a training output",),
        )

    unique_bboxes = {bbox for bbox in overlay_bboxes if bbox is not None}
    if len(unique_bboxes) != 1:
        return StructuralLearningResult(
            None,
            False,
            (),
            ("overlay region is not structurally stable",),
        )
    overlay_bbox = next(iter(unique_bboxes))

    motifs = [minority_motif(pair["input"]) for pair in pairs]
    if any(motif is None for motif in motifs):
        return StructuralLearningResult(
            None,
            False,
            (),
            ("could not isolate an input motif",),
        )

    repeat_factors: set[tuple[int, int]] = set()
    motif_to_tile: dict[str, Tile] = {}
    periods: set[tuple[int, int]] = set()
    evidence: list[str] = []

    top, left, bottom, right = overlay_bbox
    overlay_h = bottom - top + 1
    overlay_w = right - left + 1

    for index, (pair, motif) in enumerate(zip(pairs, motifs), start=1):
        assert motif is not None

        motif_h = len(motif.mask)
        motif_w = len(motif.mask[0]) if motif.mask else 0
        if not motif_h or not motif_w:
            return StructuralLearningResult(
                None,
                False,
                tuple(evidence),
                ("empty motif",),
            )

        if overlay_h % motif_h or overlay_w % motif_w:
            return StructuralLearningResult(
                None,
                False,
                tuple(evidence),
                ("overlay dimensions are not integer repeats of the motif",),
            )

        row_repeat = overlay_h // motif_h
        col_repeat = overlay_w // motif_w
        repeated = tile_mask(motif.mask, row_repeat, col_repeat)
        observed_overlay = color_mask_in_bbox(
            pair["output"],
            overlay_color,
            overlay_bbox,
        )

        if repeated != observed_overlay:
            return StructuralLearningResult(
                None,
                False,
                tuple(evidence),
                (
                    f"pair {index}: output overlay is not a repeated input motif",
                ),
            )

        repeat_factors.add((row_repeat, col_repeat))

        periodic = infer_periodic_template(
            pair["output"],
            ignore_color=overlay_color,
        )
        if periodic is None:
            return StructuralLearningResult(
                None,
                False,
                tuple(evidence),
                (f"pair {index}: no exact periodic base found",),
            )

        periods.add(periodic.period)
        key = mask_signature(motif.mask)

        existing = motif_to_tile.get(key)
        if existing is not None and existing != periodic.tile:
            return StructuralLearningResult(
                None,
                False,
                tuple(evidence),
                (
                    "same motif shape maps to conflicting periodic templates",
                ),
            )

        motif_to_tile[key] = periodic.tile
        evidence.append(
            f"pair {index}: motif {key} repeats "
            f"{row_repeat}x{col_repeat} into overlay; "
            f"base period={periodic.period}"
        )

    if len(repeat_factors) != 1:
        return StructuralLearningResult(
            None,
            False,
            tuple(evidence),
            ("motif repeat factor changes across training pairs",),
        )

    if len(periods) != 1:
        return StructuralLearningResult(
            None,
            False,
            tuple(evidence),
            ("periodic base dimensions change across training pairs",),
        )

    program = StructuralProgram(
        output_shape=output_shape,
        period=next(iter(periods)),
        overlay_color=overlay_color,
        overlay_bbox=overlay_bbox,
        overlay_repeat=next(iter(repeat_factors)),
        motif_to_tile=motif_to_tile,
    )

    exact = True
    for pair in pairs:
        prediction = program.apply(pair["input"])
        if prediction != pair["output"]:
            exact = False
            break

    unresolved: list[str] = []
    if len(motif_to_tile) == len(pairs):
        unresolved.append(
            "periodic tile is still associated with observed motif shape; "
            "a new motif needs a learned relational rule before test prediction"
        )

    return StructuralLearningResult(
        program=program,
        exact_train=exact,
        evidence=tuple(evidence),
        unresolved=tuple(unresolved),
    )
