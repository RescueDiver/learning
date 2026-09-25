from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from new.executor import Grid, palette, shape
from new.motif_tile_relation import (
    RelationExample,
    RelationRule,
    apply_relation_rule,
    learn_motif_to_tile_relation,
)
from new.structural_perception import (
    Tile,
    color_bbox,
    color_mask_in_bbox,
    dominant_color,
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
    relation_rule: RelationRule | None = None
    tile_other_color: int | None = None

    def apply(self, input_grid: Grid) -> Grid | None:
        motif = minority_motif(input_grid)
        if motif is None:
            return None

        tile: Tile | None = None

        if self.relation_rule is not None and self.tile_other_color is not None:
            binary_tile = apply_relation_rule(
                self.relation_rule,
                motif,
            )
            input_background = dominant_color(input_grid)
            tile = tuple(
                tuple(
                    input_background if value else self.tile_other_color
                    for value in row
                )
                for row in binary_tile
            )
        else:
            key = mask_signature(motif.mask)
            tile = self.motif_to_tile.get(key)

        if tile is None:
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
    Learn a structural decomposition:

        input motif
            -> normalized shape
            -> repeated overlay mask

        output
            -> periodic base template
            + overlay

    Then learn the missing relationship:

        input motif
            -> periodic base tile

    That second relationship is validated with leave-one-out folds before it
    is allowed to predict an unseen test motif.
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

    motifs = [minority_motif(pair["input"]) for pair in pairs]
    if any(motif is None for motif in motifs):
        return StructuralLearningResult(
            None,
            False,
            (),
            ("could not isolate an input motif",),
        )

    overlay_choice = _choose_overlay_color(
        pairs=pairs,
        motifs=motifs,
        candidate_colors=common_added,
    )
    if overlay_choice is None:
        return StructuralLearningResult(
            None,
            False,
            (),
            (
                "could not identify an added color whose stable region "
                "matches a repeated input motif",
            ),
        )

    overlay_color, overlay_bbox, overlay_repeat = overlay_choice

    repeat_factors: set[tuple[int, int]] = set()
    motif_to_tile: dict[str, Tile] = {}
    periods: set[tuple[int, int]] = set()
    tile_other_colors: set[int] = set()
    relation_examples: list[RelationExample] = []
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

        if (row_repeat, col_repeat) != overlay_repeat:
            return StructuralLearningResult(
                None,
                False,
                tuple(evidence),
                ("motif repeat factor changed after overlay selection",),
            )

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

        input_background = dominant_color(pair["input"])
        tile_values = {
            value
            for row in periodic.tile
            for value in row
        }

        if input_background in tile_values and len(tile_values) == 2:
            other_color = next(
                value
                for value in tile_values
                if value != input_background
            )
            tile_other_colors.add(other_color)

            target = tuple(
                tuple(
                    1 if value == input_background else 0
                    for value in row
                )
                for row in periodic.tile
            )
            relation_examples.append(
                RelationExample(
                    motif=motif,
                    target=target,
                )
            )

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

    relation_rule: RelationRule | None = None
    relation_other_color: int | None = None
    unresolved: list[str] = []

    if (
        len(relation_examples) == len(pairs)
        and len(tile_other_colors) == 1
    ):
        relation_result = learn_motif_to_tile_relation(
            relation_examples,
        )

        evidence.append(
            f"motif->tile relation candidates searched: "
            f"{relation_result.candidate_count}"
        )
        evidence.extend(
            f"LOO {detail}"
            for detail in relation_result.fold_details
        )

        if (
            relation_result.rule is not None
            and relation_result.exact_all
            and relation_result.loo_passed
        ):
            relation_rule = relation_result.rule
            relation_other_color = next(iter(tile_other_colors))
            evidence.append(
                f"accepted motif->tile rule: {relation_rule.name}"
            )
        else:
            unresolved.append(
                "no reusable motif->periodic-tile rule passed leave-one-out"
            )
    else:
        unresolved.append(
            "periodic tile does not have one stable binary color relationship"
        )

    program = StructuralProgram(
        output_shape=output_shape,
        period=next(iter(periods)),
        overlay_color=overlay_color,
        overlay_bbox=overlay_bbox,
        overlay_repeat=overlay_repeat,
        motif_to_tile=motif_to_tile,
        relation_rule=relation_rule,
        tile_other_color=relation_other_color,
    )

    exact = all(
        program.apply(pair["input"]) == pair["output"]
        for pair in pairs
    )

    # If no relation rule passed LOO, exact training reconstruction can still
    # come from the observed motif->tile associations. That is useful for
    # structural diagnosis, but it is not allowed to generalize to a new motif.
    if relation_rule is None and len(motif_to_tile) == len(pairs):
        unresolved.append(
            "training is exact by observed motif/template associations only; "
            "unseen motifs remain blocked from prediction"
        )

    return StructuralLearningResult(
        program=program,
        exact_train=exact,
        evidence=tuple(evidence),
        unresolved=tuple(unresolved),
    )


def _choose_overlay_color(
    pairs: list[dict[str, Any]],
    motifs: list[Any],
    candidate_colors: set[int],
) -> tuple[int, tuple[int, int, int, int], tuple[int, int]] | None:
    """
    Pick the added output color by structural evidence.

    A valid overlay color must:
      1. have the same bounding box in every training output,
      2. have a box that is an integer multiple of each input motif,
      3. reproduce the output color mask exactly when the motif is tiled.
    """
    choices: list[
        tuple[int, tuple[int, int, int, int], tuple[int, int]]
    ] = []

    for color in sorted(candidate_colors):
        bboxes = [color_bbox(pair["output"], color) for pair in pairs]
        if any(bbox is None for bbox in bboxes):
            continue

        unique_bboxes = {bbox for bbox in bboxes if bbox is not None}
        if len(unique_bboxes) != 1:
            continue

        bbox = next(iter(unique_bboxes))
        top, left, bottom, right = bbox
        overlay_h = bottom - top + 1
        overlay_w = right - left + 1

        repeat: tuple[int, int] | None = None
        valid = True

        for pair, motif in zip(pairs, motifs):
            if motif is None or not motif.mask:
                valid = False
                break

            motif_h = len(motif.mask)
            motif_w = len(motif.mask[0])

            if overlay_h % motif_h or overlay_w % motif_w:
                valid = False
                break

            current_repeat = (
                overlay_h // motif_h,
                overlay_w // motif_w,
            )

            if repeat is None:
                repeat = current_repeat
            elif repeat != current_repeat:
                valid = False
                break

            predicted_mask = tile_mask(
                motif.mask,
                current_repeat[0],
                current_repeat[1],
            )
            observed_mask = color_mask_in_bbox(
                pair["output"],
                color,
                bbox,
            )

            if predicted_mask != observed_mask:
                valid = False
                break

        if valid and repeat is not None:
            choices.append((color, bbox, repeat))

    if not choices:
        return None

    choices.sort(
        key=lambda item: (
            (item[1][2] - item[1][0] + 1)
            * (item[1][3] - item[1][1] + 1),
            item[0],
        )
    )
    return choices[0]
