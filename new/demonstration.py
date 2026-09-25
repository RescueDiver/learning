from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from new.executor import (
    Grid,
    copy_grid,
    execute_word,
    mostcolor,
    objects,
    paint_object,
    cover_object,
    shift_object,
)


@dataclass(frozen=True)
class Step:
    """
    One human-teachable action.

    Examples:
        Step("select_object", {"by": "largest"})
        Step("move_selected", {"dr": 0, "dc": 3})
        Step("paint_selected")
        Step("rot90")

    This is intentionally readable. A demonstration should describe what the
    person sees and does, not hide the answer inside task-specific code.
    """
    word: str
    args: dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class Demonstration:
    name: str
    steps: tuple[Step, ...]


@dataclass
class ExecutionState:
    grid: Grid
    selected: dict[str, Any] | None = None
    background: int | None = None


def execute_demonstration(
    input_grid: Grid,
    demonstration: Demonstration,
) -> Grid:
    state = ExecutionState(
        grid=copy_grid(input_grid),
        background=mostcolor(input_grid),
    )

    for step in demonstration.steps:
        _execute_step(state, step)

    return state.grid


def exact_on_pairs(
    pairs: list[dict[str, Any]],
    demonstration: Demonstration,
) -> bool:
    if not pairs:
        return False

    for pair in pairs:
        prediction = execute_demonstration(
            pair["input"],
            demonstration,
        )
        if prediction != pair["output"]:
            return False

    return True


def _execute_step(state: ExecutionState, step: Step) -> None:
    word = step.word
    args = step.args

    if word == "select_object":
        candidates = objects(
            state.grid,
            background=state.background,
            diagonal=bool(args.get("diagonal", False)),
        )
        if not candidates:
            state.selected = None
            return

        mode = args.get("by", "largest")

        if mode == "largest":
            state.selected = max(candidates, key=lambda obj: obj["size"])
        elif mode == "smallest":
            state.selected = min(candidates, key=lambda obj: obj["size"])
        elif mode == "color":
            color = int(args["color"])
            state.selected = next(
                (obj for obj in candidates if obj["color"] == color),
                None,
            )
        elif mode == "index":
            state.selected = candidates[int(args["index"])]
        else:
            raise ValueError(f"unknown object selector: {mode}")
        return

    if word == "move_selected":
        if state.selected is None:
            return

        dr = _resolve_int(args.get("dr", 0), state)
        dc = _resolve_int(args.get("dc", 0), state)

        if args.get("cover_source", True):
            state.grid = cover_object(
                state.grid,
                state.selected,
                background=state.background,
            )

        state.selected = shift_object(state.selected, dr, dc)

        if args.get("paint", True):
            state.grid = paint_object(state.grid, state.selected)
        return

    if word == "paint_selected":
        if state.selected is not None:
            state.grid = paint_object(state.grid, state.selected)
        return

    if word == "recolor_selected":
        if state.selected is None:
            return
        color = int(args["color"])
        state.selected = dict(state.selected)
        state.selected["color"] = color
        state.grid = paint_object(state.grid, state.selected)
        return

    if word == "copy_selected":
        if state.selected is None:
            return
        dr = _resolve_int(args.get("dr", 0), state)
        dc = _resolve_int(args.get("dc", 0), state)
        copied = shift_object(state.selected, dr, dc)
        state.grid = paint_object(state.grid, copied)
        return

    # Pure grid words use the executable ARC vocabulary.
    state.grid = execute_word(word, state.grid, **args)


def _resolve_int(value: Any, state: ExecutionState) -> int:
    if isinstance(value, int):
        return value

    if value == "selected_height":
        return int(state.selected["height"]) if state.selected else 0
    if value == "selected_width":
        return int(state.selected["width"]) if state.selected else 0
    if value == "negative_selected_height":
        return -int(state.selected["height"]) if state.selected else 0
    if value == "negative_selected_width":
        return -int(state.selected["width"]) if state.selected else 0

    raise ValueError(f"cannot resolve integer parameter: {value!r}")
