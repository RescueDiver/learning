from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Any

from new.demonstration import Demonstration, Step, exact_on_pairs
from new.executor import palette, shape


@dataclass(frozen=True)
class LearnedProgram:
    demonstration: Demonstration
    exact_train: bool
    source: str


def infer_simple_programs(
    pairs: list[dict[str, Any]],
    max_steps: int = 2,
) -> tuple[LearnedProgram, ...]:
    """
    Small proof-of-concept program learner over imported ARC vocabulary.

    This is not meant to brute-force the entire ARC-DSL. It shows the new
    architecture: build task-supported words/parameters, compose them, then
    let exact reconstruction decide.
    """
    if not pairs:
        return ()

    candidate_steps = _candidate_steps(pairs)
    learned: list[LearnedProgram] = []

    for step in candidate_steps:
        demo = Demonstration(
            name=f"learned__{step.word}",
            steps=(step,),
        )
        if exact_on_pairs(pairs, demo):
            learned.append(
                LearnedProgram(
                    demonstration=demo,
                    exact_train=True,
                    source="single_word",
                )
            )

    if learned or max_steps < 2:
        return tuple(learned)

    for first, second in product(candidate_steps, repeat=2):
        demo = Demonstration(
            name=f"learned__{first.word}__{second.word}",
            steps=(first, second),
        )
        if exact_on_pairs(pairs, demo):
            learned.append(
                LearnedProgram(
                    demonstration=demo,
                    exact_train=True,
                    source="two_word_composition",
                )
            )
            # Keep the first few exact programs. They are hypotheses; a later
            # simplicity/generalization stage can choose among them.
            if len(learned) >= 12:
                break

    return tuple(learned)


def _candidate_steps(
    pairs: list[dict[str, Any]],
) -> tuple[Step, ...]:
    steps: list[Step] = [
        Step("identity"),
        Step("rot90"),
        Step("rot180"),
        Step("rot270"),
        Step("hmirror"),
        Step("vmirror"),
        Step("trim"),
    ]

    input_colors = set()
    output_colors = set()

    for pair in pairs:
        input_colors.update(palette(pair["input"]))
        output_colors.update(palette(pair["output"]))

    for source in sorted(input_colors):
        for target in sorted(output_colors):
            if source != target:
                steps.append(
                    Step(
                        "replace",
                        {"source": source, "target": target},
                    )
                )

    # Scale candidates are supported only when dimensions across every pair
    # agree on the same integer factor.
    factors: set[tuple[int, int]] = set()
    factor_possible = True

    for pair in pairs:
        ih, iw = shape(pair["input"])
        oh, ow = shape(pair["output"])

        if ih == 0 or iw == 0 or oh % ih or ow % iw:
            factor_possible = False
            break

        factors.add((oh // ih, ow // iw))

    if factor_possible and len(factors) == 1:
        row_factor, col_factor = next(iter(factors))
        if row_factor >= 1 and col_factor >= 1 and (row_factor, col_factor) != (1, 1):
            steps.append(
                Step(
                    "upscale",
                    {
                        "row_factor": row_factor,
                        "col_factor": col_factor,
                    },
                )
            )

    return tuple(_dedupe_steps(steps))


def _dedupe_steps(steps: list[Step]) -> list[Step]:
    unique: list[Step] = []
    seen: set[str] = set()

    for step in steps:
        key = repr((step.word, sorted(step.args.items())))
        if key in seen:
            continue
        seen.add(key)
        unique.append(step)

    return unique
