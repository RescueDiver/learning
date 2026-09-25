from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations_with_replacement
from typing import Iterable

from new.structural_perception import Mask, Motif


@dataclass(frozen=True)
class RelationRule:
    """
    A small reusable rule mapping an input motif to a binary periodic tile.

    Rules are built from a fixed vocabulary of structural masks and at most
    one Boolean composition. They never contain task IDs or stored answers.
    """
    op: str
    left: str
    right: str | None = None
    complexity: int = 1

    @property
    def name(self) -> str:
        if self.right is None:
            return self.left
        return f"{self.op}({self.left},{self.right})"


@dataclass(frozen=True)
class RelationExample:
    motif: Motif
    target: Mask


@dataclass(frozen=True)
class RelationLearningResult:
    rule: RelationRule | None
    exact_all: bool
    loo_passed: bool
    fold_details: tuple[str, ...]
    candidate_count: int


def learn_motif_to_tile_relation(
    examples: list[RelationExample],
) -> RelationLearningResult:
    if len(examples) < 2:
        return RelationLearningResult(
            rule=None,
            exact_all=False,
            loo_passed=False,
            fold_details=("need at least two examples",),
            candidate_count=0,
        )

    rules = _candidate_rules(examples[0].motif)
    exact_all = [
        rule
        for rule in rules
        if _matches_examples(rule, examples)
    ]

    # Deterministic leave-one-out: learn the simplest rule on n-1 examples
    # without looking at the held-out target, then test it on the holdout.
    fold_details: list[str] = []
    fold_winners: list[RelationRule] = []
    loo_passed = True

    for held_out in range(len(examples)):
        train = [
            example
            for index, example in enumerate(examples)
            if index != held_out
        ]

        fitting = [
            rule
            for rule in rules
            if _matches_examples(rule, train)
        ]
        if not fitting:
            loo_passed = False
            fold_details.append(
                f"holdout {held_out + 1}: no rule fits remaining examples"
            )
            continue

        winner = min(
            fitting,
            key=lambda rule: (
                rule.complexity,
                rule.name,
            ),
        )
        fold_winners.append(winner)

        predicted = apply_relation_rule(
            winner,
            examples[held_out].motif,
        )
        passed = predicted == examples[held_out].target
        loo_passed = loo_passed and passed
        fold_details.append(
            f"holdout {held_out + 1}: {winner.name} -> "
            f"{'PASS' if passed else 'FAIL'}"
        )

    if not exact_all:
        return RelationLearningResult(
            rule=None,
            exact_all=False,
            loo_passed=False,
            fold_details=tuple(fold_details),
            candidate_count=len(rules),
        )

    best_all = min(
        exact_all,
        key=lambda rule: (
            rule.complexity,
            rule.name,
        ),
    )

    # Stronger than merely fitting all examples: the final rule must be no
    # more complex than what each LOO fold selected. This keeps us from
    # accepting a complicated post-hoc formula discovered only after seeing
    # every answer.
    if fold_winners:
        max_fold_complexity = max(rule.complexity for rule in fold_winners)
        if best_all.complexity > max_fold_complexity:
            loo_passed = False

    return RelationLearningResult(
        rule=best_all,
        exact_all=True,
        loo_passed=loo_passed,
        fold_details=tuple(fold_details),
        candidate_count=len(rules),
    )


def apply_relation_rule(
    rule: RelationRule,
    motif: Motif,
) -> Mask:
    bases = _base_masks(motif)

    if rule.left not in bases:
        raise KeyError(f"unknown relation base: {rule.left}")

    left = bases[rule.left]

    if rule.right is None:
        return left

    if rule.right not in bases:
        raise KeyError(f"unknown relation base: {rule.right}")

    right = bases[rule.right]
    return _combine(left, right, rule.op)


def _candidate_rules(motif: Motif) -> tuple[RelationRule, ...]:
    names = tuple(sorted(_base_masks(motif)))

    rules: list[RelationRule] = [
        RelationRule(
            op="base",
            left=name,
            complexity=1,
        )
        for name in names
    ]

    for left, right in combinations_with_replacement(names, 2):
        for op in ("and", "or", "xor", "left_minus_right", "right_minus_left"):
            rules.append(
                RelationRule(
                    op=op,
                    left=left,
                    right=right,
                    complexity=2,
                )
            )

    return tuple(rules)


def _matches_examples(
    rule: RelationRule,
    examples: Iterable[RelationExample],
) -> bool:
    return all(
        apply_relation_rule(rule, example.motif) == example.target
        for example in examples
    )


def _base_masks(motif: Motif) -> dict[str, Mask]:
    mask = motif.mask
    if not mask:
        return {"empty": ()}

    h = len(mask)
    w = len(mask[0])

    bases: dict[str, Mask] = {}

    def add(name: str, value: Mask) -> None:
        if len(value) == h and (not value or len(value[0]) == w):
            bases[name] = value

    add("identity", mask)
    add("not", _invert(mask))

    # Dihedral transforms are useful when the motif is square. For non-square
    # motifs only transforms that preserve dimensions survive add().
    r90 = _rot90(mask)
    r180 = _rot90(r90)
    r270 = _rot90(r180)

    add("rot90", r90)
    add("rot180", r180)
    add("rot270", r270)
    add("mirror_h", _mirror_h(mask))
    add("mirror_v", _mirror_v(mask))
    add("not_rot90", _invert(r90))
    add("not_rot180", _invert(r180))
    add("not_rot270", _invert(r270))
    add("not_mirror_h", _invert(_mirror_h(mask)))
    add("not_mirror_v", _invert(_mirror_v(mask)))

    # Relative movement vocabulary. Zero-filled shifts represent moving a
    # structural feature, not wrapping pixels around the edge.
    for dr in range(-2, 3):
        for dc in range(-2, 3):
            if dr == 0 and dc == 0:
                continue
            add(
                f"shift_{dr}_{dc}",
                _shift(mask, dr, dc),
            )
            add(
                f"not_shift_{dr}_{dc}",
                _invert(_shift(mask, dr, dc)),
            )

    row_counts = [sum(row) for row in mask]
    col_counts = [
        sum(mask[r][c] for r in range(h))
        for c in range(w)
    ]

    for count in range(0, w + 1):
        add(
            f"row_count_{count}",
            tuple(
                tuple(1 if row_counts[r] == count else 0 for _ in range(w))
                for r in range(h)
            ),
        )

    for count in range(0, h + 1):
        add(
            f"col_count_{count}",
            tuple(
                tuple(1 if col_counts[c] == count else 0 for c in range(w))
                for _ in range(h)
            ),
        )

    add(
        "row_has_mark",
        tuple(
            tuple(1 if row_counts[r] > 0 else 0 for _ in range(w))
            for r in range(h)
        ),
    )
    add(
        "col_has_mark",
        tuple(
            tuple(1 if col_counts[c] > 0 else 0 for c in range(w))
            for _ in range(h)
        ),
    )

    # Local structural summaries.
    add("neighbor_any", _neighbor_summary(mask, mode="any"))
    add("neighbor_all", _neighbor_summary(mask, mode="all"))
    add("neighbor_one", _neighbor_summary(mask, mode="one"))
    add("diagonal_any", _diagonal_summary(mask, mode="any"))
    add("diagonal_one", _diagonal_summary(mask, mode="one"))

    # Geometry of the motif location. These masks let the learner discover
    # phase relationships without memorizing absolute coordinates.
    top, left, _, _ = motif.bbox
    for phase in range(3):
        add(
            f"bbox_row_phase_{phase}",
            tuple(
                tuple(1 if (top + r) % 3 == phase else 0 for _ in range(w))
                for r in range(h)
            ),
        )
        add(
            f"bbox_col_phase_{phase}",
            tuple(
                tuple(1 if (left + c) % 3 == phase else 0 for c in range(w))
                for _ in range(h)
            ),
        )

    # Keep the vocabulary names stable across every example.
    #
    # Do NOT deduplicate names by their current mask value here. Two different
    # structural words can happen to produce the same mask for one motif but
    # different masks for another motif. Removing one name on a per-example
    # basis made a rule learned from pair 1 disappear when it was evaluated
    # on pair 2/3.
    return bases


def _combine(a: Mask, b: Mask, op: str) -> Mask:
    h = len(a)
    w = len(a[0]) if a else 0

    def cell(x: int, y: int) -> int:
        if op == "and":
            return x & y
        if op == "or":
            return x | y
        if op == "xor":
            return x ^ y
        if op == "left_minus_right":
            return x & (1 - y)
        if op == "right_minus_left":
            return y & (1 - x)
        raise ValueError(f"unknown relation operator: {op}")

    return tuple(
        tuple(cell(a[r][c], b[r][c]) for c in range(w))
        for r in range(h)
    )


def _invert(mask: Mask) -> Mask:
    return tuple(
        tuple(1 - value for value in row)
        for row in mask
    )


def _rot90(mask: Mask) -> Mask:
    if not mask:
        return ()
    return tuple(
        tuple(row[c] for row in reversed(mask))
        for c in range(len(mask[0]))
    )


def _mirror_h(mask: Mask) -> Mask:
    return tuple(reversed(mask))


def _mirror_v(mask: Mask) -> Mask:
    return tuple(tuple(reversed(row)) for row in mask)


def _shift(mask: Mask, dr: int, dc: int) -> Mask:
    h = len(mask)
    w = len(mask[0]) if mask else 0

    return tuple(
        tuple(
            mask[r - dr][c - dc]
            if 0 <= r - dr < h and 0 <= c - dc < w
            else 0
            for c in range(w)
        )
        for r in range(h)
    )


def _neighbor_summary(mask: Mask, mode: str) -> Mask:
    h = len(mask)
    w = len(mask[0]) if mask else 0

    result = []
    for r in range(h):
        row = []
        for c in range(w):
            values = [
                mask[nr][nc]
                for nr, nc in (
                    (r - 1, c),
                    (r + 1, c),
                    (r, c - 1),
                    (r, c + 1),
                )
                if 0 <= nr < h and 0 <= nc < w
            ]
            total = sum(values)

            if mode == "any":
                row.append(1 if total > 0 else 0)
            elif mode == "all":
                row.append(1 if values and total == len(values) else 0)
            elif mode == "one":
                row.append(1 if total == 1 else 0)
            else:
                raise ValueError(mode)
        result.append(tuple(row))

    return tuple(result)


def _diagonal_summary(mask: Mask, mode: str) -> Mask:
    h = len(mask)
    w = len(mask[0]) if mask else 0

    result = []
    for r in range(h):
        row = []
        for c in range(w):
            values = [
                mask[nr][nc]
                for nr, nc in (
                    (r - 1, c - 1),
                    (r - 1, c + 1),
                    (r + 1, c - 1),
                    (r + 1, c + 1),
                )
                if 0 <= nr < h and 0 <= nc < w
            ]
            total = sum(values)

            if mode == "any":
                row.append(1 if total > 0 else 0)
            elif mode == "one":
                row.append(1 if total == 1 else 0)
            else:
                raise ValueError(mode)
        result.append(tuple(row))

    return tuple(result)
