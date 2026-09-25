from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class LearnedConcept:
    concept_id: str
    group_name: str
    center: dict[str, float]
    contrast: dict[str, float]
    strongest_features: tuple[str, ...]


def _mean(rows: list[dict[str, float]], key: str) -> float:
    values = [row.get(key, 0.0) for row in rows]
    return sum(values) / len(values) if values else 0.0


def _std(rows: list[dict[str, float]], key: str) -> float:
    if not rows:
        return 0.0
    mean = _mean(rows, key)
    variance = sum((row.get(key, 0.0) - mean) ** 2 for row in rows)
    return sqrt(variance / len(rows))


def learn_group_concept(
    group_name: str,
    positives: list[dict[str, float]],
    negatives: list[dict[str, float]],
    concept_number: int = 1,
) -> LearnedConcept:
    keys = sorted({key for row in positives + negatives for key in row})

    center = {key: _mean(positives, key) for key in keys}
    contrast: dict[str, float] = {}

    for key in keys:
        pos_mean = _mean(positives, key)
        neg_mean = _mean(negatives, key)
        scale = _std(positives + negatives, key)
        contrast[key] = 0.0 if scale < 1e-9 else (pos_mean - neg_mean) / scale

    strongest = tuple(
        key
        for key, _ in sorted(
            contrast.items(),
            key=lambda item: (-abs(item[1]), item[0]),
        )[:8]
    )

    return LearnedConcept(
        concept_id=f"concept_{concept_number:03d}",
        group_name=group_name,
        center=center,
        contrast=contrast,
        strongest_features=strongest,
    )


def score_against_concept(
    features: dict[str, float],
    concept: LearnedConcept,
) -> float:
    if not concept.strongest_features:
        return 0.0

    distance = 0.0
    used = 0

    for key in concept.strongest_features:
        weight = abs(concept.contrast.get(key, 0.0))
        if weight <= 0.0:
            continue

        target = concept.center.get(key, 0.0)
        value = features.get(key, 0.0)
        scale = max(abs(target), 1.0)
        distance += weight * abs(value - target) / scale
        used += 1

    return 0.0 if used == 0 else 1.0 / (1.0 + distance / used)
