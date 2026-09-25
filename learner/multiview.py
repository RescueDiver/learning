from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class ViewResult:
    name: str
    ordered_categories: tuple[str, ...]
    selected_features: tuple[str, ...]


VIEW_ORDERS: tuple[tuple[str, tuple[str, ...]], ...] = (
    (
        "geometry_first",
        ("geometry", "change", "output", "reconstruction"),
    ),
    (
        "change_first",
        ("change", "output", "reconstruction", "geometry"),
    ),
    (
        "output_first",
        ("output", "reconstruction", "change", "geometry"),
    ),
    (
        "reconstruction_first",
        ("reconstruction", "geometry", "output", "change"),
    ),
)


def _category(feature: str) -> str:
    name = feature.removeprefix("all_")

    if (
        "solid_component" in name
        or "matches_solid" in name
        or "same_shape" in name
    ):
        return "geometry"

    if (
        "color_removed" in name
        or "some_color_removed" in name
    ):
        return "change"

    if (
        "output_smaller" in name
        or "output_matches" in name
    ):
        return "output"

    if (
        "crop" in name
        or "transformed" in name
    ):
        return "reconstruction"

    return "other"


def _mean(rows: list[dict[str, float]], key: str) -> float:
    if not rows:
        return 0.0
    return sum(row.get(key, 0.0) for row in rows) / len(rows)


def _std(rows: list[dict[str, float]], key: str) -> float:
    if not rows:
        return 0.0

    mean = _mean(rows, key)
    variance = sum(
        (row.get(key, 0.0) - mean) ** 2
        for row in rows
    ) / len(rows)
    return sqrt(variance)


def _discrimination(
    key: str,
    positives: list[dict[str, float]],
    negatives: list[dict[str, float]],
) -> float:
    all_rows = positives + negatives
    scale = _std(all_rows, key)

    if scale < 1e-9:
        return 0.0

    return (_mean(positives, key) - _mean(negatives, key)) / scale


def _candidate_features(
    positives: list[dict[str, float]],
    negatives: list[dict[str, float]],
) -> list[str]:
    keys = {
        key
        for row in positives + negatives
        for key in row
        if key.startswith("all_")
    }

    # Stage 5 is trying to find ideas characteristic of Eric's group.
    # Negative contrasts are useful later, but they are not "surviving ideas".
    return [
        key
        for key in keys
        if _discrimination(key, positives, negatives) > 0.0
    ]


def run_multiview_passes(
    positives: list[dict[str, float]],
    negatives: list[dict[str, float]],
    features_per_pass: int = 5,
) -> tuple[ViewResult, ...]:
    candidates = _candidate_features(positives, negatives)
    discrimination = {
        key: _discrimination(key, positives, negatives)
        for key in candidates
    }

    results: list[ViewResult] = []

    for view_name, category_order in VIEW_ORDERS:
        category_rank = {
            category: index
            for index, category in enumerate(category_order)
        }

        ordered = sorted(
            candidates,
            key=lambda key: (
                category_rank.get(_category(key), len(category_order)),
                -discrimination[key],
                key,
            ),
        )

        selected = tuple(ordered[:features_per_pass])

        results.append(
            ViewResult(
                name=view_name,
                ordered_categories=category_order,
                selected_features=selected,
            )
        )

    return tuple(results)


def surviving_ideas(
    passes: tuple[ViewResult, ...],
    minimum_passes: int = 3,
) -> tuple[tuple[str, int], ...]:
    counts: dict[str, int] = {}

    for result in passes:
        for feature in result.selected_features:
            counts[feature] = counts.get(feature, 0) + 1

    survivors = [
        (feature, count)
        for feature, count in counts.items()
        if count >= minimum_passes
    ]

    survivors.sort(key=lambda item: (-item[1], item[0]))
    return tuple(survivors)
