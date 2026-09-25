from __future__ import annotations

from dataclasses import dataclass
from math import sqrt


@dataclass(frozen=True)
class ViewResult:
    name: str
    ordered_categories: tuple[str, ...]
    selected_features: tuple[str, ...]


@dataclass(frozen=True)
class ComparedIdea:
    feature: str
    selected_in_passes: int
    total_passes: int
    teaching_tasks_supported: int
    total_teaching_tasks: int
    decision: str


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

    if "color_removed" in name or "some_color_removed" in name:
        return "change"

    if "output_smaller" in name or "output_matches" in name:
        return "output"

    if "crop" in name or "transformed" in name:
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

    return [
        key
        for key in keys
        if _discrimination(key, positives, negatives) > 0.0
    ]


def run_single_view(
    view_name: str,
    positives: list[dict[str, float]],
    negatives: list[dict[str, float]],
    features_per_pass: int = 5,
) -> ViewResult:
    view_lookup = dict(VIEW_ORDERS)

    if view_name not in view_lookup:
        raise ValueError(f"Unknown view: {view_name}")

    category_order = view_lookup[view_name]
    category_rank = {
        category: index
        for index, category in enumerate(category_order)
    }

    candidates = _candidate_features(positives, negatives)
    discrimination = {
        key: _discrimination(key, positives, negatives)
        for key in candidates
    }

    ordered = sorted(
        candidates,
        key=lambda key: (
            category_rank.get(_category(key), len(category_order)),
            -discrimination[key],
            key,
        ),
    )

    return ViewResult(
        name=view_name,
        ordered_categories=category_order,
        selected_features=tuple(ordered[:features_per_pass]),
    )


def compare_view_results(
    passes: tuple[ViewResult, ...],
    teaching_rows: dict[str, dict[str, float]],
) -> tuple[ComparedIdea, ...]:
    pass_counts: dict[str, int] = {}

    for result in passes:
        for feature in result.selected_features:
            pass_counts[feature] = pass_counts.get(feature, 0) + 1

    ideas: list[ComparedIdea] = []
    total_passes = len(passes)
    total_tasks = len(teaching_rows)

    for feature, pass_count in pass_counts.items():
        task_support = sum(
            1
            for features in teaching_rows.values()
            if features.get(feature, 0.0) >= 0.5
        )

        if pass_count >= 3 and task_support == total_tasks:
            decision = "KEEP_PARENT"
        elif pass_count >= 3 and task_support > 0:
            decision = "KEEP_SUBTYPE"
        else:
            decision = "WEAK"

        ideas.append(
            ComparedIdea(
                feature=feature,
                selected_in_passes=pass_count,
                total_passes=total_passes,
                teaching_tasks_supported=task_support,
                total_teaching_tasks=total_tasks,
                decision=decision,
            )
        )

    decision_rank = {
        "KEEP_PARENT": 0,
        "KEEP_SUBTYPE": 1,
        "WEAK": 2,
    }

    ideas.sort(
        key=lambda idea: (
            decision_rank[idea.decision],
            -idea.selected_in_passes,
            -idea.teaching_tasks_supported,
            idea.feature,
        )
    )

    return tuple(ideas)
