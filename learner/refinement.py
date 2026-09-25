from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class SubConcept:
    concept_id: str
    task_ids: tuple[str, ...]
    required_features: tuple[str, ...]
    distinguishing_features: tuple[str, ...]


def active_all_features(features: dict[str, float]) -> set[str]:
    return {
        key
        for key, value in features.items()
        if key.startswith("all_") and value >= 0.5
    }


def shared_parent_features(
    teaching_rows: dict[str, dict[str, float]],
) -> tuple[str, ...]:
    if not teaching_rows:
        return ()

    sets = [
        active_all_features(features)
        for features in teaching_rows.values()
    ]

    shared = set(sets[0])
    for feature_set in sets[1:]:
        shared.intersection_update(feature_set)

    return tuple(sorted(shared))


def discover_subconcepts(
    teaching_rows: dict[str, dict[str, float]],
) -> tuple[SubConcept, ...]:
    """
    Split a human-created parent group by the rule invariants each teaching
    task actually exhibits.

    This does not use task IDs as rules. IDs are only attached afterward so
    Eric can see which teaching examples landed in each learned subtype.
    """
    parent = set(shared_parent_features(teaching_rows))

    buckets: dict[tuple[str, ...], list[str]] = {}

    for task_id, features in teaching_rows.items():
        active = active_all_features(features)
        distinguishing = tuple(sorted(active - parent))
        buckets.setdefault(distinguishing, []).append(task_id)

    subconcepts: list[SubConcept] = []

    for index, (distinguishing, task_ids) in enumerate(
        sorted(
            buckets.items(),
            key=lambda item: (item[0], tuple(sorted(item[1]))),
        ),
        start=1,
    ):
        required = tuple(sorted(parent | set(distinguishing)))
        subconcepts.append(
            SubConcept(
                concept_id=f"concept_004_{index:02d}",
                task_ids=tuple(sorted(task_ids)),
                required_features=required,
                distinguishing_features=distinguishing,
            )
        )

    return tuple(subconcepts)


def signature_similarity(
    features: dict[str, float],
    required_features: tuple[str, ...],
) -> float:
    """
    Exact-rule signature similarity.

    Reward required invariants being present, while penalizing extra
    all_* invariants so unrelated tasks do not get a free pass merely for
    containing a rectangle.
    """
    active = active_all_features(features)
    required = set(required_features)

    if not required and not active:
        return 1.0
    if not required:
        return 0.0

    intersection = len(active & required)
    union = len(active | required)

    if union == 0:
        return 0.0

    return intersection / union


def best_subconcept_match(
    features: dict[str, float],
    subconcepts: tuple[SubConcept, ...],
) -> tuple[str | None, float]:
    best_id: str | None = None
    best_score = -1.0

    for concept in subconcepts:
        score = signature_similarity(
            features,
            concept.required_features,
        )

        if score > best_score:
            best_score = score
            best_id = concept.concept_id

    return best_id, max(best_score, 0.0)
