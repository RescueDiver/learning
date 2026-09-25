from __future__ import annotations

from dataclasses import asdict
from typing import Any

from learner.features import (
    extract_task_router_features,
    extract_task_rule_features,
)
from learner.model import learn_group_concept, score_against_concept


# Eric's "nothing" group is not a solution family. It means no obvious
# human category jumped out yet, so the router treats it as UNKNOWN rather
# than learning it as a competing concept.
UNCLASSIFIED_GROUPS = {"nothing"}

def _specialist_filter(features: dict[str, float]) -> dict[str, float]:
    # Keep the full vocabulary. The previous hard filter hurt routing badly,
    # so surface clues are allowed again and will be controlled by blending
    # coarse and specialist scores instead of deleting information.
    return dict(features)


def _learnable_groups(
    groups: dict[str, list[str]],
) -> dict[str, list[str]]:
    return {
        group_name: task_ids
        for group_name, task_ids in groups.items()
        if group_name not in UNCLASSIFIED_GROUPS
    }


def _feature_rows(data: dict[str, Any]) -> tuple[
    dict[str, dict[str, float]],
    dict[str, dict[str, float]],
]:
    coarse_rows = {
        task_id: extract_task_rule_features(task).values
        for task_id, task in data.items()
    }
    specialist_rows = {
        task_id: _specialist_filter(
            extract_task_router_features(task).values
        )
        for task_id, task in data.items()
    }
    return coarse_rows, specialist_rows


def _learn_concepts_for_rows(
    rows: dict[str, dict[str, float]],
    groups: dict[str, list[str]],
    strongest_count: int,
    excluded_task_id: str | None = None,
) -> dict[str, Any]:
    concepts: dict[str, Any] = {}
    learnable = _learnable_groups(groups)

    for index, (group_name, task_ids) in enumerate(
        sorted(learnable.items()),
        start=1,
    ):
        positive_ids = [
            task_id
            for task_id in task_ids
            if task_id in rows and task_id != excluded_task_id
        ]
        if not positive_ids:
            continue

        positives = [rows[task_id] for task_id in positive_ids]
        positive_set = set(positive_ids)
        negatives = [
            features
            for task_id, features in rows.items()
            if task_id != excluded_task_id and task_id not in positive_set
        ]

        concepts[group_name] = learn_group_concept(
            group_name=group_name,
            positives=positives,
            negatives=negatives,
            concept_number=index,
            strongest_count=strongest_count,
        )

    return concepts


def build_group_concepts(
    data: dict[str, Any],
    groups: dict[str, list[str]],
) -> dict[str, Any]:
    """
    Build both routing layers.

    coarse:
        small rule-level vocabulary used only to narrow the neighborhood.

    specialist:
        richer vocabulary. Each group learns its own strongest questions.
        Surface clues remain available, but final routing blends this layer
        with the coarse structural score so neither layer can dominate alone.
    """
    coarse_rows, specialist_rows = _feature_rows(data)

    return {
        "coarse": _learn_concepts_for_rows(
            coarse_rows,
            groups,
            strongest_count=8,
        ),
        "specialist": _learn_concepts_for_rows(
            specialist_rows,
            groups,
            strongest_count=12,
        ),
    }


def _rank(
    features: dict[str, float],
    concepts: dict[str, Any],
) -> list[dict[str, Any]]:
    ranked = [
        {
            "group": group_name,
            "score": score_against_concept(features, concept),
        }
        for group_name, concept in concepts.items()
    ]
    ranked.sort(key=lambda item: (-item["score"], item["group"]))
    return ranked



def _blend_ranked_candidates(
    coarse_ranked: list[dict[str, Any]],
    specialist_ranked: list[dict[str, Any]],
    specialist_concepts: dict[str, Any],
    coarse_weight: float,
) -> list[dict[str, Any]]:
    coarse_scores = {
        item["group"]: item["score"]
        for item in coarse_ranked
    }
    specialist_weight = 1.0 - coarse_weight

    blended = []
    for item in specialist_ranked:
        group_name = item["group"]
        coarse_score = coarse_scores.get(group_name, 0.0)
        specialist_score = item["score"]
        final_score = (
            coarse_weight * coarse_score
            + specialist_weight * specialist_score
        )
        blended.append(
            {
                "group": group_name,
                "coarse_score": coarse_score,
                "specialist_score": specialist_score,
                "score": final_score,
                "questions": list(
                    specialist_concepts[group_name].strongest_features
                ),
            }
        )

    blended.sort(key=lambda item: (-item["score"], item["group"]))
    return blended


def rank_groups_for_task(
    task: dict[str, Any],
    concepts: dict[str, Any],
    top_k: int = 3,
    coarse_k: int = 6,
    coarse_weight: float = 0.5,
) -> list[dict[str, Any]]:
    """
    Two-stage route:
      1. generic rule features narrow the neighborhood;
      2. candidate groups use their own learned specialist vocabularies.
    """
    coarse_features = extract_task_rule_features(task).values
    specialist_features = _specialist_filter(
        extract_task_router_features(task).values
    )

    coarse_ranked = _rank(
        coarse_features,
        concepts["coarse"],
    )
    candidate_names = {
        item["group"]
        for item in coarse_ranked[:coarse_k]
    }

    specialist_candidates = {
        group_name: concept
        for group_name, concept in concepts["specialist"].items()
        if group_name in candidate_names
    }
    specialist_ranked = _rank(
        specialist_features,
        specialist_candidates,
    )

    blended = _blend_ranked_candidates(
        coarse_ranked=coarse_ranked,
        specialist_ranked=specialist_ranked,
        specialist_concepts=concepts["specialist"],
        coarse_weight=coarse_weight,
    )
    return blended[:top_k]


def evaluate_known_groups(
    data: dict[str, Any],
    groups: dict[str, list[str]],
    top_k: int = 3,
    coarse_k: int = 6,
    coarse_weights: tuple[float, ...] = (0.25, 0.5, 0.75),
) -> dict[str, Any]:
    coarse_rows, specialist_rows = _feature_rows(data)

    task_to_group = {
        task_id: group_name
        for group_name, task_ids in groups.items()
        for task_id in task_ids
        if task_id in data
    }

    learnable = _learnable_groups(groups)

    results_by_weight: dict[str, list[dict[str, Any]]] = {
        f"{weight:.2f}": [] for weight in coarse_weights
    }
    stats = {
        f"{weight:.2f}": {"top1": 0, "top3": 0}
        for weight in coarse_weights
    }
    coarse_hit = 0
    evaluated = 0
    singleton_skipped = 0
    unclassified_count = 0

    for task_id, expected_group in sorted(task_to_group.items()):
        if expected_group in UNCLASSIFIED_GROUPS:
            unclassified_count += 1
            full_concepts = build_group_concepts(data, groups)
            for weight in coarse_weights:
                key = f"{weight:.2f}"
                ranked = rank_groups_for_task(
                    data[task_id],
                    full_concepts,
                    top_k=top_k,
                    coarse_k=coarse_k,
                    coarse_weight=weight,
                )
                results_by_weight[key].append(
                    {
                        "task_id": task_id,
                        "expected_group": expected_group,
                        "status": "human_unclassified",
                        "predictions": ranked,
                    }
                )
            continue

        if len([x for x in learnable[expected_group] if x in data]) < 2:
            singleton_skipped += 1
            for weight in coarse_weights:
                key = f"{weight:.2f}"
                results_by_weight[key].append(
                    {
                        "task_id": task_id,
                        "expected_group": expected_group,
                        "status": "singleton_not_loo_evaluated",
                        "predictions": [],
                    }
                )
            continue

        coarse_concepts = _learn_concepts_for_rows(
            coarse_rows,
            groups,
            strongest_count=8,
            excluded_task_id=task_id,
        )
        specialist_concepts = _learn_concepts_for_rows(
            specialist_rows,
            groups,
            strongest_count=12,
            excluded_task_id=task_id,
        )

        coarse_ranked = _rank(
            coarse_rows[task_id],
            coarse_concepts,
        )
        coarse_candidates = coarse_ranked[:coarse_k]
        candidate_names = {
            item["group"]
            for item in coarse_candidates
        }

        evaluated += 1
        if expected_group in candidate_names:
            coarse_hit += 1

        specialist_candidates = {
            group_name: concept
            for group_name, concept in specialist_concepts.items()
            if group_name in candidate_names
        }
        specialist_ranked = _rank(
            specialist_rows[task_id],
            specialist_candidates,
        )

        for weight in coarse_weights:
            key = f"{weight:.2f}"
            blended = _blend_ranked_candidates(
                coarse_ranked=coarse_ranked,
                specialist_ranked=specialist_ranked,
                specialist_concepts=specialist_concepts,
                coarse_weight=weight,
            )[:top_k]

            predicted_groups = [item["group"] for item in blended]
            if predicted_groups and predicted_groups[0] == expected_group:
                stats[key]["top1"] += 1
            if expected_group in predicted_groups:
                stats[key]["top3"] += 1

            results_by_weight[key].append(
                {
                    "task_id": task_id,
                    "expected_group": expected_group,
                    "status": "evaluated",
                    "coarse_candidates": coarse_candidates,
                    "predictions": blended,
                }
            )

    experiments = {}
    for weight in coarse_weights:
        key = f"{weight:.2f}"
        top1 = stats[key]["top1"]
        top3 = stats[key]["top3"]
        experiments[key] = {
            "coarse_weight": weight,
            "specialist_weight": 1.0 - weight,
            "top1_correct": top1,
            "top3_correct": top3,
            "top1_accuracy": top1 / evaluated if evaluated else 0.0,
            "top3_accuracy": top3 / evaluated if evaluated else 0.0,
            "results": results_by_weight[key],
        }

    best_key = max(
        experiments,
        key=lambda key: (
            experiments[key]["top3_accuracy"],
            experiments[key]["top1_accuracy"],
        ),
    )

    return {
        "evaluated_tasks": evaluated,
        "human_unclassified_tasks": unclassified_count,
        "singleton_tasks_not_loo_evaluated": singleton_skipped,
        "coarse_candidate_size": coarse_k,
        "coarse_hit_count": coarse_hit,
        "coarse_hit_accuracy": coarse_hit / evaluated if evaluated else 0.0,
        "experiments": experiments,
        "best_weight_key": best_key,
        "best": experiments[best_key],
    }


def serialize_concepts(concepts: dict[str, Any]) -> dict[str, Any]:
    return {
        layer_name: {
            group_name: asdict(concept)
            for group_name, concept in layer.items()
        }
        for layer_name, layer in concepts.items()
    }
