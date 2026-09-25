from __future__ import annotations

from dataclasses import asdict
from typing import Any

from learner.features import extract_task_rule_features
from learner.model import learn_group_concept, score_against_concept


# Eric's "nothing" group is not a solution family. It means no obvious
# human category jumped out yet, so the router treats it as UNKNOWN rather
# than learning it as a competing concept.
UNCLASSIFIED_GROUPS = {"nothing"}


def _learnable_groups(
    groups: dict[str, list[str]],
) -> dict[str, list[str]]:
    return {
        group_name: task_ids
        for group_name, task_ids in groups.items()
        if group_name not in UNCLASSIFIED_GROUPS
    }


def build_group_concepts(
    data: dict[str, Any],
    groups: dict[str, list[str]],
) -> dict[str, Any]:
    rows = {
        task_id: extract_task_rule_features(task).values
        for task_id, task in data.items()
    }

    concepts: dict[str, Any] = {}
    learnable = _learnable_groups(groups)

    for index, (group_name, task_ids) in enumerate(
        sorted(learnable.items()),
        start=1,
    ):
        positives = [
            rows[task_id]
            for task_id in task_ids
            if task_id in rows
        ]
        positive_set = {
            task_id for task_id in task_ids if task_id in rows
        }

        # "nothing" tasks are still useful as contrast examples. They simply
        # do not become a learned destination.
        negatives = [
            features
            for task_id, features in rows.items()
            if task_id not in positive_set
        ]

        if not positives:
            continue

        concept = learn_group_concept(
            group_name=group_name,
            positives=positives,
            negatives=negatives,
            concept_number=index,
        )
        concepts[group_name] = concept

    return concepts


def rank_groups_for_task(
    task: dict[str, Any],
    concepts: dict[str, Any],
    top_k: int = 3,
) -> list[dict[str, Any]]:
    features = extract_task_rule_features(task).values

    ranked = [
        {
            "group": group_name,
            "score": score_against_concept(features, concept),
        }
        for group_name, concept in concepts.items()
    ]
    ranked.sort(key=lambda item: (-item["score"], item["group"]))
    return ranked[:top_k]


def evaluate_known_groups(
    data: dict[str, Any],
    groups: dict[str, list[str]],
    top_k: int = 3,
) -> dict[str, Any]:
    """
    Evaluate routing against Eric's human labels.

    "nothing" means human-unclassified, not a rule family. Those tasks are
    reported separately and are not counted as correct/incorrect routing.

    For learnable groups with more than one member, the task being evaluated
    is removed from its own positive examples before learning that group's
    concept. Singleton groups cannot be meaningfully leave-one-out evaluated.
    """
    rows = {
        task_id: extract_task_rule_features(task).values
        for task_id, task in data.items()
    }

    task_to_group = {
        task_id: group_name
        for group_name, task_ids in groups.items()
        for task_id in task_ids
        if task_id in data
    }

    learnable = _learnable_groups(groups)

    results: list[dict[str, Any]] = []
    top1 = 0
    top3 = 0
    evaluated = 0
    singleton_skipped = 0
    unclassified_count = 0

    for task_id, expected_group in sorted(task_to_group.items()):
        if expected_group in UNCLASSIFIED_GROUPS:
            unclassified_count += 1

            concepts = build_group_concepts(data, groups)
            ranked = [
                {
                    "group": group_name,
                    "score": score_against_concept(
                        rows[task_id],
                        concept,
                    ),
                }
                for group_name, concept in concepts.items()
            ]
            ranked.sort(key=lambda item: (-item["score"], item["group"]))

            results.append(
                {
                    "task_id": task_id,
                    "expected_group": expected_group,
                    "status": "human_unclassified",
                    "predictions": ranked[:top_k],
                }
            )
            continue

        if len([x for x in learnable[expected_group] if x in data]) < 2:
            singleton_skipped += 1
            results.append(
                {
                    "task_id": task_id,
                    "expected_group": expected_group,
                    "status": "singleton_not_loo_evaluated",
                    "predictions": [],
                }
            )
            continue

        concepts: dict[str, Any] = {}

        for index, (group_name, member_ids) in enumerate(
            sorted(learnable.items()),
            start=1,
        ):
            positive_ids = [
                member_id
                for member_id in member_ids
                if member_id in rows and member_id != task_id
            ]

            if not positive_ids:
                continue

            positives = [rows[x] for x in positive_ids]
            positive_set = set(positive_ids)
            negatives = [
                feature_row
                for other_id, feature_row in rows.items()
                if other_id != task_id and other_id not in positive_set
            ]

            concepts[group_name] = learn_group_concept(
                group_name=group_name,
                positives=positives,
                negatives=negatives,
                concept_number=index,
            )

        ranked = [
            {
                "group": group_name,
                "score": score_against_concept(
                    rows[task_id],
                    concept,
                ),
            }
            for group_name, concept in concepts.items()
        ]
        ranked.sort(key=lambda item: (-item["score"], item["group"]))
        ranked = ranked[:top_k]

        predicted_groups = [item["group"] for item in ranked]
        evaluated += 1

        if predicted_groups and predicted_groups[0] == expected_group:
            top1 += 1
        if expected_group in predicted_groups:
            top3 += 1

        results.append(
            {
                "task_id": task_id,
                "expected_group": expected_group,
                "status": "evaluated",
                "predictions": ranked,
            }
        )

    return {
        "evaluated_tasks": evaluated,
        "human_unclassified_tasks": unclassified_count,
        "singleton_tasks_not_loo_evaluated": singleton_skipped,
        "top1_correct": top1,
        "top3_correct": top3,
        "top1_accuracy": top1 / evaluated if evaluated else 0.0,
        "top3_accuracy": top3 / evaluated if evaluated else 0.0,
        "results": results,
    }


def serialize_concepts(concepts: dict[str, Any]) -> dict[str, Any]:
    return {
        group_name: asdict(concept)
        for group_name, concept in concepts.items()
    }
