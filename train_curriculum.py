from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from learner.features import extract_task_rule_features
from learner.multiview import run_multiview_passes, surviving_ideas
from learner.refinement import (
    best_subconcept_match,
    discover_subconcepts,
    shared_parent_features,
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Teach an ARC group from several ordered viewpoints and keep "
            "the ideas that survive the different passes."
        )
    )
    parser.add_argument(
        "--group",
        default="missing mask",
        help="Human-created task group to learn.",
    )
    args = parser.parse_args()

    data_path = Path("data") / "data.json"
    groups_path = Path("data") / "task_groups.json"

    if not data_path.exists():
        raise SystemExit(f"Training data not found: {data_path}")
    if not groups_path.exists():
        raise SystemExit(f"Task groups not found: {groups_path}")

    data = load_json(data_path)
    groups = load_json(groups_path)

    if args.group not in groups:
        available = ", ".join(sorted(groups))
        raise SystemExit(
            f"Group {args.group!r} not found. Available groups: {available}"
        )

    positive_ids = [
        task_id
        for task_id in groups[args.group]
        if task_id in data
    ]
    other_ids = sorted({
        task_id
        for group_name, task_ids in groups.items()
        if group_name != args.group
        for task_id in task_ids
        if task_id in data
    })

    if not positive_ids:
        raise SystemExit("Selected group has no usable tasks.")

    positive_rows = {
        task_id: extract_task_rule_features(data[task_id]).values
        for task_id in positive_ids
    }
    negative_rows = {
        task_id: extract_task_rule_features(data[task_id]).values
        for task_id in other_ids
    }

    parent_features = shared_parent_features(positive_rows)
    subconcepts = discover_subconcepts(positive_rows)

    passes = run_multiview_passes(
        positives=list(positive_rows.values()),
        negatives=list(negative_rows.values()),
        features_per_pass=5,
    )
    survivors = surviving_ideas(
        passes,
        minimum_passes=3,
    )

    print("=" * 72)
    print("CURRICULUM LEARNING EXPERIMENT - STAGE 5")
    print("=" * 72)
    print(f"Human group: {args.group}")
    print(f"Teaching tasks: {len(positive_rows)}")
    print(f"Contrast tasks: {len(negative_rows)}")
    print()

    print("Same lesson, different viewing orders:")
    for result in passes:
        print()
        print(
            f"  {result.name}: "
            f"{' -> '.join(result.ordered_categories)}"
        )
        for feature in result.selected_features:
            print(f"    {feature}")

    print()
    print("Ideas that survived different viewing orders:")
    if survivors:
        for feature, count in survivors:
            print(
                f"  {feature:58s} survived {count}/{len(passes)} passes"
            )
    else:
        print("  none survived the current threshold")

    print()
    print("Shared parent concept from ALL teaching tasks:")
    if parent_features:
        for feature in parent_features:
            print(f"  {feature}")
    else:
        print("  (no all-pair invariant shared by every teaching task)")

    print()
    print("Stage 4 sub-concepts, kept for comparison:")
    for concept in subconcepts:
        print(
            f"  {concept.concept_id}: "
            f"{', '.join(concept.task_ids)}"
        )

    print()
    print("Teaching-task matches:")
    for task_id, features in positive_rows.items():
        concept_id, score = best_subconcept_match(
            features,
            subconcepts,
        )
        print(f"  {task_id} -> {concept_id}  score={score:.4f}")

    print()
    print("Closest outside tasks:")
    outside_matches = []

    for task_id, features in negative_rows.items():
        concept_id, score = best_subconcept_match(
            features,
            subconcepts,
        )
        outside_matches.append((score, task_id, concept_id))

    outside_matches.sort(reverse=True)

    for score, task_id, concept_id in outside_matches[:20]:
        print(f"  {score:.4f}  {task_id} -> {concept_id}")

    output_dir = Path("learned")
    output_dir.mkdir(exist_ok=True)

    safe_name = args.group.replace(" ", "_")
    output_path = output_dir / f"{safe_name}.json"

    payload = {
        "stage": "multi_view_survival",
        "human_group": args.group,
        "teaching_task_ids": positive_ids,
        "view_passes": [asdict(result) for result in passes],
        "surviving_ideas": [
            {
                "feature": feature,
                "survived_passes": count,
                "total_passes": len(passes),
            }
            for feature, count in survivors
        ],
        "parent_required_features": list(parent_features),
        "subconcepts": [asdict(concept) for concept in subconcepts],
        "teaching_rule_features": positive_rows,
        "outside_matches": [
            {
                "task_id": task_id,
                "subconcept_id": concept_id,
                "score": score,
            }
            for score, task_id, concept_id in outside_matches
        ],
        "next_stage": (
            "add finer changed-cell and reconstruction features, then rerun "
            "the same multi-view survival test"
        ),
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    print()
    print(f"Saved multi-view concept to: {output_path}")
    print()
    print(
        "Stage 5 changes the order of attention without changing the "
        "teaching tasks. Ideas that recur across different orders are "
        "treated as more trustworthy than one-pass clues."
    )


if __name__ == "__main__":
    main()
