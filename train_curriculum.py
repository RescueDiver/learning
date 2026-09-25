from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from learner.features import extract_task_rule_features
from learner.multiview import (
    VIEW_ORDERS,
    compare_view_results,
    run_single_view,
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run the same ARC lesson in different orders, save every pass, "
            "then compare the saved results before choosing concepts."
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

    safe_name = args.group.replace(" ", "_")
    experiment_dir = Path("learned") / safe_name / "stage_6"
    experiment_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 72)
    print("CURRICULUM LEARNING EXPERIMENT - STAGE 6")
    print("=" * 72)
    print(f"Human group: {args.group}")
    print(f"Teaching tasks: {len(positive_rows)}")
    print(f"Contrast tasks: {len(negative_rows)}")
    print()

    saved_passes = []

    # Deliberately run one view, save it, then move to the next.
    for pass_number, (view_name, _) in enumerate(VIEW_ORDERS, start=1):
        print(f"PASS {pass_number}: {view_name}")

        result = run_single_view(
            view_name=view_name,
            positives=list(positive_rows.values()),
            negatives=list(negative_rows.values()),
            features_per_pass=5,
        )

        pass_path = experiment_dir / (
            f"pass_{pass_number:02d}_{view_name}.json"
        )

        pass_payload = {
            "stage": "single_view_pass",
            "pass_number": pass_number,
            "human_group": args.group,
            "view": asdict(result),
            "teaching_task_ids": positive_ids,
        }

        save_json(pass_path, pass_payload)
        saved_passes.append(result)

        print(
            f"  order: {' -> '.join(result.ordered_categories)}"
        )
        for feature in result.selected_features:
            print(f"  picked: {feature}")
        print(f"  SAVED: {pass_path}")
        print()

    comparison = compare_view_results(
        passes=tuple(saved_passes),
        teaching_rows=positive_rows,
    )

    print("=" * 72)
    print("COMPARE THE SAVED RUNS")
    print("=" * 72)

    for idea in comparison:
        print(
            f"{idea.decision:12s} "
            f"{idea.feature:58s} "
            f"runs={idea.selected_in_passes}/{idea.total_passes} "
            f"tasks={idea.teaching_tasks_supported}/{idea.total_teaching_tasks}"
        )

    parent_choices = [
        idea.feature
        for idea in comparison
        if idea.decision == "KEEP_PARENT"
    ]
    subtype_choices = [
        idea.feature
        for idea in comparison
        if idea.decision == "KEEP_SUBTYPE"
    ]
    weak_choices = [
        idea.feature
        for idea in comparison
        if idea.decision == "WEAK"
    ]

    comparison_path = experiment_dir / "comparison.json"
    save_json(
        comparison_path,
        {
            "stage": "saved_pass_comparison",
            "human_group": args.group,
            "passes": [asdict(result) for result in saved_passes],
            "compared_ideas": [asdict(idea) for idea in comparison],
        },
    )

    final_path = experiment_dir / "final_choice.json"
    save_json(
        final_path,
        {
            "stage": "consensus_choice",
            "human_group": args.group,
            "parent_concepts": parent_choices,
            "subtype_concepts": subtype_choices,
            "weak_clues": weak_choices,
            "rule": {
                "parent": (
                    "selected in at least 3 viewing orders and supported by "
                    "every teaching task"
                ),
                "subtype": (
                    "selected in at least 3 viewing orders but supported by "
                    "only part of the teaching group"
                ),
                "weak": (
                    "did not survive enough independently ordered runs"
                ),
            },
        },
    )

    print()
    print("FINAL CHOICE")
    print("Parent concepts:")
    if parent_choices:
        for feature in parent_choices:
            print(f"  KEEP  {feature}")
    else:
        print("  none")

    print()
    print("Subtype concepts:")
    if subtype_choices:
        for feature in subtype_choices:
            print(f"  KEEP  {feature}")
    else:
        print("  none")

    print()
    print("Weak clues:")
    if weak_choices:
        for feature in weak_choices:
            print(f"  DROP FOR NOW  {feature}")
    else:
        print("  none")

    print()
    print(f"Saved comparison to: {comparison_path}")
    print(f"Saved final choice to: {final_path}")
    print()
    print(
        "Stage 6 does not compare temporary rankings in memory and forget "
        "them. It saves each run first, then compares those saved sets, "
        "then makes the final keep/subtype/weak decision."
    )


if __name__ == "__main__":
    main()
