from __future__ import annotations

import argparse
import json
from dataclasses import asdict
from pathlib import Path

from learner.features import extract_task_features
from learner.model import learn_group_concept, score_against_concept


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Teach a broad ARC group before refining its tasks."
    )
    parser.add_argument(
        "--group",
        default="missing mask",
        help="Human-created task group to learn.",
    )
    parser.add_argument(
        "--arcs6",
        type=Path,
        default=Path("..") / "ARCS6",
        help="Path to the ARCS6 repository.",
    )
    args = parser.parse_args()

    data_path = args.arcs6 / "data" / "data.json"
    groups_path = args.arcs6 / "task_groups.json"

    if not data_path.exists():
        raise SystemExit(f"ARCS6 data not found: {data_path}")
    if not groups_path.exists():
        raise SystemExit(f"Task groups not found: {groups_path}")

    data = load_json(data_path)
    groups = load_json(groups_path)

    if args.group not in groups:
        available = ", ".join(sorted(groups))
        raise SystemExit(
            f"Group {args.group!r} not found. Available groups: {available}"
        )

    positive_ids = [task_id for task_id in groups[args.group] if task_id in data]
    other_group_ids = {
        task_id
        for group_name, task_ids in groups.items()
        if group_name != args.group
        for task_id in task_ids
        if task_id in data
    }

    if not positive_ids:
        raise SystemExit("Selected group has no usable tasks.")

    positive_rows = {
        task_id: extract_task_features(data[task_id]).values
        for task_id in positive_ids
    }
    negative_rows = {
        task_id: extract_task_features(data[task_id]).values
        for task_id in sorted(other_group_ids)
    }

    concept = learn_group_concept(
        group_name=args.group,
        positives=list(positive_rows.values()),
        negatives=list(negative_rows.values()),
        concept_number=1,
    )

    print("=" * 72)
    print("CURRICULUM LEARNING EXPERIMENT")
    print("=" * 72)
    print(f"Human group: {args.group}")
    print(f"Positive teaching tasks: {len(positive_rows)}")
    print(f"Contrast tasks: {len(negative_rows)}")
    print()
    print(f"Internal concept: {concept.concept_id}")
    print("Strongest learned distinctions:")

    for key in concept.strongest_features:
        value = concept.contrast[key]
        direction = "higher" if value > 0 else "lower"
        print(f"  {key:40s} {direction:6s} strength={abs(value):.3f}")

    print()
    print("How well the learner matches Eric's sorting:")

    ranked = []
    all_rows = {**positive_rows, **negative_rows}

    for task_id, features in all_rows.items():
        score = score_against_concept(features, concept)
        expected = task_id in positive_rows
        ranked.append((score, task_id, expected))

    ranked.sort(reverse=True)

    for score, task_id, expected in ranked[:20]:
        marker = "TEACH" if expected else "other"
        print(f"  {score:0.4f}  {marker:5s}  {task_id}")

    output_dir = Path("learned")
    output_dir.mkdir(exist_ok=True)

    safe_name = args.group.replace(" ", "_")
    output_path = output_dir / f"{safe_name}.json"

    payload = {
        "stage": "broad_group_concept",
        "human_group": args.group,
        "teaching_task_ids": positive_ids,
        "concept": asdict(concept),
        "task_scores": {
            task_id: score_against_concept(features, concept)
            for task_id, features in all_rows.items()
        },
        "next_stage": (
            "refine the broad concept by comparing each teaching task's "
            "training input/output transformations"
        ),
    }

    with output_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)

    print()
    print(f"Saved learned concept to: {output_path}")
    print()
    print(
        "Stage 1 teaches HOW Eric sorted the tasks. "
        "It does not yet teach the exact repair rule."
    )


if __name__ == "__main__":
    main()
