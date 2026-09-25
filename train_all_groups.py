from __future__ import annotations

import json
from pathlib import Path

from learner.router import (
    build_group_concepts,
    evaluate_known_groups,
    serialize_concepts,
)


def load_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, indent=2)


def main() -> None:
    data_path = Path("data") / "data.json"
    groups_path = Path("data") / "task_groups.json"

    if not data_path.exists():
        raise SystemExit(f"Training data not found: {data_path}")
    if not groups_path.exists():
        raise SystemExit(f"Task groups not found: {groups_path}")

    data = load_json(data_path)
    groups = load_json(groups_path)

    concepts = build_group_concepts(data, groups)
    evaluation = evaluate_known_groups(
        data=data,
        groups=groups,
        top_k=3,
    )

    output_dir = Path("learned") / "group_router"
    output_dir.mkdir(parents=True, exist_ok=True)

    router_path = output_dir / "group_concepts.json"
    eval_path = output_dir / "evaluation.json"

    save_json(
        router_path,
        {
            "purpose": (
                "Use broad learned group knowledge as a routing prior. "
                "For a new ARC task, try rules from the highest-ranked "
                "groups first, but fall back to other groups and then "
                "general search if exact training reconstruction fails."
            ),
            "groups": serialize_concepts(concepts),
        },
    )
    save_json(eval_path, evaluation)

    print("=" * 72)
    print("FULL TASK-GROUP ROUTER EXPERIMENT")
    print("=" * 72)
    print(f"Human groups learned: {len(concepts)}")
    print(f"Tasks LOO-evaluated: {evaluation['evaluated_tasks']}")
    print(
        "Singleton tasks not LOO-evaluated: "
        f"{evaluation['singleton_tasks_not_loo_evaluated']}"
    )
    print()
    print(
        f"Top-1 human-group match: "
        f"{evaluation['top1_correct']}/{evaluation['evaluated_tasks']} "
        f"({evaluation['top1_accuracy']:.1%})"
    )
    print(
        f"Top-3 human-group match: "
        f"{evaluation['top3_correct']}/{evaluation['evaluated_tasks']} "
        f"({evaluation['top3_accuracy']:.1%})"
    )

    print()
    print("TASK ROUTING RESULTS")
    print("Expected group -> learner's top 3 guesses")
    print()

    for result in evaluation["results"]:
        if result["status"] != "evaluated":
            print(
                f"{result['task_id']} | {result['expected_group']} "
                "| singleton: no independent LOO test"
            )
            continue

        guesses = " | ".join(
            f"{item['group']} {item['score']:.3f}"
            for item in result["predictions"]
        )
        marker = (
            "OK"
            if result["expected_group"]
            in [item["group"] for item in result["predictions"]]
            else "MISS"
        )
        print(
            f"{result['task_id']} | {result['expected_group']} "
            f"| {guesses} | {marker}"
        )

    print()
    print(f"Saved router knowledge to: {router_path}")
    print(f"Saved evaluation to: {eval_path}")
    print()
    print(
        "This is a ROUTER, not a final solver. Later, a task can use the "
        "top-ranked group as the first rule toolbox, try the next group if "
        "exact reconstruction fails, and only then fall back to wider search."
    )


if __name__ == "__main__":
    main()
