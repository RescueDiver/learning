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
    print(f"Learnable human groups: {len(concepts['specialist'])}")
    print('"nothing" treated as UNKNOWN, not a learned rule family')
    print(f"Tasks LOO-evaluated: {evaluation['evaluated_tasks']}")
    print(
        "Human-unclassified ('nothing') tasks: "
        f"{evaluation['human_unclassified_tasks']}"
    )
    print(
        "Singleton tasks not LOO-evaluated: "
        f"{evaluation['singleton_tasks_not_loo_evaluated']}"
    )
    print()
    print(
        f"Coarse neighborhood hit (top {evaluation['coarse_candidate_size']}): "
        f"{evaluation['coarse_hit_count']}/{evaluation['evaluated_tasks']} "
        f"({evaluation['coarse_hit_accuracy']:.1%})"
    )
    print(
        f"Top-1 human-group match after specialist rerank: "
        f"{evaluation['top1_correct']}/{evaluation['evaluated_tasks']} "
        f"({evaluation['top1_accuracy']:.1%})"
    )
    print(
        f"Top-3 human-group match after specialist rerank: "
        f"{evaluation['top3_correct']}/{evaluation['evaluated_tasks']} "
        f"({evaluation['top3_accuracy']:.1%})"
    )

    print()
    print("TASK ROUTING RESULTS")
    print("Expected group -> learner's top 3 guesses after structural specialist questions")
    print()

    for result in evaluation["results"]:
        if result["status"] == "human_unclassified":
            guesses = " | ".join(
                f"{item['group']} {item['score']:.3f}"
                for item in result["predictions"]
            )
            print(
                f"{result['task_id']} | UNKNOWN | {guesses} "
                "| diagnostic only"
            )
            continue

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
        "\"nothing\" is now an UNKNOWN/fallback state, not a rule family. "
        "Routing now happens in two stages: generic features narrow the "
        "neighborhood, then each candidate group uses its own learned "
        "structural vocabulary to rerank the candidates. Raw dimensions, "
        "areas, color counts, and train-pair count are blocked from becoming "
        "specialist-defining questions. This is still a router, not a final solver."
    )


if __name__ == "__main__":
    main()
