from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

from new.general_scene import task_signature


DEFAULT_PER_GROUP = 2
DEFAULT_LIMIT = 24


def load_json(path: Path) -> Any:
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def choose_mixed_tasks(
    data: dict[str, Any],
    groups: dict[str, list[str]],
    per_group: int,
    limit: int,
) -> list[tuple[str, str]]:
    chosen: list[tuple[str, str]] = []
    used: set[str] = set()

    # Round-robin across human groups so the benchmark does not accidentally
    # become another single-family exercise.
    for round_index in range(per_group):
        for group_name, task_ids in groups.items():
            candidates = [
                task_id
                for task_id in task_ids
                if task_id in data
            ]
            if round_index >= len(candidates):
                continue

            task_id = candidates[round_index]
            if task_id in used:
                continue

            chosen.append((group_name, task_id))
            used.add(task_id)

            if len(chosen) >= limit:
                return chosen

    # Fill any remaining slots with tasks that are present in data but were
    # not part of the manual groups.
    for task_id in sorted(data):
        if task_id in used:
            continue
        chosen.append(("ungrouped", task_id))
        used.add(task_id)
        if len(chosen) >= limit:
            break

    return chosen


def jaccard(a: set[str], b: set[str]) -> float:
    union = a | b
    if not union:
        return 1.0
    return len(a & b) / len(union)


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Benchmark the general structural language across a deliberately "
            "mixed set of ARC tasks."
        )
    )
    parser.add_argument(
        "--data",
        required=True,
        help="Path to ARCS6 data/data.json",
    )
    parser.add_argument(
        "--groups",
        required=True,
        help="Path to ARCS6 data/task_groups.json",
    )
    parser.add_argument(
        "--per-group",
        type=int,
        default=DEFAULT_PER_GROUP,
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=DEFAULT_LIMIT,
    )
    parser.add_argument(
        "--report",
        default="general_scene_report.json",
    )
    args = parser.parse_args()

    data = load_json(Path(args.data))
    groups = load_json(Path(args.groups))

    chosen = choose_mixed_tasks(
        data,
        groups,
        per_group=max(1, args.per_group),
        limit=max(1, args.limit),
    )

    print("=" * 88)
    print("GENERAL STRUCTURAL LANGUAGE BENCHMARK")
    print("=" * 88)
    print(f"Tasks selected: {len(chosen)}")
    print(
        "Purpose: test whether one scene/relationship vocabulary says useful "
        "things across unrelated ARC task families."
    )
    print()

    records: list[dict[str, Any]] = []
    token_frequency: Counter[str] = Counter()
    group_tokens: dict[str, Counter[str]] = defaultdict(Counter)

    for index, (group_name, task_id) in enumerate(chosen, start=1):
        task = data[task_id]
        signature = task_signature(task.get("train", []))
        invariant = set(signature["invariant_tokens"])
        any_tokens = set(signature["any_tokens"])

        token_frequency.update(invariant)
        group_tokens[group_name].update(invariant)

        record = {
            "task_id": task_id,
            "group": group_name,
            "signature": signature,
        }
        records.append(record)

        facts = signature["structural_facts"]
        print(
            f"[{index:>2}/{len(chosen)}] {task_id} | "
            f"group={group_name} | "
            f"pairs={signature['pair_count']} | "
            f"invariant_tokens={len(invariant)}"
        )
        print(
            "    objects in->out: "
            f"{facts.get('input_object_counts')} -> "
            f"{facts.get('output_object_counts')} | "
            f"matches={facts.get('match_counts')}"
        )

        if invariant:
            print("    invariant: " + ", ".join(sorted(invariant)))
        else:
            print("    invariant: NONE")

        pair_details = signature.get("pair_details", [])
        if pair_details:
            concise = [
                "; ".join(details)
                for details in pair_details
                if details
            ]
            if concise:
                print("    details: " + " || ".join(concise))
        print()

    print("=" * 88)
    print("VOCABULARY COVERAGE")
    print("=" * 88)

    covered = sum(
        1
        for record in records
        if record["signature"]["invariant_tokens"]
    )
    print(
        f"Tasks with at least one invariant structural token: "
        f"{covered}/{len(records)}"
    )

    if token_frequency:
        print()
        print("Most common invariant tokens:")
        for token, count in token_frequency.most_common(20):
            print(f"  {count:>2}/{len(records)}  {token}")

    print()
    print("=" * 88)
    print("DISCRIMINATION CHECK")
    print("=" * 88)
    print(
        "For each task, nearest neighbor is chosen ONLY from invariant "
        "structural tokens. Human group is shown afterward as a diagnostic."
    )

    same_group = 0
    comparable = 0
    nearest_rows: list[dict[str, Any]] = []

    for record in records:
        task_tokens = set(record["signature"]["invariant_tokens"])
        candidates: list[tuple[float, str, str]] = []

        for other in records:
            if other["task_id"] == record["task_id"]:
                continue
            other_tokens = set(other["signature"]["invariant_tokens"])
            score = jaccard(task_tokens, other_tokens)
            candidates.append(
                (score, other["task_id"], other["group"])
            )

        if not candidates:
            continue

        candidates.sort(key=lambda item: (-item[0], item[1]))
        score, neighbor_id, neighbor_group = candidates[0]
        match = neighbor_group == record["group"]

        comparable += 1
        same_group += int(match)
        nearest_rows.append(
            {
                "task_id": record["task_id"],
                "group": record["group"],
                "nearest_task": neighbor_id,
                "nearest_group": neighbor_group,
                "jaccard": score,
                "same_human_group": match,
            }
        )

        print(
            f"  {record['task_id']} ({record['group']}) -> "
            f"{neighbor_id} ({neighbor_group}) "
            f"J={score:.3f} "
            f"{'SAME-GROUP' if match else 'different-group'}"
        )

    if comparable:
        print()
        print(
            "Nearest-neighbor same human group: "
            f"{same_group}/{comparable} "
            f"({100.0 * same_group / comparable:.1f}%)"
        )

    print()
    print("=" * 88)
    print("GROUP-SPECIFIC STRUCTURAL TOKENS")
    print("=" * 88)

    for group_name in sorted(group_tokens):
        members = sum(1 for record in records if record["group"] == group_name)
        if members == 0:
            continue

        own = group_tokens[group_name]
        other = Counter()
        for other_group, counts in group_tokens.items():
            if other_group != group_name:
                other.update(counts)

        distinctive = [
            (token, count, other[token])
            for token, count in own.items()
            if count >= max(1, members // 2)
        ]
        distinctive.sort(
            key=lambda item: (
                item[2],
                -item[1],
                item[0],
            )
        )

        print(f"{group_name} ({members} sampled):")
        if not distinctive:
            print("    no stable distinguishing token yet")
        else:
            for token, own_count, other_count in distinctive[:8]:
                print(
                    f"    {token}: own={own_count}/{members}, "
                    f"other={other_count}"
                )

    report = {
        "task_count": len(records),
        "selection": [
            {"task_id": task_id, "group": group}
            for group, task_id in chosen
        ],
        "records": records,
        "nearest_neighbors": nearest_rows,
        "token_frequency": dict(token_frequency),
        "same_group_nearest": {
            "count": same_group,
            "total": comparable,
            "fraction": (
                same_group / comparable
                if comparable
                else None
            ),
        },
    }

    report_path = Path(args.report)
    with report_path.open("w", encoding="utf-8") as file:
        json.dump(report, file, indent=2)

    print()
    print("=" * 88)
    print("IMPORTANT INTERPRETATION")
    print("=" * 88)
    print(
        "This benchmark does NOT try to solve the tasks. It tests whether the "
        "shared structural language is rich enough to describe many different "
        "kinds of task without task-ID logic. Weak coverage or near-random "
        "group discrimination means the vocabulary still needs work before "
        "we build a solver on top of it."
    )
    print(f"Report saved to: {report_path}")


if __name__ == "__main__":
    main()
