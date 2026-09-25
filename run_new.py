from __future__ import annotations

import argparse
import json
from pathlib import Path

from new.program_learner import infer_simple_programs
from new.vocabulary import build_vocabulary, executable_words


def load_task(path: Path, task_id: str) -> dict:
    with path.open("r", encoding="utf-8") as file:
        data = json.load(file)
    return data[task_id]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("task_id")
    parser.add_argument(
        "--data",
        default="data/data.json",
    )
    args = parser.parse_args()

    task = load_task(Path(args.data), args.task_id)
    pairs = task.get("train", [])

    vocabulary = build_vocabulary()
    print("=" * 72)
    print("NEW VOCABULARY + DEMONSTRATION LEARNER")
    print("=" * 72)
    print(f"Task: {args.task_id}")
    print(f"Known imported words: {len(vocabulary)}")
    print(f"Executable core words: {len(executable_words())}")
    print()

    learned = infer_simple_programs(pairs)

    if not learned:
        print("No exact 1-2 word program found yet.")
        print(
            "That is useful: the next step is to add perception/parameter "
            "binding or teach a demonstration trace, not random task-ID code."
        )
        return

    print(f"Exact learned programs: {len(learned)}")
    for index, program in enumerate(learned, start=1):
        print()
        print(f"{index}. {program.demonstration.name}")
        for step in program.demonstration.steps:
            print(f"   {step.word} {step.args}")


if __name__ == "__main__":
    main()
