from __future__ import annotations

import argparse
import json
from pathlib import Path

from new.program_learner import infer_simple_programs
from new.structural_learner import learn_periodic_motif_program
from new.structural_perception import mask_signature, minority_motif
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
    print("NEW VOCABULARY + STRUCTURAL DEMONSTRATION LEARNER")
    print("=" * 72)
    print(f"Task: {args.task_id}")
    print(f"Known imported words: {len(vocabulary)}")
    print(f"Executable core words: {len(executable_words())}")
    print()

    learned = infer_simple_programs(pairs)

    print("SIMPLE WORD SEARCH")
    print("-" * 72)
    if not learned:
        print("No exact 1-2 word program found.")
    else:
        print(f"Exact learned programs: {len(learned)}")
        for index, program in enumerate(learned, start=1):
            print(f"{index}. {program.demonstration.name}")
            for step in program.demonstration.steps:
                print(f"   {step.word} {step.args}")
    print()

    structural = learn_periodic_motif_program(pairs)

    print("STRUCTURAL PERCEPTION / PARAMETER BINDING")
    print("-" * 72)
    if structural.program is None:
        print("No periodic + motif structural program learned.")
        for reason in structural.unresolved:
            print(f"  unresolved: {reason}")
        return

    program = structural.program
    print(f"Train exact: {structural.exact_train}")
    print(f"Output shape: {program.output_shape}")
    print(f"Periodic base: {program.period}")
    print(f"Overlay color: {program.overlay_color}")
    print(f"Overlay bbox: {program.overlay_bbox}")
    print(f"Motif repeat: {program.overlay_repeat}")
    print(f"Learned motif/template associations: {len(program.motif_to_tile)}")
    print(
        "Motif->tile relation: "
        + (program.relation_rule.name if program.relation_rule else "NONE")
    )
    print()

    for item in structural.evidence:
        print(f"  evidence: {item}")

    if structural.unresolved:
        print()
        for reason in structural.unresolved:
            print(f"  unresolved: {reason}")

    tests = task.get("test", [])
    if not tests:
        return

    print()
    print("TEST STRUCTURAL CHECK")
    print("-" * 72)

    for index, test in enumerate(tests, start=1):
        motif = minority_motif(test["input"])
        if motif is None:
            print(f"Test {index}: no input motif found")
            continue

        signature = mask_signature(motif.mask)
        known = signature in program.motif_to_tile
        prediction = program.apply(test["input"])

        print(f"Test {index} motif: {signature}")
        print(f"Test {index} motif known from training: {known}")
        print(
            f"Test {index} relation generalization available: "
            f"{program.relation_rule is not None}"
        )

        if prediction is None:
            print(
                f"Test {index}: NO PREDICTION -- no LOO-validated relationship "
                "can map this unseen motif to a periodic tile."
            )
        else:
            print(f"Test {index}: structural prediction generated")
            for row in prediction:
                print(" ".join(str(value) for value in row))


if __name__ == "__main__":
    main()
