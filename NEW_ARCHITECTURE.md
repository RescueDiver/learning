# NEW branch: vocabulary + demonstration learning

This branch is a clean experiment separate from the earlier router work.

## What changed

The old learner mostly tried to classify tasks into broad groups. That can
change search order, but it cannot solve a task if the solver does not have
the right concepts or operations.

This branch attacks the two suspected bottlenecks directly:

1. **Vocabulary** — import the names/ideas of proven ARC DSL operations.
2. **Perception + demonstration** — represent a worked solution as readable
   actions such as select an object, move it, recolor it, rotate it, copy it,
   and place it.

## Sources

### ARC-DSL

Michael Hodel's ARC-DSL is MIT licensed and provides a 160-word DSL plus
hand-written solver programs for all 400 ARC training tasks.

Source:
https://github.com/michaelhodel/arc-dsl

We keep the vocabulary names as a reference and reimplement a smaller
executable core in this repository rather than copying the source file.

Useful word families include:

- object extraction and filtering
- crop / subgrid / trim
- shift / move / gravitate
- rotate / mirror
- recolor / replace / switch
- fill / paint / cover
- connect / box / frontiers
- upscale / downscale / concatenate
- period / occurrence / pattern operations

### ARGA

ARGA (Xu, Khalil, Sanner, AAAI 2023) is also MIT licensed. It represents ARC
images as object graphs and searches an object-centric DSL.

Source:
https://github.com/khalil-research/ARGA-AAAI23

Ideas brought into this branch:

- filter objects by color, size, degree, and neighbors
- bind transformation parameters from another object
- move / extend / rotate nodes
- add borders
- fill or hollow rectangles
- mirror / flip / insert / remove objects
- reason about relative position

## Files

- `new/vocabulary.py`
  - 160 ARC-DSL word names
  - ARGA object/graph word names
  - marks which words currently have executable support

- `new/executor.py`
  - independent implementations of the first executable word set

- `new/demonstration.py`
  - human-readable worked-example traces
  - e.g. select object -> move -> paint

- `new/program_learner.py`
  - first program-synthesis proof of concept
  - proposes task-supported parameters
  - exact reconstruction is still the judge

- `run_new.py`
  - command-line experiment

## Run

From the repository root:

```powershell
python run_new.py eee78d87
```

## Demonstration idea

A human demonstration can look like:

```python
Demonstration(
    name="move_largest_right",
    steps=(
        Step("select_object", {"by": "largest"}),
        Step("move_selected", {"dr": 0, "dc": "selected_width"}),
    ),
)
```

The important difference from a label such as "make a shape" is that the
demonstration says **what moved, how it moved, and what operation happened**.

A learned demonstration must reconstruct every training output exactly before
it is accepted.

## Next experiment

Use failed ARC tasks as teaching examples. For each failure:

1. Describe the smallest sequence of existing words that a human uses.
2. If a needed word does not exist, add one reusable word.
3. Parameterize the demonstration from visible task evidence.
4. Verify the same program on every train pair.
5. Save successful programs as reusable skills.

That gives us a controlled way to grow the language instead of randomly
adding solver hacks.
