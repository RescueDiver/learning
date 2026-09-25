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


---

# General structural language pivot

The earlier eee78d87 experiment was useful as a microscope, but it exposed a
danger: a learner can become very good at describing one task family without
building concepts that transfer broadly enough to matter.

The New branch therefore now treats task-specific structures such as
`periodic base + repeated overlay` as just one possible vocabulary family.
They are no longer the center of the architecture.

The new center is a shared scene language.

## General scene representation

`new/general_scene.py` describes every grid through several simultaneous
views rather than betting on one segmentation:

- same-color 4-connected objects
- same-color 8-connected objects
- all-foreground 4-connected objects
- all-foreground 8-connected objects

For each object it records reusable properties including:

- color
- area
- bounding box
- height and width
- density
- border contact
- line / rectangle / square
- holes
- centroid
- normalized shape
- canonical shape under rotation/reflection

For pairs of objects it records reusable relationships including:

- left/right ordering
- above/below ordering
- aligned top/bottom/left/right edges
- aligned centers
- same area
- same normalized shape
- same shape family under rotation/reflection
- same color
- touching / adjacent
- bounding-box containment
- Manhattan gap

At grid level it records:

- dimensions
- palette and dominant color
- horizontal / vertical / diagonal symmetry
- row and column periodicity

## Input -> output structural description

The pair analyzer does not try to solve a task. It asks what changed.

It attempts object correspondences and emits transformation tokens such as:

- grid:size_preserved
- grid:size_changed
- grid:integer_upscale
- color:added
- color:removed
- color:palette_preserved
- color:deterministic_cell_map
- objects:created
- objects:deleted
- objects:moved
- objects:recolored
- objects:shape_preserved
- objects:rot90 / rot180 / mirror...
- movement:uniform_displacement
- movement:mixed_displacements
- relations:preserved
- relations:added
- relations:removed
- symmetry:* created / preserved / broken
- global:rotation / reflection / identity

A task signature is the intersection of these tokens across all of its training
pairs. That means the signature contains relationships that survive changes in
the individual examples, not details from only one pair.

## Mixed-task benchmark

`run_scene_batch.py` deliberately samples across many human task groups.
The benchmark is not a solver and does not award itself credit for explaining
one special task.

It measures:

1. **Coverage** — does the same structural language produce stable invariant
   descriptions across many unrelated tasks?
2. **Discrimination** — using only those structural tokens, which other tasks
   look structurally closest?
3. **Human-group diagnostic** — after the structural comparison is made, the
   manual task group is shown so we can see whether the representation is
   capturing useful distinctions.
4. **Group-specific vocabulary** — which structural tokens recur inside a
   sampled group without being ubiquitous everywhere else?

If this benchmark is weak, the response should be to improve the scene
language. We should not paper over weak representation by adding task IDs,
special-case routers, or increasingly elaborate rules for one puzzle.

## Run the benchmark

From the learning repository root:

```powershell
python -m unittest tests.test_general_scene
```

Then run a mixed sample:

```powershell
python run_scene_batch.py `
  --data "C:\Users\eric_\GitHub\ARCS6\data\data.json" `
  --groups "C:\Users\eric_\GitHub\ARCS6\data\task_groups.json" `
  --limit 24
```

The default takes up to two tasks from each human group in round-robin order
until the requested limit is reached. Increase `--limit` when the smaller
benchmark is healthy.

A JSON report is written to:

```text
general_scene_report.json
```

That report preserves per-task invariant tokens, pair-level details, nearest
structural neighbors, and token frequencies for later analysis.

## Current principle

The architecture should grow in this order:

```text
cells
  -> several candidate object views
  -> object attributes
  -> object-to-object relationships
  -> input/output correspondences
  -> changes in relationships
  -> stable task-level structural facts
  -> candidate transformations
  -> exact training validation
```

The solver should eventually search over transformations expressed in this
shared language. It should not begin by guessing a human task label and it
should not require a special rule family to see ordinary structural facts.
