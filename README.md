# learning

Experimental curriculum learner for ARCS6.

The goal is to teach broad task families first, then refine them into reusable skills.

Initial experiment: the human-created **missing mask** group.

Run this repo beside ARCS6:

```text
GitHub/
    ARCS6/
    learning/
```

Then:

```powershell
python train_curriculum.py --group "missing mask"
```

The program reads ARCS6's `data/data.json` and `task_groups.json`, learns a broad internal concept from Eric's grouping, compares that group against the other human groups, and saves learned state under `learned/`.

This is intentionally a small teaching experiment first, not a replacement for the ARCS6 solver.
