"""Greedy EE-order experiment — separate from production C3-2 / Ours.

## What this is
Myopic planner: among dependency-ready subgoals, pick the (subgoal, EE) with
the smallest *immediate* EE-switch cost. No EE freeze. No M4 global search.

## What this is not
- Not ``without-planner`` (that keeps M2 order and ranks by suitability).
- Not production ``python scripts/run.py c3_2`` (ENTRIES / full M4 unchanged).

## C3_2 scripted extras (greedy path only)
Opened via ``GREEDY_EXTRA_ENTRIES`` + ``scripted_grasp_greedy_extra``:

- bread / spoon / fork → 2F (in addition to production 3F / vac)
- fruit → 2F (AABB ~81 mm < 85 mm stroke)
- mug → 3F only (AABB ~88 mm > 85 mm stroke)

## Run
```bash
python scripts/run.py c3_2 --planner-mode greedy-ee-order --seed 0
# or
python scripts/run_greedy_ee_order.py   # imported via run.py dispatch
```

Outputs under ``output/greedy-ee-order/<task>/seed_<k>/``.
"""
