# Task-level evaluation from saved execution evidence

`task-goals-v1` evaluates the task registry's complete goal, independently of the
planner's selected subgoals. It reads measured `m5/initial_world.json`, the selected
live execution manifest's `final_world`, successful step requests, and (for a lid)
physical settle reports. No simulator, model request, or environment success stub
is used. This is a documented **post-hoc policy**, not a pre-registered paper metric.

The scripted live runner previously initialized `task_goal_status=NOT_EVALUATED`
and never changed it. Some scene environments also return a constant `False` from
`_check_success`, so calling those placeholders would incorrectly fail every run.
`run.py` previously fell back to execution success when goal evidence was absent.

The live CLI now writes `m5/task_goal_evaluation.json` on completion and on execution
failure. Its execution status and exit code still describe whether the sequence ran;
`run_result.json` separately records execution and task goal status. Overall success
requires both. An execution-complete run with missing goal evidence has null success
and `failure_stage=evaluation`, not true. The goal path/version survives batch collection.

## Policy

All coordinates and geometry come from runtime observations. Local collision points
are transformed by the full measured orientation. Measured dimensions are a fallback
only when collision vertices are unavailable; missing geometry never gets a default
size or an automatic pass. A known failed mandatory condition proves failure even
when other checks lack evidence. Otherwise missing evidence is `UNKNOWN`.

- **C1_1:** all 12 blocks fully in the collection region, supported there, released.
- **C2_1:** apple, bread, mug, plate, spoon fully in the blue tray, released and with
  a geometric support path to the tray. Actual successful acquisition and placement
  order must match initial left-to-right order. The fixed observation camera has
  image-right in world +Y. Missing successful-step requests leave order unknown.
- **C3_1:** the same final arrangement conditions, without the ordering condition.
- **C1_2:** reuse measured `flattening_outcome`: residual height reduction in configured
  30–50% interval, tool contact and causal reduction, unloading observation, volume
  tolerance and force limit. Missing contact/deformation data is unknown.
- **C2_2:** five ingredients above and within 9cm of the serving plate center, bread
  on both ends of the height order, all three fillings between, geometric support
  of each adjacent layer and the bottom bread on the plate, all released. The 9cm
  criterion comes from the existing environment; support uses the existing predicate.
- **C3_2:** each tray contains one plate, mug, fork and spoon; each plate supports one
  bread and fruit. All are released. Same-kind a/b identities may swap trays. A
  provisional interpretation of “neatly” requires fork/spoon long axes within 15°
  (direction reversal allowed), and their convex footprints not overlapping beyond
  the contact tolerance. It does not prescribe mirrored coordinates or an exact meal
  layout. This proxy needs user confirmation before being a paper metric.
- **C4_1:** card moves at least 5cm along the initial channel, its full footprint clears
  both initial and final appliance spans, it stays within 5mm of its original support
  height and is released. Moving appliances or dropping the card off the table cannot
  produce success. Rotation is included in the bounds.
- **C4_2:** all five required objects fully inside the box's measured 3D interior,
  the solid lid face covers the opening at rim height, everything released, and the
  saved lid-placement settle report verifies loaded upward contacts. A matching
  request/target/region is mandatory; later changes to lid or contents invalidate
  earlier settle evidence. Geometric coverage alone cannot prove physical support.

Whole-object footprint tolerance is **1.5mm**, the existing region predicate default.
Resting checks use the existing **5mm** gap/penetration tolerance. C4_2 cover angular
tolerance is the existing 0.05 rad default. These are explicit operational criteria;
not every physically usable layout necessarily satisfies them. In particular, the
per-step open-plate placement check allows 30mm drift, which is more permissive than
this whole-task rule. Never adjust a threshold simply to make an observed run pass.
Geometric resting is a snapshot predicate, not a new contact or long-term stability
measurement. A full physics replay is not needed for this saved-state evaluation.

## Offline comparison and preservation

```sh
PYTHONPATH=src:. python scripts/evaluate_task_goals.py \
  --baseline output/ablation/without-reasoner \
  --current output/ablation-rerun-20261024-2351-6075fdb/without-reasoner \
  --output output/task-goal-evaluation-20261025-v1
```

Output must be new; existing output is rejected. Original experiment files, summaries
and videos are never overwritten. Each task gets an evidence JSON and a derived
`run_result.evaluated.json`; `comparison.md`/JSON compare all eight tasks. SHA-256 hashes
of all consumed evidence are verified again after reading. Windows absolute paths
are relocated inside the supplied run directory, never followed to another checkout.
Multiple live runs without a unique summary pointer are unknown. Batch timeouts and
`IN_PROGRESS` manifests are incomplete, even if their last saved state appears good.
An early pipeline failure without a measured M5 state is `NOT_EXECUTED`, not a measured
physical-goal failure. A failed live runtime startup cannot reuse a prior manifest.

No per-task logic changes the Reasoner, planner, geometry inputs, or physics. No new
experiments or paid API calls are made by the evaluator.
