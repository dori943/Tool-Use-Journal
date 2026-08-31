# Precomputed EE attach, return, and exchange trajectories

The production default is `precomputed-required`.  Store only trajectories
that have completed controller replay and the applicable lock/release model
transition validation.

Expected layout:

```text
configs/precomputed_ee_paths/
  C1_1_LegoSweep/
    bare_to_2F.json
    bare_to_3F.json
    bare_to_vac.json
    2F_to_bare.json
    3F_to_bare.json
    vac_to_bare.json
```

Do not copy a request-bound `MotionPlan` into this directory.  After a dynamic
development run has passed controller validation, export it with
`save_ee_attach_template(...)`.  The exporter removes request/scene provenance
and stores the time-parameterized joint commands, static workcell signatures,
collision model versions, and lock events in `EEAttachTrajectoryTemplate` 1.0.

Pairwise exchanges do not need six additional artifacts.  Motion Planner
composes `<source>_to_bare.json` with `bare_to_<target>.json`, offsets the second
timeline, verifies the joint-space seam and all four release/lock events, and
returns one request-bound `MotionPlan`.

The commissioning command performs dynamic generation once, controller
execution, fresh-runtime precomputed replay (three times by default), and only
then publishes the template:

```text
python scripts/generate_precomputed_ee_attach.py 2F
python scripts/generate_precomputed_ee_attach.py 3F
python scripts/generate_precomputed_ee_attach.py vac
```

After the attach trajectories exist, commission the three return trajectories.
The command derives a reverse-path candidate, rebuilds its collision contexts
and unlock events, controller-replays the return, and then executes both
possible composed exchanges before publishing it:

```text
python scripts/generate_precomputed_ee_return.py 2F
python scripts/generate_precomputed_ee_return.py 3F
python scripts/generate_precomputed_ee_return.py vac
```

Revalidate a published trajectory from fresh bare runtimes without regenerating
it:

```text
python scripts/generate_precomputed_ee_attach.py 2F --replay-existing --replay-count 3
python scripts/generate_precomputed_ee_return.py 2F --replay-existing --replay-count 3
```

The return start state is the corresponding attach trajectory's final
`exchange-entry` joint state.  For an attached-EE exchange, the selected-plan
orchestrator emits an explicit `EE_EXCHANGE_ENTRY` request before
`EE_EXCHANGE`.  That preparation request loads `<source>_to_bare.json` only to
obtain and validate the stored entry state, then plans `current joints → entry`
under `ee-attached:<source>` collision geometry.  It checks a direct joint path
first and uses bounded RRT-Connect only when needed.  The precomputed
return/attach playback begins only after this request succeeds.  Direct callers
that skip the preparation request still fail closed with `START_STATE_MISMATCH`.
The return ends at the exact canonical bare-home state required by every attach
artifact.

Until the applicable validated files exist, production attach/exchange fails
with `PRECOMPUTED_EE_PATH_NOT_FOUND`. Dynamic fallback must be requested
explicitly with `--ee-attach-policy precomputed-or-plan`.
