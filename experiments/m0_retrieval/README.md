# M0 Experiment 1 — Physical Knowledge Factor Ablation

Find which combination of **BBox / Material / Density** lets M3 SiPhy-style
inference recover the remaining physical properties most accurately at the
lowest token cost.

## Scope

Included:

- Condition → one model call → physical property prediction → GT comparison
- Token cost from API `usage`
- Observation PNG validation (raw + bbox overlay)

Not included:

- A–B similarity / reuse / calibration
- Previous Case 1 / Case 2-A / Case 2-B / C1–C15 similarity experiments

## Why an experiment-side adapter?

Production `src/tuj/m3_grounding/siphy_backend.py`:

- Requires a crop RGB image
- User message is **image-only** (no text factors)
- Does not accept BBox / Material / Density GT as inputs
- `cls_hint` is unused in the VLM prompt (object name is not leaked there)

For factor ablation, using the production image path would leak geometry /
appearance into every condition. Production code is **not** modified.

`siphy_runner.ConditionedSiPhyRunner` reuses:

- `SiPhyBackend._make_client` (OpenAI client + API key resolution)
- Default model `gpt-4o-mini`
- One `chat.completions.create` call per Object × Condition
- Token fields from `response.usage` (`prompt_tokens`, `completion_tokens`, `total_tokens`)

and sends **text factors only** (no image, no object name).

## Conditions

| ID | Inputs | Predict |
|----|--------|---------|
| C1 | BBox | Material, Density, Mass, Mu, Young's |
| C2 | Material | Density, Mass, Mu, Young's |
| C3 | Density | Material, Mass, Mu, Young's |
| C4 | BBox + Material | Density, Mass, Mu, Young's |
| C5 | BBox + Density | Material, Mass, Mu, Young's |
| C6 | Material + Density | Mass, Mu, Young's |
| C7 | BBox + Material + Density | Mass, Mu, Young's |

Input properties are never scored as predictions.

## Ground Truth sources

| Property | Source |
|----------|--------|
| BBox | MJCF `reg_bbox` size (mm). Mug falls back to placement sites if `reg_bbox` missing. |
| Material | MJCF `custom/text[@name=material_gt]` |
| Density | MJCF collision geom `density` (often placeholder `100`) |
| Mass | MuJoCo compiled `body_mass` sum |
| Mu | MJCF collision geom `friction[0]` |
| Young's | **Unavailable** in assets (excluded from evaluation) |

## Overall Error

```
material_error = 1 - material_accuracy
overall_error = mean(available property mean-errors)
```

Properties with no evaluated samples are excluded. Do not treat categorical
accuracy and relative error as interchangeable without this conversion.

## Commands

Dry-run (no LLM):

```powershell
Set-Location C:\Users\yebin\tool-use-robosuite\Tool-Use-Journal-Experiment
$env:PYTHONIOENCODING = "utf-8"
$env:MUJOCO_GL = "wgl"
& C:\Users\yebin\anaconda3\envs\robocasa\python.exe experiments/m0_retrieval/run_experiment1.py --dry-run
```

## Observation capture (single-object scene)

Observations are **not** taken from c1_1 / c2_1 / c2_2.

`single_object_scene.py` builds a MuJoCo scene with floor + table + one
production object class + fixed camera `exp1_cam`.

```powershell
python experiments/m0_retrieval/run_experiment1.py --capture-observations --object bottle
# later, all objects:
python experiments/m0_retrieval/run_experiment1.py --capture-observations
```

Outputs: `output/m0_retrieval/experiment1/observations/{object}_raw.png` and `_bbox.png`.
BBox overlay is validation-only and is **not** sent to SiPhy/LLM.


Live experiment (49 units max; skip units with missing input GT):

```powershell
& C:\Users\yebin\anaconda3\envs\robocasa\python.exe experiments/m0_retrieval/run_experiment1.py --model gpt-4o-mini
```

## Outputs

`output/m0_retrieval/experiment1/`

- `raw_results.csv`
- `condition_summary.csv`
- `gt_manifest.json`
- `run_metadata.json`
- `observations/{object}_raw.png`
- `observations/{object}_bbox.png`
