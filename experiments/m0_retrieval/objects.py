"""Experiment 1 object catalog — maps logical names to production assets."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OBJECTS_ASSET_DIR = ROOT / "environments" / "assets" / "objects"


@dataclass(frozen=True)
class ExperimentObject:
    key: str
    label: str
    asset: str
    observation_task: str
    observation_instance: str
    xml_name: str = "model.xml"


# All seven targets exist under environments/assets/objects/.
OBJECTS: dict[str, ExperimentObject] = {
    o.key: o
    for o in (
        ExperimentObject(
            key="bottle",
            label="Bottle",
            asset="bottle",
            observation_task="exp1_single_object",
            observation_instance="target",
        ),
        ExperimentObject(
            key="spoon",
            label="Spoon",
            asset="spoon",
            observation_task="exp1_single_object",
            observation_instance="target",
        ),
        ExperimentObject(
            key="ladle",
            label="Ladle",
            asset="ladle",
            observation_task="exp1_single_object",
            observation_instance="target",
        ),
        ExperimentObject(
            key="plate",
            label="Plate",
            asset="plate",
            observation_task="exp1_single_object",
            observation_instance="target",
        ),
        ExperimentObject(
            key="mug",
            label="Mug",
            asset="mug_3143a4ac",
            observation_task="exp1_single_object",
            observation_instance="target",
        ),
        ExperimentObject(
            key="apple",
            label="Apple",
            asset="apple",
            observation_task="exp1_single_object",
            observation_instance="target",
        ),
        ExperimentObject(
            key="bread",
            label="Bread",
            asset="bread",
            observation_task="exp1_single_object",
            observation_instance="target",
        ),
    )
}


def asset_dir(obj: ExperimentObject) -> Path:
    return OBJECTS_ASSET_DIR / obj.asset
