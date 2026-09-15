"""Render one YCB mesh beside one locked EV crop per target object."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render_mesh(mesh_path: str, size: int = 256) -> Image.Image:
    xml = f'''<mujoco model="preview"><asset><mesh name="m" file="{mesh_path}"/></asset>
<worldbody><light pos="0 -0.3 0.5" diffuse="1 1 1"/>
<camera name="cam" pos="0 -0.65 0.25" xyaxes="1 0 0 0 0.35 0.94"/>
<body pos="0 0 0"><geom type="mesh" mesh="m" rgba="0.75 0.75 0.78 1"/></body></worldbody></mujoco>'''
    model = mujoco.MjModel.from_xml_string(xml)
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)
    renderer = mujoco.Renderer(model, height=size, width=size)
    renderer.update_scene(data, camera="cam")
    return Image.fromarray(renderer.render().copy())


def build(manifest: Path, audit_json: Path, output: Path, repo_root: Path) -> Path:
    rows = list(csv.DictReader(manifest.open(encoding="utf-8", newline="")))
    audit = json.loads(audit_json.read_text(encoding="utf-8"))
    by_object = {}
    for row in rows:
        by_object.setdefault(row["object_id"], row)
    object_ids = list(audit["targets"])
    canvas = Image.new("RGB", (1200, len(object_ids) * 330), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, oid in enumerate(object_ids):
        row = by_object[oid]
        crop = Path(row["mass_crop_path"])
        if not crop.is_absolute():
            crop = repo_root / crop
        crop_img = Image.open(crop).convert("RGB").resize((256, 256))
        mesh = audit["targets"][oid]["mesh_files"][0]["path"]
        mesh_img = render_mesh(mesh)
        y = idx * 330
        canvas.paste(crop_img, (20, y + 35))
        canvas.paste(mesh_img, (300, y + 35))
        draw.text((20, y + 8), f"{oid}  EV static crop", fill="black")
        draw.text((300, y + 8), "YCB mesh preview (pose not aligned)", fill="black")
        draw.text((600, y + 50), "image↔mesh instance: UNVERIFIED", fill="black")
        draw.text((600, y + 80), "shape comparison only", fill="black")
    output.parent.mkdir(parents=True, exist_ok=True)
    canvas.save(output)
    return output


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", type=Path, required=True)
    ap.add_argument("--audit-json", type=Path, required=True)
    ap.add_argument("--repo-root", type=Path, default=Path("."))
    ap.add_argument("--output", type=Path, required=True)
    args = ap.parse_args()
    print(build(args.manifest, args.audit_json, args.output, args.repo_root.resolve()))


if __name__ == "__main__":
    main()
