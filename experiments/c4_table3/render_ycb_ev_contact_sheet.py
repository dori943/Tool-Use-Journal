"""Render one YCB mesh beside one locked EV crop per target object."""
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image, ImageDraw, ImageFont


def render_mesh(mesh_path: str, size: int = 256, view: int = 0) -> Image.Image:
    camera_positions = [(0, -0.62, 0.16), (0.62, 0, 0.16), (0, -0.30, 0.62)]
    cam_pos = camera_positions[view % len(camera_positions)]
    xml = f'''<mujoco model="preview"><option integrator="RK4"/>
<visual><headlight ambient="0.8 0.8 0.8" diffuse="0.9 0.9 0.9" specular="0.3 0.3 0.3"/></visual>
<asset><mesh name="m" file="{mesh_path}"/></asset>
<worldbody>
<light name="key" pos="0 -0.4 0.6" diffuse="1 1 1" specular="0.4 0.4 0.4"/>
<light name="fill" pos="0.4 0.2 0.4" diffuse="0.8 0.8 0.8"/>
<light name="rim" pos="-0.4 0.2 0.5" diffuse="0.7 0.7 0.7"/>
<camera name="cam" mode="targetbody" target="obj" pos="{cam_pos[0]} {cam_pos[1]} {cam_pos[2]}"/>
<geom name="floor" type="plane" size="1 1 0.01" pos="0 0 -0.14" rgba="0.92 0.92 0.92 1" contype="0" conaffinity="0"/>
<body name="obj" pos="0 0 0"><geom type="mesh" mesh="m" rgba="0.82 0.84 0.88 1"/></body></worldbody></mujoco>'''
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
    canvas = Image.new("RGB", (1500, len(object_ids) * 330), "white")
    draw = ImageDraw.Draw(canvas)
    for idx, oid in enumerate(object_ids):
        row = by_object[oid]
        crop = Path(row["mass_crop_path"])
        if not crop.is_absolute():
            crop = repo_root / crop
        crop_img = Image.open(crop).convert("RGB").resize((256, 256))
        mesh = audit["targets"][oid]["mesh_files"][0]["path"]
        mesh_imgs = [render_mesh(mesh, view=view) for view in range(3)]
        y = idx * 330
        canvas.paste(crop_img, (20, y + 35))
        for view, mesh_img in enumerate(mesh_imgs):
            canvas.paste(mesh_img, (300 + view * 280, y + 35))
        draw.text((20, y + 8), f"{oid}  EV static crop", fill="black")
        draw.text((300, y + 8), "YCB mesh previews (3 views; pose not aligned)", fill="black")
        draw.text((1140, y + 50), "image↔mesh instance: UNVERIFIED", fill="black")
        draw.text((1140, y + 80), "shape comparison only", fill="black")
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
