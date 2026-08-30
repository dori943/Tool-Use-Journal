"""Build a trimmed RoboCasa runtime for this project's C1-T2 / C2-T2 kitchens.

The script does not modify RoboCasa or this project.  It observes files opened
while Layout004 + Style002 environments are built, inspects the final merged
MJCF, and copies the union for all requested seeds into a new runtime tree.
"""

from __future__ import annotations

import argparse
import os
import shutil
import sys
import xml.etree.ElementTree as ET
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def parse_seeds(value: str) -> list[int]:
    """Accept ``0:100`` (end-exclusive) or ``0,4,9``."""
    if ":" in value:
        parts = [int(part) if part else None for part in value.split(":")]
        if len(parts) not in (2, 3):
            raise argparse.ArgumentTypeError("seed range must be START:STOP[:STEP]")
        return list(range(*parts))
    return [int(part) for part in value.split(",") if part.strip()]


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


class AssetOpenRecorder:
    def __init__(self, assets_root: Path):
        self.assets_root = assets_root.resolve()
        self.files: set[Path] = set()

    def audit(self, event, args) -> None:
        if event != "open" or not args:
            return
        raw = args[0]
        if not isinstance(raw, (str, bytes, os.PathLike)):
            return
        try:
            path = Path(os.fsdecode(raw)).resolve()
        except (OSError, TypeError, ValueError):
            return
        if path.is_file() and is_within(path, self.assets_root):
            self.files.add(path)


def add_xml_file_references(xml_text: str, roots: tuple[Path, ...], found: set[Path]) -> None:
    """Collect absolute file/include references from a merged MJCF document."""
    root = ET.fromstring(xml_text)
    for elem in root.iter():
        raw = elem.get("file")
        if not raw:
            continue
        candidate = Path(raw)
        if not candidate.is_absolute():
            continue
        candidate = candidate.resolve()
        if candidate.is_file() and any(is_within(candidate, base) for base in roots):
            found.add(candidate)


def copy_runtime_source(robocasa_root: Path, output: Path) -> None:
    """Copy RoboCasa Python source/metadata, deliberately excluding all assets."""
    output.mkdir(parents=True, exist_ok=True)
    for name in ("setup.py", "MANIFEST.in", "LICENSE", "README.md"):
        source = robocasa_root / name
        if source.exists():
            shutil.copy2(source, output / name)

    source_pkg = robocasa_root / "robocasa"
    target_pkg = output / "robocasa"

    def ignore(directory: str, names: list[str]) -> set[str]:
        directory_path = Path(directory).resolve()
        ignored = {"__pycache__"}
        if directory_path == (source_pkg / "models").resolve() and "assets" in names:
            ignored.add("assets")
        return ignored

    shutil.copytree(source_pkg, target_pkg, dirs_exist_ok=True, ignore=ignore)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seeds", type=parse_seeds, default=parse_seeds("0:100"))
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()

    import mujoco
    import robocasa
    import robosuite as suite
    from robocasa.models.scenes.scene_registry import get_layout_path, get_style_path

    # Importing the project registers all four custom environments.
    import environments  # noqa: F401

    robocasa_root = Path(robocasa.__file__).resolve().parent.parent
    assets_root = Path(robocasa.models.assets_root).resolve()
    output = args.output.resolve()
    if is_within(output, robocasa_root):
        raise RuntimeError("--output must not be inside the source RoboCasa tree")

    recorder = AssetOpenRecorder(assets_root)
    sys.addaudithook(recorder.audit)
    required: set[Path] = {
        Path(get_layout_path(4)).resolve(),
        Path(get_style_path(2)).resolve(),
    }

    env_names = ("C1_2_DoughFlatten", "C2_2_SandwichAssembly")
    per_run_counts: list[tuple[str, int, int]] = []
    for seed in args.seeds:
        for env_name in env_names:
            before = len(required | recorder.files)
            env = suite.make(
                env_name=env_name,
                robots="UR5e",
                has_renderer=False,
                has_offscreen_renderer=False,
                use_camera_obs=False,
                render_camera=None,
                seed=seed,
            )
            try:
                env.reset()
                xml_text = env.model.get_xml()
                # Validate that the merged XML itself is compilable.
                mujoco.MjModel.from_xml_string(xml_text)
                add_xml_file_references(xml_text, (assets_root,), required)
            finally:
                env.close()
            required.update(recorder.files)
            per_run_counts.append((env_name, seed, len(required) - before))

    copy_runtime_source(robocasa_root, output)
    target_assets = output / "robocasa" / "models" / "assets"
    for source in sorted(required):
        relative = source.relative_to(assets_root)
        destination = target_assets / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    manifest = output / "asset-manifest.txt"
    manifest.write_text(
        "\n".join(path.relative_to(assets_root).as_posix() for path in sorted(required))
        + "\n",
        encoding="utf-8",
    )
    report = output / "seed-report.tsv"
    report.write_text(
        "environment\tseed\tnew_union_files\n"
        + "\n".join(f"{env}\t{seed}\t{count}" for env, seed, count in per_run_counts)
        + "\n",
        encoding="utf-8",
    )
    print(f"Copied {len(required)} RoboCasa asset files to {target_assets}")
    print(f"Manifest: {manifest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
