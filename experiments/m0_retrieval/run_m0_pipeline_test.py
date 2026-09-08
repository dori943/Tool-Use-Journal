"""Independent wrapper for production M0 + M1 + M2 + M3 integration tests.

The wrapper only prepares an isolated memory, invokes ``scripts/run.py``, and
reports artifacts. Retrieval and inference decisions remain production-owned.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path
from typing import Any


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MEMORY = ROOT / "output" / "m0_pipeline_test" / "memory.json"
PRODUCTION_MEMORY = ROOT / "output" / "memory.json"


def _empty_memory() -> dict[str, Any]:
    return {
        "schema_version": "1.0",
        "object_knowledge": {"objects": {}},
        "failure_recovery_experience": {"experiences": []},
    }


def _load_document(path: Path) -> dict[str, Any]:
    if not path.exists():
        return _empty_memory()
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"memory root must be an object: {path}")
    return raw


def _objects(document: dict[str, Any]) -> dict[str, Any]:
    section = document.get("object_knowledge") or {}
    objects = section.get("objects") or {}
    return objects if isinstance(objects, dict) else {}


def _fingerprint(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, default=str)


def _snapshot(path: Path) -> tuple[dict[str, Any], dict[str, str]]:
    document = _load_document(path)
    objects = _objects(document)
    return document, {key: _fingerprint(value) for key, value in objects.items()}


def _print_memory(label: str, path: Path, document: dict[str, Any]) -> None:
    objects = _objects(document)
    print(f"\n[{label}]")
    print(f"memory path: {path}")
    print(f"schema_version: {document.get('schema_version')}")
    print(f"entry count: {len(objects)}")
    print(f"keys: {sorted(objects)}")


def _print_entries(document: dict[str, Any]) -> None:
    objects = _objects(document)
    print("\n[Object Knowledge Entries]")
    if not objects:
        print("(empty)")
        return
    for key in sorted(objects):
        entry = objects[key] if isinstance(objects[key], dict) else {}
        identity = entry.get("identity") or {}
        geometry = entry.get("geometry") or {}
        physical = entry.get("physical_properties") or {}
        density = physical.get("density_kgm3") or {}
        metadata = entry.get("metadata") or {}
        print(json.dumps({
            "key": key,
            "object_id": identity.get("object_id"),
            "source_task": metadata.get("source_task"),
            "source_episode": metadata.get("source_episode"),
            "density_kgm3": density.get("value"),
            "extents_mm": geometry.get("extents_mm"),
            "stage": metadata.get("stage"),
            "episodes_seen": metadata.get("episodes_seen"),
            "updated_at": metadata.get("updated_at"),
            "stale": metadata.get("stale"),
        }, ensure_ascii=False))


def _read_retrieval_debug_path(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        print(f"warning: could not read retrieval debug {path}: {exc}")
        return None


def _debug_stamps(task: str) -> dict[Path, int]:
    out = ROOT / "output" / task
    return {path: path.stat().st_mtime_ns
            for path in out.glob("m0_retrieval*.json")}


def _fresh_retrieval_debug(task: str, before: dict[Path, int]):
    out = ROOT / "output" / task
    round_paths = sorted(out.glob("m0_retrieval.*.json"))
    fresh = [path for path in round_paths
             if before.get(path) != path.stat().st_mtime_ns]
    if not fresh:  # Backward-compatible fallback for older production output.
        latest = out / "m0_retrieval.json"
        if latest.exists() and before.get(latest) != latest.stat().st_mtime_ns:
            fresh = [latest]
    return [(path, _read_retrieval_debug_path(path)) for path in fresh]


def _display_command(command: list[str]) -> str:
    if os.name == "nt":
        return subprocess.list2cmdline(command)
    return shlex.join(command)


def _build_command(args: argparse.Namespace, memory: Path) -> list[str]:
    command = [
        sys.executable, str(ROOT / "scripts" / "run.py"), args.task,
        "--memory", str(memory), "--stop-after", "m3",
        "--backend", args.backend, "--seed", str(args.seed),
    ]
    if args.provider:
        command += ["--provider", args.provider]
    if args.model:
        command += ["--model", args.model]
    if args.no_roundtrip:
        command.append("--no-roundtrip")
    return command


def _api_configuration_available(provider: str | None) -> bool:
    if provider == "openai":
        names = ("OPENAI_API_KEY",)
    elif provider == "gemini":
        names = ("GEMINI_API_KEY", "GOOGLE_API_KEY")
    else:
        names = ("OPENAI_API_KEY", "GEMINI_API_KEY", "GOOGLE_API_KEY")
    if any(os.environ.get(name) for name in names):
        return True
    key_file = ROOT / "my_api_key.py"
    if not key_file.exists():
        return False
    text = key_file.read_text(encoding="utf-8", errors="ignore")
    return any(name in text for name in names)


def _validate_memory_path(path: Path) -> None:
    if path.resolve() == PRODUCTION_MEMORY.resolve():
        raise SystemExit(
            f"refusing to use production memory: {PRODUCTION_MEMORY}\n"
            "Choose an isolated path (default: output/m0_pipeline_test/memory.json)."
        )


def inspect(memory: Path) -> int:
    document = _load_document(memory)
    _print_memory("Memory", memory, document)
    _print_entries(document)
    return 0


def run_case(args: argparse.Namespace, memory: Path) -> int:
    if args.case == "initial":
        memory.parent.mkdir(parents=True, exist_ok=True)
        memory.write_text(json.dumps(_empty_memory(), ensure_ascii=False, indent=2),
                          encoding="utf-8")
    elif not memory.exists():
        raise SystemExit(
            f"{args.case} requires an existing test memory: {memory}\n"
            "Run --case initial first."
        )

    before_doc, before = _snapshot(memory)
    command = _build_command(args, memory)
    debug_stamps_before = _debug_stamps(args.task)

    print("[Test Configuration]")
    print(f"case: {args.case}")
    print(f"task: {args.task}")
    print(f"memory path: {memory}")
    print(f"provider: {args.provider or '(production default)'}")
    print(f"model: {args.model or '(production default)'}")
    print(f"backend: {args.backend}")
    _print_memory("Memory Before", memory, before_doc)
    print("\n[Pipeline Execution]")
    print(f"command: {_display_command(command)}")

    if args.command_only:
        print("return code: not executed (--command-only)")
        return 0
    if args.backend == "siphy" and not _api_configuration_available(args.provider):
        raise SystemExit(
            "No compatible API key was found; live SiPhy/C3 pipeline was not run. "
            "Set the provider API key, or use --backend mock for an API-free plumbing test."
        )

    completed = subprocess.run(command, cwd=ROOT, check=False)
    print(f"return code: {completed.returncode}")
    after_doc, after = _snapshot(memory)
    _print_memory("Memory After", memory, after_doc)

    before_keys, after_keys = set(before), set(after)
    added = sorted(after_keys - before_keys)
    removed = sorted(before_keys - after_keys)
    updated = sorted(key for key in before_keys & after_keys
                     if before[key] != after[key])
    print(f"newly added keys: {added}")
    print(f"updated keys: {updated}")
    print(f"removed keys: {removed}")

    debug_rounds = _fresh_retrieval_debug(args.task, debug_stamps_before)
    print("\n[Production Retrieval Debug]")
    if not debug_rounds:
        print("not available")
    else:
        for path, debug in debug_rounds:
            label = (debug or {}).get("round") or path.stem
            print(f"\n[{label} Retrieval Debug]")
            print(f"source: {path}")
            print(json.dumps(debug, ensure_ascii=False, indent=2)
                  if debug is not None else "debug read error")

    print("\n[Test Interpretation]")
    if args.case == "initial":
        print(f"initial Object Knowledge storage observed: {bool(added)}")
    elif args.case == "exact":
        same_task_before = [key for key, value in _objects(before_doc).items()
                            if (value.get("metadata") or {}).get("source_task") == args.task]
        print(f"same-task entries existed before execution: {sorted(same_task_before)}")
    else:
        print(f"new entries created by cross-task run: {added}")
    print("HIT/MISS and call counts are authoritative only when shown above by production debug.")
    return completed.returncode


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Run production M0+M1+M2+M3 against an isolated test memory.")
    parser.add_argument("--case", required=True,
                        choices=("initial", "exact", "cross", "inspect"))
    parser.add_argument("--task", help="production task id (required except for inspect)")
    parser.add_argument("--memory", type=Path, default=DEFAULT_MEMORY,
                        help="isolated test memory path")
    parser.add_argument("--provider", choices=("gemini", "openai"))
    parser.add_argument("--model")
    parser.add_argument("--backend", choices=("siphy", "mock"), default="siphy")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--no-roundtrip", action="store_true",
                        help="forward production run.py --no-roundtrip")
    parser.add_argument("--command-only", action="store_true",
                        help="print the exact production command without executing it")
    return parser


def main() -> int:
    args = build_parser().parse_args()
    memory = args.memory.resolve()
    _validate_memory_path(memory)
    if args.case == "inspect":
        return inspect(memory)
    if not args.task:
        raise SystemExit("--task is required for initial, exact, and cross cases")
    return run_case(args, memory)


if __name__ == "__main__":
    raise SystemExit(main())
