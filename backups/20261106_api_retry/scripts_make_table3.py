# -*- coding: utf-8 -*-
"""Table III (task success rate) 생성기 — run_result.json 만 읽는다. API/시뮬레이터 호출 없음.

  python scripts/make_table3.py                       # output/ablation 아래 전부
  python scripts/make_table3.py --root output/ablation output/ablation_gemini
  python scripts/make_table3.py --out output/table3   # table3.md / table3.csv / table3.json

집계 규칙
  · trial 1건 = <root>/<method>/<task>/seed_<s>[_r<k>]/run_result.json  (seed_*_prev_* 제외)
  · success 는 run_result.json 의 "success" (= exec_success AND goal_success). None(증거 부족) 은
    기본적으로 실패로 세지 않고 표본에서 제외하며 n 에 표시한다 (--unknown-as-fail 로 바꿀 수 있음)
  · Macro SR = 태스크별 SR 의 산술평균, Pooled = 전체 성공/전체 시도, 95% CI = Wilson
  · McNemar = Ours 와 (task, seed, rep, provider) 로 짝지은 trial 에 대한 exact 검정 (양측)
  · 태스크별 Fisher exact(p) + Holm 보정 — 논문 표 각주 재현용
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TASK_ORDER = ["c1_1", "c1_2", "c2_1", "c2_2", "c3_1", "c3_2", "c4_1", "c4_2"]
METHOD_LABEL = {
    "full": "Ours",
    "without-reasoner": "w/o Reasoner",
    "without-planner": "w/o EE Swap-Aware Planner",
    "without-grounding": "w/o Reasoner (legacy)",
    "greedy-ee-order": "Greedy EE order",
}
METHOD_ORDER = ["without-reasoner", "without-planner", "without-grounding", "greedy-ee-order", "full"]


def wilson(k: int, n: int, z: float = 1.959964) -> tuple[float, float]:
    if n == 0:
        return (float("nan"), float("nan"))
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return (max(0.0, center - half), min(1.0, center + half))


def mcnemar_exact(b: int, c: int) -> float:
    """b = Ours 성공 & 비교군 실패, c = Ours 실패 & 비교군 성공. 양측 exact binomial."""
    n = b + c
    if n == 0:
        return float("nan")
    k = min(b, c)
    tail = sum(math.comb(n, i) for i in range(0, k + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def fisher_exact(a: int, b: int, c: int, d: int) -> float:
    """2x2 [[a, b], [c, d]] 양측 Fisher exact (확률 합산 방식)."""
    n = a + b + c + d
    r1, c1 = a + b, a + c

    def prob(x):
        return (math.comb(r1, x) * math.comb(n - r1, c1 - x)) / math.comb(n, c1)

    lo, hi = max(0, c1 - (n - r1)), min(r1, c1)
    p_obs = prob(a)
    return min(1.0, sum(prob(x) for x in range(lo, hi + 1) if prob(x) <= p_obs * (1 + 1e-9)))


def holm(pvals: dict[str, float]) -> dict[str, float]:
    items = sorted((p, k) for k, p in pvals.items() if not math.isnan(p))
    m = len(items)
    out, running = {}, 0.0
    for i, (p, k) in enumerate(items):
        running = max(running, (m - i) * p)
        out[k] = min(1.0, running)
    for k, p in pvals.items():
        out.setdefault(k, p)
    return out


def _read(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


def load_trials(roots: list[Path]) -> list[dict]:
    trials = []
    for root in roots:
        for seed_dir in sorted(root.glob("*/*/seed_*")):
            if not seed_dir.is_dir() or "_prev_" in seed_dir.name or seed_dir.parent.name == "logs":
                continue
            method, task = seed_dir.parent.parent.name, seed_dir.parent.name
            rr = seed_dir / "run_result.json"
            r = None
            if rr.is_file():
                try:
                    r = json.loads(rr.read_text(encoding="utf-8"))
                except (OSError, json.JSONDecodeError):
                    r = None
            if r is None:
                # M5 전에 죽은 trial (M2/M4 NoCandidate 등): run_result.json 이 없다.
                # ablation_manifest.failure / batch_record 로 실패로 집계한다.
                man = _read(seed_dir / "ablation_manifest.json")
                rec = _read(seed_dir / "batch_record.json")
                fail = (man or {}).get("failure") or {}
                if not fail and not rec:
                    continue  # 아직 실행 중이거나 빈 폴더
                r = {"provider": (rec or {}).get("provider"), "model": None,
                     "success": False if (fail or (rec or {}).get("success") is False) else None,
                     "exec_success": False, "goal_success": None,
                     "failure_stage": fail.get("failure_stage") or fail.get("stage") or (rec or {}).get("failure_stage"),
                     "failure_code": fail.get("failure_code") or (rec or {}).get("reason_code"),
                     "plan_status": (rec or {}).get("plan_status"), "pre_m5_failure": True}
            name = seed_dir.name  # seed_0 | seed_0_r3
            parts = name.split("_")
            seed = int(parts[1])
            rep = int(parts[2][1:]) if len(parts) > 2 and parts[2].startswith("r") else 0
            trials.append({
                "root": str(root), "method": method, "task": task, "seed": seed, "rep": rep,
                "provider": r.get("provider"), "model": r.get("model"),
                "success": r.get("success"), "exec_success": r.get("exec_success"),
                "goal_success": r.get("goal_success"), "failure_stage": r.get("failure_stage"),
                "failure_code": r.get("failure_code"), "plan_status": r.get("plan_status"),
                "pre_m5_failure": bool(r.get("pre_m5_failure")), "dir": str(seed_dir),
            })
    return trials


def build(trials: list[dict], unknown_as_fail: bool) -> dict:
    def outcome(t):
        s = t["success"]
        if s is None:
            return False if unknown_as_fail else None
        return bool(s)

    methods = [m for m in METHOD_ORDER if any(t["method"] == m for t in trials)]
    methods += sorted({t["method"] for t in trials} - set(methods))
    tasks = [t for t in TASK_ORDER if any(x["task"] == t for x in trials)]
    tasks += sorted({x["task"] for x in trials} - set(tasks))

    table = {}
    for m in methods:
        row = {"label": METHOD_LABEL.get(m, m), "tasks": {}, "unknown": 0}
        succ_all = n_all = 0
        srs = []
        for task in tasks:
            ts = [t for t in trials if t["method"] == m and t["task"] == task]
            outs = [outcome(t) for t in ts]
            known = [o for o in outs if o is not None]
            row["unknown"] += sum(o is None for o in outs)
            k, n = sum(known), len(known)
            row["tasks"][task] = {"k": k, "n": n, "sr": (k / n) if n else float("nan"),
                                  "stages": dict(sorted(
                                      ((t["failure_stage"] or "?", 1) for t in ts if outcome(t) is False)
                                  ))}
            stages = defaultdict(int)
            for t in ts:
                if outcome(t) is False:
                    stages[t["failure_stage"] or "?"] += 1
            row["tasks"][task]["stages"] = dict(stages)
            succ_all += k
            n_all += n
            if n:
                srs.append(k / n)
        row["macro_sr"] = sum(srs) / len(srs) if srs else float("nan")
        row["pooled"] = (succ_all, n_all)
        row["ci"] = wilson(succ_all, n_all)
        table[m] = row

    # 짝지은 McNemar / 태스크별 Fisher (Ours 대비)
    if "full" in table:
        ours = {(t["task"], t["seed"], t["rep"], t["provider"]): outcome(t)
                for t in trials if t["method"] == "full"}
        for m in methods:
            if m == "full":
                continue
            b = c = paired = 0
            for t in trials:
                if t["method"] != m:
                    continue
                key = (t["task"], t["seed"], t["rep"], t["provider"])
                o, u = ours.get(key), outcome(t)
                if o is None or u is None:
                    continue
                paired += 1
                b += int(o and not u)
                c += int(u and not o)
            table[m]["mcnemar"] = {"paired": paired, "ours_only": b, "variant_only": c,
                                   "p": mcnemar_exact(b, c)}
            pv = {}
            for task in tasks:
                a = table["full"]["tasks"][task]
                d = table[m]["tasks"][task]
                pv[task] = fisher_exact(a["k"], a["n"] - a["k"], d["k"], d["n"] - d["k"]) \
                    if a["n"] and d["n"] else float("nan")
            table[m]["fisher_p"] = pv
            table[m]["fisher_p_holm"] = holm(pv)
    return {"methods": methods, "tasks": tasks, "table": table, "n_trials": len(trials)}


def fmt_p(p):
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return "–"
    return f"{p:.3g}" if p >= 1e-4 else f"<1e-4"


def write_outputs(res: dict, out: Path, trials: list[dict]) -> None:
    out.mkdir(parents=True, exist_ok=True)
    tasks, methods, table = res["tasks"], res["methods"], res["table"]
    head = "| Method | " + " | ".join(t.upper().replace("_", "-") for t in tasks) + \
           " | Macro SR | Pooled | 95% CI | McNemar p (vs Ours) |"
    lines = ["# Table III — Task success rate", "",
             f"trials={res['n_trials']}  (success=None 인 trial 은 n 에서 제외; 표 아래 unknown 열 참고)", "",
             head, "|" + "---|" * (len(tasks) + 5)]
    for m in methods:
        r = table[m]
        cells = []
        for t in tasks:
            c = r["tasks"][t]
            cells.append(f"{c['sr']:.1f} ({c['k']}/{c['n']})" if c["n"] else "–")
        k, n = r["pooled"]
        lo, hi = r["ci"]
        mc = r.get("mcnemar")
        lines.append(f"| {r['label']} | " + " | ".join(cells) +
                     f" | {r['macro_sr']:.3f} | {k}/{n} | [{lo:.2f}, {hi:.2f}] | "
                     + (f"{fmt_p(mc['p'])} (n={mc['paired']})" if mc else "–") + " |")
    lines += ["", "## Task-level Fisher exact p (Holm-corrected) vs Ours", ""]
    for m in methods:
        if "fisher_p_holm" not in table[m]:
            continue
        lines.append(f"- {table[m]['label']}: " + ", ".join(
            f"{t}={fmt_p(table[m]['fisher_p_holm'][t])}" for t in tasks))
    lines += ["", "## Failure stage breakdown (failed trials)", ""]
    for m in methods:
        for t in tasks:
            st = table[m]["tasks"][t]["stages"]
            if st:
                lines.append(f"- {table[m]['label']} / {t}: " + ", ".join(f"{k}={v}" for k, v in st.items()))
    unk = {m: table[m]["unknown"] for m in methods if table[m]["unknown"]}
    if unk:
        lines += ["", "## success=None (goal evidence missing) — excluded from n", ""]
        lines += [f"- {METHOD_LABEL.get(m, m)}: {n}" for m, n in unk.items()]
    (out / "table3.md").write_text("\n".join(lines), encoding="utf-8")
    (out / "table3.json").write_text(json.dumps(res, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    with (out / "trials.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=list(trials[0].keys()) if trials else ["method"])
        w.writeheader()
        w.writerows(trials)
    print("\n".join(lines))
    print(f"\n[table3] -> {out / 'table3.md'}")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--root", nargs="+", type=Path, default=[ROOT / "output" / "ablation"])
    p.add_argument("--out", type=Path, default=ROOT / "output" / "table3")
    p.add_argument("--unknown-as-fail", action="store_true",
                   help="success=None(목표 증거 부족) 을 실패로 집계")
    args = p.parse_args(argv)
    trials = load_trials([r.resolve() for r in args.root])
    if not trials:
        print("[table3] run_result.json 을 찾지 못했습니다:", [str(r) for r in args.root])
        return 1
    write_outputs(build(trials, args.unknown_as_fail), args.out.resolve(), trials)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
