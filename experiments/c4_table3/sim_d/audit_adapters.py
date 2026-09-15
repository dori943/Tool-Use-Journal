"""Write a human-readable audit of the six D_SIM condition adapters."""
from __future__ import annotations

import argparse
from pathlib import Path

from adapters import adapter_audit


def render(rows: list[dict]) -> str:
    lines = [
        "# D_SIM condition adapter audit",
        "",
        "이 문서는 adapter 인터페이스 연결 상태를 기록한다. `NEEDS_PROVIDER`는 코드 경로가 있으나 실제 model/provider 설정과 prediction 실행이 아직 없다는 뜻이다.",
        "",
        "| condition | status | active modules | prediction source | supported metrics | GT channel |",
        "|---|---|---|---|---|---|",
    ]
    for row in rows:
        lines.append("| {condition} | {status} | {modules} | {source} | {metrics} | {gt} |".format(
            condition=row["condition"], status=row["status"], modules=", ".join(row["active_modules"]),
            source=row["prediction_source"], metrics=", ".join(row["supports"]), gt=row["gt_leakage"]))
    lines += [
        "",
        "## 규칙",
        "",
        "- `+ Geometric grounding`과 `Ours`의 Mass Acc는 SiPhy mass prediction cache를 공유하며 추가 model call을 하지 않는다.",
        "- SiPhy mass cache가 없으면 shared adapter는 실패하고 새 호출을 조용히 만들지 않는다.",
        "- GT 필드가 observation에 들어오면 `GT_LEAKAGE_DETECTED`로 중단한다.",
        "- GT numerics는 `OracleAdapter(oracle=True)`로만 생성하며 일반 provider registry에 등록하지 않는다.",
        "- 현재 상태는 end-to-end prediction 완료가 아니므로 condition matrix의 `NEEDS_IMPLEMENTATION`을 유지한다.",
    ]
    return "\n".join(lines) + "\n"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.write_text(render(adapter_audit()), encoding="utf-8")
    print(args.output)


if __name__ == "__main__":
    main()

