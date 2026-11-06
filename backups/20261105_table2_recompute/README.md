# table2_recomputed 재생성 백업 (2026-11-05)

`scripts/table2_recompute.py`를 현재 입력(docs/*coverage_validation*.json, vac_gt_admissibility.json)으로
재실행하여 docs/table2_recomputed.{md,json}을 갱신하기 전의 원본.

구버전(단일 "요약" 표 형식)이며 c1_1 2F 링키지와 c2_2 turkey/cheese id 매칭이 반영되기 전 결과다.

되돌리기:
    cp backups/20261105_table2_recompute/table2_recomputed.md docs/table2_recomputed.md
    cp backups/20261105_table2_recompute/table2_recomputed.json docs/table2_recomputed.json
