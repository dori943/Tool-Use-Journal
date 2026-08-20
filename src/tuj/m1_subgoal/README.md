# M1 (Subgoal Decomposition) — 해야 할 것 / M2 인터페이스

## M1의 책임 (이것만 하면 됨)

1. **입력**: 태스크 자연어 + `output/<task>/m0.json`(Tier-1: 노드 id·class·center·bbox·coarse 관계) + `configs/robot_spec.json`
2. **출력**: `output/<task>/m1.json` — 서브골 리스트(EE 무관 논리 단위) + 수치 필요 술어를 질의로 기술
3. 수치가 필요한 술어(Type-B)는 **직접 계산하지 말고 질의로 요청** — M2가 실행 후 `output/<task>/m2.json`으로 응답

## 파일 계약 (모든 모듈 출력은 output/<task>/ 아래)

### M1 → M2: `m1.json` — `m1_queries` 리스트를 M2가 실행 (M1 실제 스키마 반영)

```json
{"m1_queries": [
  {"subgoal_id": "SG1", "queried_by": "SG1_d1_p2",
   "m2_call": {"kind": "ee", "node_id": "obj_plate_light_plate"}},
  {"subgoal_id": "SG1", "queried_by": "SG1_d1_p0",
   "m2_call": {"kind": "intrinsic", "node_id": "obj_plate_heavy_plate"}},
  {"subgoal_id": "SG1", "queried_by": "SG1_d3_p2",
   "m2_call": {"kind": "relational", "a": "obj_plate_light_plate",
               "b": "obj_zone_collection_zone_visual", "relation": "distance"}}
]}
```

- `queried_by` = **술어 id** → 응답과 G_k에 각인 (M5 역추적 근거, 필수)
- `kind` ∈ `intrinsic`(node_id) / `ee`(node_id) / `relational`(a, b, relation ∈ distance|fits_inside|clearance)
- 같은 술어를 여러 후보에 걸어도 됨 (예: `SG1_d1_p2` × light/heavy) — 응답은 리스트라 안 덮임
- 객체 지칭은 **m0.json의 `nodes[].id`** 그대로. M0에 없는 id는 `{"error": ...}` 응답으로 돌아옴

### M2 → M1: `m2.json` — `{"responses": [응답, ...]}`. 각 응답에 subgoal_id + queried_by + 대상 id 포함:

```json
// relational
{"queried_by": "SG1_d3_p2", "from": "obj_plate_light_plate",
 "to": "obj_zone_collection_zone_visual", "value_mm": 312.4, "check": "distance", "pass": true}

// intrinsic
{"queried_by": "SG1_p3", "node_id": "obj_block_block_0",
 "geometry": {"extents_mm": [21.0, 20.7, 12.2], "cylinder_like": false, "surface_rms_mm": 0.4},
 "material": "plastic", "density_kgm3": 1110.3, "mass_kg": 0.005,
 "mass_range_kg": [0.003, 0.008], "mu": {"mu": 0.34, "stage": 0}, "confidence": 0.47}

// ee
{"queried_by": "SG1_p1", "node_id": "obj_plate_light_plate",
 "ee": {"2F":  {"feasible": true,  "margin": 0.63, "margin_unit": "ratio", "reason": "ok"},
        "3F":  {"feasible": true,  "margin": 0.41, "margin_unit": "ratio", "reason": "ok"},
        "vac": {"feasible": false, "margin": -0.24, "margin_unit": "ratio", "reason": "mass_lt_payload"}},
 "reachability": {"reachable": true, "margin_mm": 240.0}}
```

실행: `python scripts/run_m2_c1_1.py` 가 m1.json을 읽어 질의 실행 → m2.json + 서브골별 `gk_<SG>.json` 생성.
(m1.json이 아직 없으면 내장 데모 질의로 폴백 — 위 스키마 그대로 예시 출력됨)

## 파이썬으로 직접 호출할 수도 있음 (파일 왕복 없이)

```python
from tuj.m2_grounding import Materializer, new_gk
mat = Materializer(m0); gk = new_gk("SG1")
r = mat.query_relational(gk, "obj_A", "obj_B", "distance", queried_by="SG1_d3_p2")
# 반환 스키마는 위 m2.json과 동일
```

## 규약 3줄

- **객체 지칭은 m0.json의 노드 id로만.** 점군·이미지 직접 접근 금지.
- **같은 객체 재질의는 공짜** (M2가 캐시) — 중복 걱정 말고 술어마다 질의.
- `|margin| < ε` 처리(재관측·프로브)는 M2 내부 책임 — M1은 받은 값을 그대로 신뢰하면 됨.
