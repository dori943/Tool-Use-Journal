# C4 Static Real-5 조건 adapter 감사

정지 평가용 구현을 기존 Table 3 궤적 평가와 구분한다. `infer.py`는 evaluator/GT loader를 import하지 않고 숫자 GT 파일명을 런타임 파일 접근 경계에서 거부한다. `evaluate_static.py`만 prediction 완료 후 `evaluator_only_gt.yaml`을 읽는다. GT·과거 prediction·`scene_properties` 물리 값은 prompt, crop, 선택, 파싱, prior에 사용하지 않았다.

| 조건 | 실제 입력 | 추정 모듈·prompt | 질량 | 정지 시각 마찰 | downstream |
|---|---|---|---|---|---|
| Name-only | 이름만 허용 | train-only 이름 prior 미구현 | NEEDS_IMPLEMENTATION | STRUCTURALLY_UNSUPPORTED | D 패널 미연결 |
| + Affordance labels | 이름+훈련 출처 범주 레이블 허용 | 레이블 파일/출처 미확보, prior 미구현 | NEEDS_IMPLEMENTATION | STRUCTURALLY_UNSUPPORTED | D 패널 미연결 |
| SiPhy (as adopted) | black-background mass RGB crop + 동일 프레임 M1 RGB-D 점군 | 기존 `SiPhyBackend.estimate`, `SYS_MSG` | READY (`siphy_static_rgbd_mass`) | STRUCTURALLY_UNSUPPORTED | D 패널 미연결 |
| + Geometric grounding | SiPhy와 같은 정지 프레임·crop·점군 | 질량은 **동일 SiPhy prediction 공유**, 추가 M1/M3 기하는 이 두 지표의 입력 아님 | SHARED_BACKEND | STRUCTURALLY_UNSUPPORTED | M1/M3 기하 코드 존재, Table 3 공통 decision adapter 미연결 |
| Ours (full) | 위와 같은 질량 입력; 동일 프레임의 객체+접촉 테이블 context RGB | 질량 SiPhy prediction 공유; 새 `FRICTION_PROMPT`+VLM JSON | SHARED_BACKEND | READY (`visual_context_combined_friction`) | M3/M4 spatial/EE 공통 adapter 미연결 |
| GT numerics | evaluator-only 실측 숫자 | estimator 아님; downstream oracle | STRUCTURALLY_UNSUPPORTED | STRUCTURALLY_UNSUPPORTED | D 패널 GT/adapter 없음 |

질량의 SiPhy/Geometric/Ours 세 칸은 하나의 backend와 완전히 같은 입력을 쓰므로 독립 실험 또는 세 번의 API 호출로 세지 않는다. `shared_prediction_source=siphy_static_rgbd_mass` 하나를 참조한다. Ours의 static visual friction은 비디오 감속도로 측정한 값이 아니며 불확실한 시각 prior를 Table 5 실측 combined object-table μ와 비교한다. 기존 `FrictionHead.probe_mu_from_track` 결과는 auxiliary만 유지하며 새 prompt·결과에 혼합하지 않는다. 이 μ는 gripper–object contact μ가 아니므로 `evaluate_ee.grip_slip`에 넣지 않는다.

`SYS_MSG`와 `FRICTION_PROMPT`의 SHA-256은 각 raw prediction에 기록한다. 모델/temperature/requested seed/실제 provider seed 지원 여부도 각 기록에 넣는다. Gemini seed는 API에 전달하지 않는다. temperature 0을 결정성 보장으로 주장하지 않는다. 현재 설정은 `gemini-3.6-flash`, temperature 0, 전체 반복 5회이며 이번 prep에서는 smoke만 실행한다.

Affordance labels는 저장소의 train-only 자료로 확인되지 않았다. 이 실물 객체의 실측 GT나 test 영상, 사후 성공 annotation으로 만들면 평가 GT 유출이므로 adapter를 구현/READY 처리하지 않았다.

## 잠긴 prompt와 smoke 근거

Mass SYS_MSG SHA-256: `c9d5e36da720bb37123a739e47cae0368b4498d7556ad23821fb85e46e0842c7`. Static visual-friction prompt SHA-256: `687f7253392fa351937c67618b550737c6c75a8f2d8018df09893479c6fbd767`. 입력 manifest SHA-256: `89c476f3afd9f00223fff928e6d86c124905ad53d928d70b5b95e8e9f8726dbc`. `smoke_test/raw_predictions_live.jsonl`에는 동일한 한 입력에 대한 실제 질량·마찰 API 응답 각각 1건, JSON 파싱 성공 2건과 숫자 GT 읽기 거부가 기록되어 있다. Smoke prediction은 본 평가의 50×5 입력에 포함되지 않는다. 이전 `gemini-2.5-flash` 호출은 지원 종료 HTTP 404였으며 기록을 보존했다. 본 5회 평가는 0회 실행했다.
