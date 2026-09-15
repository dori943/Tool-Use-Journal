# Table 3 조건별 평가 규약 — 사전 확정 (2026-09-14)

이 문서는 **Table 3 평가 규약과 현재 실행 상태**다. 정지 Static Real-5 R 패널은 실행 및 독립 집계를 완료했고, D 패널과 구현되지 않은 조건은 여전히 차단되어 있다. `condition_matrix.csv`의 상태는 점수가 아니라, 정확히 그 조건·지표를 평가할 준비 상태다. 수치 정의의 기계 판독본은 `metric_registry.yaml`이다. 여기서 `-`는 `NOT_APPLICABLE`(조건의 구조적 미지원), `TBD`는 `NEEDS_DATA` 또는 `NEEDS_IMPLEMENTATION`(미완료)을 뜻한다. 현재 R 수치는 `output/c4_table3/20260914T160924Z_static_real5_evaluate/`의 정지 이미지 평가 산출물에서 재현한다.

## 현재 근거와 결과 연결 감사

- [C4 Real-5 보고서](../c4_real5/RESULTS.md), [GT](../../configs/c4_real5_gt.json), [manifest](../../data/external/kandukuri_ev_realphys/manifest.json), [추론기](../c4_real5/run_inference.py), [평가기](../c4_real5/evaluate.py)를 조사했다. Real-5는 Kandukuri 등의 **Table 5 실측값**을 공식 EV-RealPhys `test_sliding`의 동일한 5개 객체에만 대응시킨다. Table 4 합성값, `scene_properties.json`의 물리값, 일반 YCB-V 영상에는 적용하지 않는다. Tuna Fish Can은 제외한다.
- manifest 파일 SHA-256은 `07752e7d89b244db2210041d8fe308a2761d0a99dc166b98210ad4198ab6b8ee`이고, 기록된 원본 압축 파일 SHA-256은 `4c484ec1e5f3f66ccb6a930437404ba7c67a1bb5590791027ced9741e1136eab`이다. GT 파일 SHA-256은 `4a0f66e54e8676abc16ca3634fed7564cc2ad2cf9da737a858540d35950433b3`이다.
- [새 prediction](../../output/c4_real5/20260914T025828Z/predictions.json)의 SHA-256은 `043f16ef5ebc90c6b45d5a8ee3bb4db8d3ae8853d7d19abe0c6b3412e86f63a6`, [result](../../output/c4_real5/20260914T030346Z/result.json)의 SHA-256은 `e932b7cd2350f8718d11af7b0280e955f03e92eca034692256f5a5ad2386a56e`이다. prediction은 archive SHA, `gemini-3.6-flash`, temperature 0, seed 100(**provider에 미전달**), prompt SHA, RGB-D adapter SHA, friction tracker SHA, 24개 예측·1개 제외·0개 추론 실패를 기록한다. prompt와 adapter/tracker SHA는 현재 코드와 일치한다.
- result의 **모든 객체별 수·중앙값·GT**와 두 macro 점수를 위 prediction으로 재계산하면 정확히 일치한다: Mass MnRE `0.9227010043238545`, Friction MAE-Real5 `0.02410000000000001`. 대조군인 [옛 prediction](../../output/c4_real5/20260914T024835Z/predictions.json)(SHA `f5173cdb5334073f9d63fa05611618f86ea4f26b84f9349b10b021350be98ee0`)은 Mass MnRE `0.8952902593856016`을 재현해 새 result와 구별된다. 옛 friction 점수는 우연히 같은 `0.0241`이므로 그것만으로 연결하면 안 된다.
- 새 prediction의 24개 `sequence_id/object_id/selected_frame`은 manifest와 모두 일치하고 24개 crop 경로가 존재한다. 1개 제외를 합치면 manifest의 25개 시퀀스와 정확히 같으며 중복은 없다. 다만 manifest 파일의 SHA와 crop별 SHA는 prediction에 봉인돼 있지 않다.
- **판정: 수치적 집계 관계는 확인, 독립적인 실행 계보 증명은 미완료.** `result.json`에는 입력 prediction/manifest의 경로·SHA, model, seed, temperature, prompt/adapter SHA가 없다. 저장 경로도 별개다. 따라서 해당 점수를 여섯 Table 3 조건 중 어느 하나의 완전 실행 결과로 배정하지 않는다. 이전 중앙값 기반 결과·파일은 변경하지 않는다. 향후 scorer는 입력 파일 SHA, manifest/GT SHA, 설정과 코드 SHA, condition ID, 포함·제외 목록, 집계 규약 버전을 새 결과에 함께 기록하고 불일치 시 중단해야 한다.

## 평가 패널과 관측 경계

**R: Real-5 물성 패널.** 공식 5개 객체×5개 `test_sliding` RGB-D 시퀀스, GT는 Table 5의 mass(kg)와 combined object–table friction(무차원)이다. 현재 채워진 Table III 결과는 `Static Real-5` 변형으로, 시퀀스당 시간적으로 떨어진 정지 프레임 2개(객체당 10장)를 사용한다. 질량은 RGB-D crop/점군, 마찰은 객체와 지지 테이블이 함께 보이는 정지 context의 visual prior다. 기존 자유 활주 30 Hz 경로와 점수는 trajectory auxiliary로 보존한다. `source_trial_count=10`은 원 논문의 기울인 테이블 마찰 GT 측정 횟수이지 inference 반복 횟수가 아니다. `000017` 대체를 포함한 static 입력 QC는 예측 전에 잠갔다.

**D: 별도 조작·기하 패널(미확보).** Mass Acc, Suction Acc/PF, Clearance RelErr, Feasibility Acc, DA, Crit에 필요한 동일 물리 객체·동일 task/EE 후보·두 고정 pose·독립 GT를 담는다. Real-5에는 두 pose의 흡착 결과, 개구부/clearance, 독립 EE 실행 성공·선택 GT가 없다. D의 객체가 R과 다르면 GT를 공유하거나 두 패널 점수를 한 분모로 섞지 않는다. Table 3 각 열에 패널과 유효 객체·sample 수를 명기한다.

현재 dataset inspector는 `scene_properties.json`의 객체 ID와 `scene_gt.json`의 모든 프레임 ID/pose로 **동일 객체와 품질을 검증**한다. 특히 GT pose를 투영한 중심 ROI의 선명도로 질량용 프레임을 선택했으므로, GT pose가 추론기 내부에는 들어가지 않았어도 **입력 선택 단계에는 영향을 주었다**. 추론기의 RGB-D 분할 마스크는 depth 평면/RGB GrabCut에서 생성하며 공개 GT mask는 없다. 마찰 궤적은 RGB-D centroid와 카메라 보정으로 만들고 GT MoCap pose를 읽지 않는다. `SiPhyBackend.estimate(crop, object_id, points_mm)`에 전달한 `object_id`는 현재 구현에서 오류 메시지용 힌트일 뿐 VLM 프롬프트에는 삽입되지 않지만, 장래 조건 비교에서는 class/name의 출처를 모든 조건에 동일하게 기록한다. 후속 D/R 비교는 **고정 프레임 번호 또는 영상만 이용한 선택 규칙**을 사전에 잠가 GT pose를 입력 선택에서 제거해야 한다. GT mask/annotation/`scene_properties`의 물리 수치는 scoring/identity audit 전용으로 격리한다.

## 여섯 조건의 정보·모듈 계약

[`condition_matrix.csv`](condition_matrix.csv)의 여섯 조건×아홉 지표 **54개 셀은 모두 네 상태 중 하나로 분류**했다. 현재 검증된 R 패널에서는 SiPhy Mass MnRE, + Geometric grounding Mass MnRE(동일 SiPhy 예측 공유), Ours full Mass MnRE 및 Static Real-5 Friction MAE가 `READY`다. D 패널의 독립 GT와 Name-only/Affordance adapter가 없으므로 나머지는 `NEEDS_DATA`, `NEEDS_IMPLEMENTATION` 또는 `NOT_APPLICABLE`로 유지한다. `NEEDS_DATA`는 데이터가 우선 장애인 셀(구현도 필요할 수 있음), `NEEDS_IMPLEMENTATION`은 필요한 패널 데이터는 있지만 정확한 조건 연결이 없는 셀이다.

모든 조건은 같은 잠긴 sample 목록, 원시 관측, EE 목록·사양([`robot_spec.json`](../../configs/robot_spec.json)), task/DAG, 판단 함수 버전, 출력 schema와 결측 정책을 사용한다. 조건 게이트가 **소비할 수 있는 정보만** 달라진다. train-only 전역 수치 prior(질량, 형태/평면성, gripper 접촉 μ)는 모든 조건에 동일한 버전으로 제공하고, 해당 조건이 관측 기반 값을 내면 그 채널만 교체한다. prior는 test GT/`scene_properties`에서 적합하지 않는다. 단독 prior 자체는 관측 기반 물성 추정치로 보고하지 않는다. 메모리 재사용(`run_m3.py`의 M0), `MockBackend`, test GT 조회와 조건별 손수 보정은 모두 끈다.

| 조건 | 허용 입력·추가 관측 | 활성 모듈과 수치/범주 출력 | 공통 규칙에 넘기는 값 |
| --- | --- | --- | --- |
| Name-only | 공통 객체 이름/ID만. 영상·depth·pose·시퀀스·test 속성 금지 | 잠긴 train-only 이름 prior; `mass_prior_kg`, 기본 기하/접촉 prior. 관측 기반 마찰·clearance 출력 없음 | 수치 prior와 미지 항목 상태; 보수적 결측 처리 |
| + Affordance labels | Name-only + **train-only, pose 비의존** `2F_graspable/3F_graspable/suctionable` 범주 레이블 | 동일 이름 prior, 별도 레이블 gate; 질량 prior는 Name-only와 동일. 레이블을 kg·μ·mm 또는 test GT로 변환 금지 | 동일 수치 prior + 레이블 허용 EE 집합 |
| SiPhy (as adopted) | 공통 단일 프레임의 검정 배경 RGB crop와 **동일 프레임의 RGB-D 점군**. 전체 활주 관측 금지 | 기존 `SiPhyBackend`의 material/density/thickness→`mass_kg` shell 적분. 현재 코드에서 점군 없이는 `mass_kg=None`이므로 이 내부 점군만 허용; 기하/공간 판정으로는 전파하지 않음 | SiPhy 질량 + 공통 기하·접촉 prior |
| + Geometric grounding | SiPhy와 **같은** 단일 RGB-D 프레임. 추가 카메라/GT pose 금지 | 기존 M1 `points_from_frame`, `geometry_from_node`, `seal_patch_rms_mm`, bbox 기반 `coarse_clearance`/`relations`; 질량은 동일 SiPhy. 관측 기하·평면성·거친 clearance 출력 | SiPhy 질량 + 관측 기하; 접촉 μ는 공통 prior. Real-5 table μ 출력 없음 |
| Ours (full: + friction, + spatial) | Static 변형에서는 공통 질량 프레임과 객체·테이블 context; trajectory auxiliary는 별도 입력 | 정지 visual combined object–table μ prior. `FrictionHead.stage0`/`probe_mu_from_track`는 static 점수에 사용하지 않음 | 질량과 static visual friction을 보고하며, trajectory 결과와 결합하지 않음 |
| GT numerics | Ours와 같은 허용 관측; scorer가 보관한 **독립 정답 수치만** 명시 채널에 주입 | 독립 scale mass, calibrated geometry/clearance, 별도 측정 μ_contact/μ_table이 존재할 때만 교체. 범주 affordance/최종 EE 정답은 주입 금지 | 동일 downstream 함수에 정답 수치 + 변하지 않은 비수치 입력. 누락된 정답 수치를 test annotation이나 동일 규칙의 출력으로 대신하지 않음 |

`run_m3.py`→`Materializer.query_intrinsic/query_ee`→`ground_intrinsic`→`evaluate_ee`와 M1 출력 규약을 재사용한다. **단일 조건 어댑터**가 공통 schema로 값을 정규화하고, 각 조건의 channel gate만 바꾼다. `evaluate_ee`의 2F/3F 폭·payload·grip force, vacuum 평면성·payload·seal 접촉, 동일 `reach_check`와 spatial relation, M4 후보 선택/제약/tie-break는 조건에 따라 바꾸지 않는다. 미지 숫자에 GT를 보충하거나 임의 0을 넣지 않고 abstain/fail-closed로 기록한다. 기존 `evaluate_ee`는 `geometry`, `mass_kg`, `mu.mu`를 필수로 받으므로 prior/UNKNOWN을 구별하는 어댑터와 공통 결측 분기가 필요하다. `robot_spec.json`의 vacuum payload 0.5 kg, 3F 2.5 kg, 2F 5.0 kg, vacuum seal 직경 30 mm와 RMS 한계 1.5 mm를 설정 해시와 함께 고정한다.

**마찰 의미 주의.** `evaluate_ee.grip_slip`의 μ는 손가락–객체 접촉에 대한 가정치다. R의 GT와 RGB-D 자유 활주 μ는 객체–테이블 결합값이므로 절대 `grip_slip`에 공급하지 않는다. 독립 μ_contact 실측치가 없으면 그 채널의 oracle 및 friction-caused EE 개선은 평가 불가다. 원 논문과 달리 본 저장소의 `FrictionHead` probe는 현재 M3 EE 경로에 연결되지 않았다.

**Affordance 출처/유출 감사.** 현 M1/M3에는 `+ Affordance labels`용 독립 파일·loader·조건 스위치가 없다. M2의 `tool_candidate_ids`는 계획 산출물이고, M5의 `StaticToolAffordanceProvider`는 제공된 접촉 patch, `CircularPlateAffordanceProvider`는 world geometry를 사용하므로 여기서 필요한 name-only 추가 레이블로 간주할 수 없다. 레이블은 훈련 객체/훈련 장면만 본 독립 작성자가 고정 ontology로 작성하고 파일 SHA, 작성 시점, 객체 분할, 금지 자료 접근 감사를 남긴다. test의 `scene_gt`, mask, 실측 mass/μ, 물리 `scene_properties`, suction/EE 성공 라벨, test 물체별 사후 조사로 레이블을 만들면 유출이다. 테스트 객체와 이름이 train에 중복되면 객체별 수치/성공 라벨이 넘어오지 않도록 물리 인스턴스 수준 분할과 감사가 필요하다.

## 공통 평가·집계·실패 규칙

- 평가 대상 ID와 D의 두 pose, R의 질량 프레임·마찰 시퀀스, 입력 QC 제외 기준은 **조건 결과 확인 전** 별도 잠긴 manifest로 고정한다. 입력 자체 결함으로 사전 제외한 ID는 모든 조건에서 함께 제외하고 사유/분모를 공개한다. 조건별 실패 때문에 분모를 줄이지 않는다.
- 연속량 주 집계: 동일 객체의 유효 R 시퀀스 **예측 중앙값 → 객체 GT와 오차 → 객체 동일 가중 평균**. 이것이 현재 C4 Real-5의 `Mass MnRE`/`Friction MAE-Real5` 정의다. 다른 순서(`sequence error 평균`, `객체별 평균 예측`, 모든 프레임 micro 평균)는 각각 `sensitivity_sequence_mean_error`, `sensitivity_object_mean_prediction` 등 별도 이름으로만 보고한다. D clearance는 GT가 관계 sample마다 달라지므로 **sample 오차 → 객체별 오차 중앙값 → 객체 macro 평균**을 쓴다.
- 분류 주 집계: 미리 정의된 sample×EE(또는 pose, pose-pair)를 분모로 객체별 정답 비율을 낸 뒤 객체 동일 가중 평균. 불확실/abstain/모듈 실패는 해당 평가 단위에서 오답이다. 독립 GT가 없는 단위는 실행 전 데이터 부족으로 선언하며 결과를 본 뒤 제외하지 않는다. 연속량에서 조건별 missing/NaN이 잠긴 대상에 생기면 임의 벌점이나 완전 사례 평균을 만들지 않고 해당 cell을 `TBD`로 보류하며 coverage·사유를 공개한다. 기존 24/25 결과는 별도의 역사적 부분 실행으로 유지한다.
- 향후 재실행 수는 조건마다 같은 입력으로 **K=3**, run ID `0,1,2`로 잠근다. seed 지원 provider는 `100,101,102`를 사용하고 실제 전달 여부를 저장한다. Gemini처럼 미지원이면 seed는 전달하지 않고 비결정성을 기록한다. run별 객체 macro 점수의 평균을 주 표시값으로, run 간 표본 표준편차는 `std(ddof=1)`로 병기한다. 한 run이면 반복 std는 `N/A`다. 객체 간 오차의 `std(ddof=1)`는 개별 run에서의 **이질성 기술통계**이며 실행 안정성이나 95% 신뢰구간이 아니다. 두 std를 합치지 않는다.
- 질량 단위 kg, 기하/clearance mm, μ와 MnRE/RelErr 무차원, 분류 정확도 [0,1]을 원시 저장하고 표에서는 %로 표시한다. 모든 macro 평균은 객체 균등 가중이며 R과 D 또는 EE 유형의 수를 몰래 가중치로 합치지 않는다. 객체별 유효 수, 전체 분모, 조건별 coverage, 사용된 GT·코드·설정 해시를 함께 기록한다.
- **Crit은 미래 D 결과를 보기 전에 여기서 고정한다.** 물리량은 mass, EE는 `vac`, 임계값은 `robot_spec.json`의 payload **0.50 kg**, 주변 구간은 실측 `|m_gt−0.50| ≤ 0.05 kg`(경계 포함)이다. 그 subset의 `m < 0.50 kg` 판정 정확도만 계산한다. 현재 Real-5에는 이 구간의 GT 객체가 없으므로 0/0을 100%로 쓰지 않고 D에 근접 사례가 확보될 때까지 `NEEDS_DATA`다. 다른 마찰/clearance 임계 subset은 Crit에 섞지 않는다.
- Feasibility GT는 동일 `evaluate_ee` 부등식을 GT 수치에 재실행해 만든 값이 아니라 독립된 EE 시도/기하·collision·물리 검증이다. DA GT는 사전에 정한 허용 EE 집합 `A_s`다. 복수 EE가 실제 가능하면 `selected_ee ∈ A_s`가 정답이며 임의의 한 EE를 지정하지 않는다. `A_s=∅`이면 정확한 `NO_FEASIBLE_EE` 반환만 정답이다. 동일 후보·DAG와 M4 결정 규칙을 모든 조건에 적용한다. Oracle도 같은 평가와 실패를 통과해야 하며 직접 주입한 mass/μ/clearance의 산술 오차 0을 추정 성능이라 부르거나 Feas/DA/Suction을 자동 100%로 채우지 않는다.

## 기존 코드와의 변경점

기존 `scripts/eval_m3_mass_c1_1.py`는 시뮬레이터 subtree 질량과 비교해 객체별 상대오차를 평균하고, **추정 intrinsic의 질량만 GT로 치환한 뒤 동일 `evaluate_ee`로 feasible 집합이 완전 일치하는지**를 `decision_agree`로 계산한다. 이는 여기서 정의하는 **EE별 Mass Acc**(각 payload 이진 판정), **Feasibility Acc**(sample×EE 독립 GT), **DA**(허용 최종 선택 집합)가 아니다. 기존 값은 대체·개명 없이 보존하고 Table 3 수치로 이식하지 않는다. 또한 그 스크립트의 XML geom friction은 Real-5 combined object–table GT가 아니며 논문도 주의 문구를 출력한다. 기존 C4 evaluator는 예측이 하나라도 있는 객체의 중앙값을 쓰고 제외된 1개를 남긴다. 새 scorer의 사전 QC/조건 실패 정책과 계보 해시 저장은 추가 구현 사항이다.

## 우선순위별 부족 사항

1. **P0 데이터:** D 패널의 고정 객체/두 pose ID, 독립 suction trial GT(두 pose 모두), 실측 질량의 payload 근접 사례, 독립 개구부·충돌 clearance mm, sample×EE 물리/기하 검증 GT와 허용 선택 집합을 확보한다. R의 5개 물리 객체 GT를 다른 YCB 사진·실물에 이식하지 않는다.
2. **P0 데이터/유출:** Name-only 수치 prior와 Affordance label을 test GT와 분리한 훈련 출처로 작성·동결하고 인스턴스 중복 및 M0 memory 오염을 감사한다. μ_contact GT가 없으면 Oracle 접촉 채널 및 friction→grip-slip 결정 주장은 보류한다.
3. **P0 구현:** 기존 M1/SiPhy/M3/M4를 호출하는 **하나의 조건 어댑터**와 공통 스키마, prior/UNKNOWN/abstain 분기, object-table과 gripper-contact μ의 별도 채널, 동일 입력 manifest·EE 사양·후보·예산의 해시 검증을 만든다. 기존 GT pose 기반 프레임 선택을 조건 공통의 영상 전용 규칙으로 교체한다.
4. **P0 구현:** 아홉 지표의 독립 GT loader/scorer와 fixed-denominator failure/coverage 처리, per-object→macro→repeat 집계, `prediction_sha256`/manifest·GT·config·prompt·code SHA를 result에 남기는 계보 검증을 추가한다. Real-5 과거 결과 파일은 수정하지 않는다.
5. **P1 품질:** R의 `000015` 등 crop 배경 잔여와 `000017` 추적 단절을 GT 미참조 기준으로 점검한다. D의 두 pose 및 near-payload 수를 결과 전에 충족시키고 run 간 std와 객체 간 std를 별도로 보고한다.
