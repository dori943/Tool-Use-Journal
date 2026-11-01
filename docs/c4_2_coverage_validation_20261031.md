# C4_2 EE 3종 파지 검증 (2026-10-31)

> 공유 결과는 [전체 진행 요약](ee_gear_checkpoint_20261102.md)과 [조합별 결과](ee_gear_coverage_results_20261102.json)에 정리했다. 이 문서의 `output/` 경로는 로컬 보관 자료이며 GitHub 업로드에 포함하지 않는다.

18개 조합을 독립 시뮬레이션으로 검증했다. 14개 성공, 2개 접촉·충돌 실패,
2개 접근 경로 실패다. 구현 예외와 시간 초과는 없었다. 기존 미등록 12개 중
8개가 새로 성공했다. seed 0의 후보 동작별 결과이며, 최종 논문 성공률이나
해당 EE의 모든 파지 방법에 대한 가능·불가능 판정은 아니다.

| 물체 | 2F | 3F | 흡착 |
|---|---|---|---|
| 밀대 | 성공 | 성공 | 성공 |
| 바게트 | 성공 | 성공 | 성공 |
| 거품기 | 성공 | 접촉 불안정 | 장착부 충돌 |
| 시리얼 | 성공 | 성공 | 성공 |
| 우유 | 성공 | 성공 | 성공 |
| 뚜껑 | 접근 자세 IK 실패 | 접근 자세 IK 실패 | 성공 |

전체 물체 기준 관측값은 2F 5/6, 3F 4/6, 흡착 5/6이다. 흡착 성공 5건과
기존 바게트·거품기 2F 성공은 파지 중 부착 보조를 사용한다. 나머지 손가락
파지 성공 7건은 부착 보조 없이 유지 검증을 통과한 뒤 운반용 부착을 했다.

- 거품기 3F: CLOSE에서 유효 손가락 접촉을 유지하지 못했다. 마지막 기록의 접촉력은 0이다.
- 거품기 흡착: CLOSE에서 컵 장착부와 거품기 형상의 약 3.635mm 침투를 감지해 중단했다.
- 뚜껑 2F·3F: PRE_CLEARANCE의 전체 자세 IK를 찾지 못했다. 실제 물체 접촉 전에 끝났으므로 파지 가능 여부는 미판정이다.
- 실패한 네 후보는 다른 접근·접촉 자세를 추가로 검증할 여지가 남는다.
- 기존 C4_2 전체 작업의 필수 도구 연결 문제와 이번 파지 검증은 별도다.

결과: `output/c4_2-all-ee-20261031-b/c4_2/summary.json`.
영상과 초기·최종 상태는 같은 폴더의 각 물체_EE 디렉터리에 저장했다.
18개 영상에서 표본 프레임을 모두 읽을 수 있었고 검은 프레임은 검출되지 않았다.
뚜껑 손가락 두 조합의 짧은 영상은 접근 계획 실패에 따른 것이며, 중간 동작 누락이 아니다.


## Scope and provenance

Base: `6156e34ef4b8800459e15a7338bfb4e5e5622632` (`fix/dohee-ablation_study`).
Work branch: `fix/ee-gear-validation-20261102`.
This experiment checks grasp, lift and hold for six objects and three EEs.
It is separate from w/o Reasoner whole-task packing and its `MISSING_REQUIRED_TOOL:SG1` failure.

During the primary matrix, all twelve new recipes were validator-only probes.
After the follow-up below, eight successful alternatives were registered for
normal execution with identical recipe fields; four unsuccessful pairs remain
validator-only. Their failures cannot establish that every possible grasp with
that EE is impossible.

## Fixed conditions

- C4_2_DiagonalFitPacking, seed 0; fresh process/environment for each pair.
- Same six scene objects and existing scene assets. Save settled poses for comparison.
- No API/LLM calls; no changes to Reasoner estimates or feasible EE verdicts.
- Source object recipe's hold duration, minimum lift, maximum slip, contact thresholds,
  force targets and material settings are preserved. Probe modifications specify
  the EE, approach/contact pose and opening, with unassisted finger acquisition.
- Existing production recipes retain their original attachment policy.
- Vacuum uses the existing contact-gated kinematic attachment model. Some original
  finger recipes also attach during acquisition. Post-validation transport attachment
  is distinguished from attachment used to pass lift/hold.
- Controller timeout is 600 wall-clock seconds per pair and is classified separately.
- Coverage denominator is all six task objects. Inference-conditioned performance is
  a separate metric. This run uses `--denominator none` so stored M1 provenance cannot
  silently determine which pairs are attempted.
- `PASS` means this simulated motion meets the existing lift/hold evaluator; it is
  not whole-task success. `FAIL` means this motion/grasp failed. Code/runtime errors,
  timeouts and missing execution support are separate.

## Implementation changes

- All exit paths persist a per-case report; reports include initial scene/state and
  whether simulation and grasp execution started.
- Preserve driver exception types, failure stages, metrics and acquisition/transport
  attachment details instead of losing them behind a generic GraspFailure wrapper.
- Synthetic expected-size static warnings are separated from controller failures.
  Actual runtime grasp targets are saved in each `grasp/targets.json`.
- Run each pair in a separate process, checkpoint summaries after each case, and reject
  an existing output root/document. Manifest includes base commit, source hashes,
  tracked patch and newly added Python source files.
- Enable controller-tick rendering. The original validator's video callback alone
  produced only one initial frame plus a final one-second hold.

## Runs

1. `output/c4_2-registered-baseline-20261031-a`: original six registrations all passed.
   Videos contain only initial/final frames; numeric and state evidence is preserved.
2. `output/c4_2-extra-ee-probes-20261031-a`: stopped for recorder correction. Completed
   cases and interruption details are recorded in `interruption.json`; missing trials
   must not be counted as failures.
3. `output/c4_2-all-ee-20261031-b`: full 18-case rerun with controller-tick video.
   This is the primary matrix; preliminary runs are not pooled into it.

Command for the primary matrix:

```sh
PYTHONDONTWRITEBYTECODE=1 PYTHONPATH=src:. MUJOCO_GL=cgl \
/opt/anaconda3/envs/tool-use/bin/python scripts/verify_table2_coverage.py \
  --tasks c4_2 --all-ees --denominator none --video \
  --out-root output/c4_2-all-ee-20261031-b \
  --doc output/c4_2-all-ee-20261031-b.md
```

## Checks

`tests/test_table2_coverage.py`, `tests/test_c4_2_probe_recipes.py` and
`src/tuj/m5_motion/tests/test_scripted_grasps.py`: **171 passed** on the final targeted test run.

The existing `test_scripted_grasp_validate.py` has 19 passes and three failures
also reproduced in the unchanged prior worktree (`9a7c57f`): two tests expect
C3_2 plate aliases to resolve to the type name, and one assumes an obsolete vacuum
seating offset. These are pre-existing test/implementation expectation mismatches;
this work does not change plate behavior to satisfy them.

## Primary matrix results

| Object | 2F | 3F | vac |
|---|---|---|---|
| rolling_pin | PASS | PASS | PASS |
| baguette | PASS | PASS | PASS |
| whisk | PASS | FAIL | FAIL |
| cereal | PASS | PASS | PASS |
| milk | PASS | PASS | PASS |
| lid | FAIL | FAIL | PASS |

Completed cases: 18. All results are from the same primary run.
Maximum settled initial position difference across scene objects: 0.007028 mm.
Maximum settled initial orientation difference: 0.003605 degrees.

| Object | EE | Acquisition attachment | Phase | Hold seconds | Failure |
|---|---|---|---|---|---|
| rolling_pin | 2F | False | POST_VALIDATED_GRASP | 4.999999999988347 |  |
| rolling_pin | 3F | False | POST_VALIDATED_GRASP | 5.000000000002057 |  |
| rolling_pin | vac | True | ACQUISITION | 5.000000000006111 |  |
| baguette | 2F | True | ACQUISITION | 4.999999999988347 |  |
| baguette | 3F | False | POST_VALIDATED_GRASP | 5.00000000000156 |  |
| baguette | vac | True | ACQUISITION | 5.0000000000040465 |  |
| whisk | 2F | True | ACQUISITION | 4.999999999988347 |  |
| whisk | 3F | False |  |  | GRASP_CONTACT_NOT_STABLE |
| whisk | vac | False |  |  | UNEXPECTED_COLLISION: [{'geoms': ['gripper0_right_vac_mount_col', 'whisk_g8'], 'penetration_m': 0.003634729406228298}, {'geoms': ['gripper0_right_vac_mount_col', 'whisk_g8'], 'penetration_m': 0.003634729406228298}] |
| cereal | 2F | False | POST_VALIDATED_GRASP | 4.999999999988347 |  |
| cereal | 3F | False | POST_VALIDATED_GRASP | 5.000000000006111 |  |
| cereal | vac | True | ACQUISITION | 5.000000000006111 |  |
| milk | 2F | False | POST_VALIDATED_GRASP | 4.999999999988347 |  |
| milk | 3F | False | POST_VALIDATED_GRASP | 5.000000000005823 |  |
| milk | vac | True | ACQUISITION | 5.000000000006111 |  |
| lid | 2F | False |  |  | IK_FAILED: PRE_CLEARANCE no full-pose IK solution converged from deterministic seeds |
| lid | 3F | False |  |  | IK_FAILED: PRE_CLEARANCE no full-pose IK solution converged from deterministic seeds |
| lid | vac | True | ACQUISITION | 5.000000000006111 |  |

Video frame counts and initial-state comparisons: `output/c4_2-all-ee-20261031-b/evidence_audit.json`.

## Remaining work and interpretation

- Eight successful alternatives are now registered for normal execution after
  checking that their recipe fields match the saved physical trials. Four failed
  pairs remain validator-only after the bounded pose follow-up below.
- Compare contact/attachment assumptions when interpreting the coverage counts. The original
  Table II constants (4/6, 1/6, 1/6) must not be treated as measured physical limits from this run.
- Keep the packing planner's required-tool issue separate, then verify full packing goals
  after any production integration. This run has no w/o Reasoner whole-task success result.
- Finger alternatives were repeated at seeds 1 and 2 below; their initial scenes
  barely differ. Arbitrary-pose robustness and Full versus w/o Reasoner on final
  common code remain unverified.

The primary run's manifest, source.patch and untracked_source preserve the executed source.
After this run, only reporting was polished: RUNNING status for intermediate reports, malformed
worker report handling, and excluding prior output snapshots from source hashing. No recipe,
physics parameter, motion path or success threshold was changed after the primary run.


## 후속 진단·공통 실행 연결

기존 18개 결과에 이어 자세 후보 12개와 손가락 파지 반복 8회를 수행했다.
후속 20회 중 자세 후보 12회는 실패, 반복 8회는 성공이다. 이 결과를 기존
18개 매트릭스의 분모에 합쳐 성공률로 계산하지 않는다.

### 실패 조합의 한정된 자세 탐색

`output/c4_2-pose-followup-20261031-a`에 실행 전에 후보 12개를 고정했다.
위치와 방향만 변경하고 힘·접촉 판정·부착 정책·성공 기준을 유지했다.
모든 후보를 실행했고, 성공할 때까지 무제한 재시도하지 않았다.

| 후보군 | 수 | 결과 |
|---|---:|---|
| 뚜껑 2F·3F, +Y/-X/-Y 모서리 | 6 | 4회 충돌 없는 접근 경로 탐색 실패, 2회 IK 실패 |
| 거품기 3F, yaw 90도/손잡이 끝/높이 변경 | 3 | 2회 안정 접촉 실패, 높이 변경은 닫기 통과 후 LIFT에서 접촉 소실 |
| 거품기 흡착, 손잡이 위치 -.38/-.44/-.18 | 3 | 모두 CLOSE에서 장착부와 물체 충돌 |

구현 예외와 시간 초과는 없었다. 뚜껑은 실제 파지 단계 전의 접근 실패이므로
손가락 파지가 물리적으로 불가능하다는 결론으로 바꾸지 않는다.
`evidence_audit.json`에서 선언된 모든 레시피 필드와 실행 입력의 일치,
초기 상태 비교 및 영상 표본 해독을 확인했다. 런타임이 덧붙인 model_class,
recipe_id, vacuum_attachment_mode는 별도 파생 메타데이터로 기록했다.

### 반복 확인과 정상 실행 등록

밀대 3F, 바게트 3F, 시리얼 3F, 우유 2F는 seed 1·2의 8회 모두 통과했다.
모두 파지 중 부착 보조 없이 들어 올림·유지를 검증하고 이후 운반용으로
부착했다. 출력은 `output/c4_2-finger-repeat-seed{1,2}-20261031-a`다.

seed 0 대비 전체 물체 초기 위치 차이의 최댓값은 0.006192mm, 방향 차이는 0.006233도였다. 배치 변화가 거의 없어 반복 재현성 확인이며 다양한 배치에 대한 강건성 검증으로 해석하지 않는다.

다음 8개를 공통 `ALTERNATIVE_ENTRIES`에 등록했다: 밀대·바게트·시리얼의
3F/흡착, 우유의 2F/흡착. 흡착 4개는 primary seed 0 검증 근거로 등록했고,
별도 seed 반복 검증은 하지 않았다. 모두 실제 실행한 레시피 필드와 동일하다.
Full과 w/o Reasoner에 공통 적용하며 M1의 feasible EE 집합을 확장하지 않는다.
실패 조합 4개는 validator 전용이다. 정상 실행 등록은 전체 운반·배치·포장
완료를 보증하지 않으며, 후속 전체 작업에서 그 부분을 별도로 검증해야 한다.

### 기존 w/o Reasoner 실패의 현재 코드 재현

`output/c4_2-frozen-plan-audit-20261031-a`에 기존 입력을 읽기 전용으로 사용한
오프라인 재현을 보존했다. 새 API 호출과 M5 전체 실행은 하지 않았다.
기존 M1의 7개 노드 × 3 EE 판정 21개가 현재 규칙으로 모두 재현됐고,
`MISSING_REQUIRED_TOOL:SG1` 및 피드백도 재현됐다. 원본 해시 변화는 없다.

- **추정값에 따른 거부:** 시리얼·뚜껑 모두 추정 파지 폭 150mm가 2F 85mm,
  3F 140mm 한계를 넘는다. 흡착 패치 RMS 5mm도 1.5mm 한계를 넘고, 시리얼은
  상면 RMS 2mm도 한계를 넘는다. 독립 파지 성공을 이유로 이를 덮어쓰지 않는다.
- **모델의 계획 실패:** 기존 모델은 재분해 피드백 후에도 포장 대상물을 도구로
  제시한 sweep_collect를 반복했다. 필수 도구 차단은 동작 중이며 런타임 예외가 아니다.
- **실행 지원 누락:** 이전 미등록이던 성공 조합 8개를 이번에 연결했다. 기존
  frozen M1에서 추정 가능이면서 정상 실행 지원되는 조합은 4개에서 7개로 늘었다.
  시리얼·뚜껑의 선택 가능 집합은 여전히 비어 있다.
- **후보 파지 실패:** 추정상 가능한 거품기 3F는 이번 후보들에서도 유지 성공을
  입증하지 못했다. 미등록과 실제 후보 실패를 구별해서 기록한다.

`integration_audit.json`에 정상 요청의 레시피 일치, 반복 결과·영상, frozen M1의
등록 전후 선택 가능 집합을 기록했다. 전체 C4_2 성공이나 Reasoner 제거의 인과
효과를 확정한 결과는 아니다. 최종 공통 코드와 유효한 Full 대조 입력으로 동일
조건의 전체 실행을 해야 최종 ablation 비교가 가능하다.

최종 관련 테스트: **261 passed**. 시리얼 실행 지원 목록 기대값만 새 3종으로
갱신했으며, 빈 feasible EE 보존 검증을 유지했다. 기존 plate 검증의 별도 3개
기대값 불일치는 앞선 기록 그대로이며 이번 변경으로 고쳤다고 주장하지 않는다.
GitHub `fix/dohee-ablation_study`는 확인 시 여전히 `6156e34`였다.


## 실제 Full 대조 후속 완료 상태

실제 SiPhy Full 대조와 공통 포장 실행 수정 a~g는
[Full 대조 기록](c4_2_full_control_20261031.md)에 별도 정리했다.
최신 g는 4개 payload 수납 유지, 바게트 높이 돌출 30.566mm로 전체 목표 실패,
뚜껑 미실행이다. 파지 성공과 전체 포장 성공을 혼동하지 않는다.
frozen v3 w/o Reasoner는 M2의 MISSING_REQUIRED_TOOL:SG1로 중단되며,
이 추정 기반 거절을 Full의 물리 포장 실패와 따로 기록한다.
최신 관련 회귀 검사는 391 passed, 2 skipped. 다음 진단은 바게트 배치 후보와
기존 적층 접촉/최종 높이이며 전체 8개 ablation 최종 재검증은 아직 남아 있다.
