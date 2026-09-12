# C4_2 실제 실행 및 녹화 검증

2026-09-13. 검증 코드 `cd2eb481ed61a728cb7b87b5c05a57b66da3bc45`; main 기준 `bd7812cc05121055c7178649616c8f7d2a2eb4c8`.

초기 M1부터 M2, G_k, M4, M5까지 실제 OpenAI API로 두 차례 전체 실행했다. 사용자 요청에 따라 C4_1은 제외했다. 저장소의 기존 사용자 변경은 별도 checkout에 보존했고 이 PR에 포함하지 않았다.

## 변경과 효과

- 안정 접촉 후 attach 시점의 손/물체 상대 자세를 보존하고 물체 속도를 손과 동기화한다. 거품기·바게트 파지에서 5초 동안 접촉과 attach를 유지했다.
- 물체 충돌 형상, 실제 파지 자세와 열린 손의 기하를 사용해 도달 가능한 수납 방향, 운반 여유와 해제 경로를 검사한다. 선택한 배치 의도를 PLACE까지 유지한다.
- 상자 벽의 기본 접촉 속성 혼합으로 얇은 물체가 크게 침투하던 원인을 기존 자산의 접촉 프로필 명시로 수정했다. 최종 거품기 최대 옆벽 초과는 0.304mm로 기존 1.5mm 연성 접촉 판정 이내다.
- detach 후 안정 상태와 실제 충돌 형상 전체 수납을 확인한다. 뚜껑은 수납 대상이 아닌 최종 덮기 대상으로 처리하며 덮임과 테두리 물리 지지를 검사한다.
- 승인된 C4_2 상자 높이 조정이 이력에 포함된다. 이후 추가 크기 변경, 충돌 검사 비활성화 또는 성공 조건 완화는 하지 않았다.
- 최신 main의 일반 쌓기 판정, 복수 지지면, 물리 분리/재접촉 조건과 3F 명령 범위 제한을 통합했다.

## 검증 결과

| 항목 | 무녹화 전체019 | 고화질 전체020 |
|---|---|---|
| Run ID | pr_full_unrecorded_019 | pr_full_hq_020 |
| 시작~종료(KST) | 04:24:44~05:44:28 | 05:51:28~06:28:53 |
| 실행 시간 | 4783.094초 | 2245.016초 |
| 종료 코드 | 0 | 0 |
| 서브골 | 18개 순서 일치 | 18개 순서 일치 |
| 실행 단계 | 25개 성공 | 25개 성공 |
| 파지/모션 | 6개/19개 성공 | 6개/19개 성공 |
| 내용물 수납 | 다섯 물체 모두 내부 | 다섯 물체 모두 내부 |
| 뚜껑 덮임/물리 안정 | 통과 | 통과 |
| 금지 충돌/처리되지 않은 오류 | 0/없음 | 0/없음 |
| 실행 중 코드 변경 | 없음 | 없음 |
| 바게트 상단 여유 | 61.743mm | 3.491mm |

AST 210개 파일 통과. 관련 회귀 모음 276개 통과(123.51초), 추가 main/병합 경계 모음 142개 통과(3.92초). 두 모음은 중복을 포함하며 고유 418개 테스트라고 해석하지 않는다. C4_1 및 다른 전체 태스크의 실제 재실행을 이번 결과로 주장하지 않는다.

최종 뚜껑은 개구부 덮임, 내용물 모두 내부, 안정 상태, detach 및 손 접촉 0을 만족한다. 테두리 오차 약 1.082마이크로미터, 지지 접촉 16개/상향 지지력 6.674724N이다. `collision_count=0`은 금지 충돌이 없다는 뜻이며 정상 물체 접촉이 없다는 뜻은 아니다.

## 영상 및 재현

- 성공 영상: `success/C4_2__success__pr_full_hq_020__20260913_062853.mp4`
- 원본: `pr_full_hq_020/video/C4_2__diagnostic__pr_full_hq_020.mp4` (보존)
- 해상도 1920x1080, 30fps, 10,192프레임, 339.733초, 199,232,881바이트.
- SHA-256: `fded88c3fa37fe06b441207f8fee94c91eddb950bc6ef117b75e2709f7a9c10f`
- 모든 프레임 읽기 통과, 검은 프레임 0개. 시작 장면/EE 교체/물체 조작/최종 덮임을 16개 표본 및 마지막 프레임으로 시각 확인했다.
- 영상은 시뮬레이션 시간 기준으로 기록하므로 API 및 계산 대기 시간을 포함한 실제 실행 시간과 다르다. 녹화 때문에 추가 물리 step을 실행하지 않았다.

산출물 루트는 작업 workspace의 `reports/c4_2_resume_20260911_2303`이며 Git 밖에 보존한다. 각 run의 `official_run.json`에 실제 실행 명령/코드 hash/시각, `final_audit.json`에 모든 단계와 최종 형상 판정, `stdout.log`/`stderr.log`에 원본 로그가 있다. `success/C4_2__pr_full_hq_020__manifest.json`은 영상·로그·run·코드를 연결한다. 외부 실행 controller의 `run.json`은 controller checkout 메타데이터이므로 실제 태스크 코드 hash는 `official_run.json`을 사용한다.

공식 명령 구조(각 실행별 새 결과 디렉터리 사용):

```text
python scripts/run.py c4_2 --provider openai --model gpt-5.4-mini --seed 0 --memory <run>/memory.json --output-dir <run>/artifacts --m5-args --simulate controller --headless --video <run>/video/full.mp4 --width 1920 --height 1080 --video-fps 30
```

무녹화 실행은 video/width/height/video-fps 옵션을 생략했다. MuJoCo 3.3.1, timestep 0.001, implicitfast, UR5e, OpenAI keyframe 최대 출력 32000/medium을 사용했다. M5 keyframe cache는 끄고 M1 intrinsic memory 재사용은 공식 정책대로 허용했다. 초기 memory와 소스 지문을 run별 보존했다. 제한 시간은 10500초였으며 두 실제 실행 모두 timeout 없이 종료했다. 인증은 기존 환경변수만 사용했고 키/인증 헤더는 저장하지 않았다.

## 해석상의 한계 및 남은 경고

기존 CLI의 `m5.json.task_goal_status`는 `NOT_EVALUATED`로 남는다. 성공 근거는 이 미갱신 집계 필드가 아닌 실제 `TaskAwareGoalEvaluator`의 최종 덮임/내용물/물리 안정 결과, 모든 실행 단계와 최종 충돌 형상 감사다. 집계 필드만 읽는 소비자는 잘못 분류할 수 있다. 성공 조건이나 집계 필드를 수정하지 않았다.

두 실행의 최종 자세와 바게트 여유는 달랐다. 무녹화019는 대체 XY를 찾지 못해 기존 위치에서 성공했으므로 F039 위치 탐색의 단독 효과나 비결정성 제거를 입증하지 않는다. 두 실제 성공이 임의 seed/장면에 대한 보증은 아니다.

선택적 mimicgen/robosuite_models/mink 및 private macro 안내 6개 경고 줄은 보존했다. 실제 사용한 UR5e/controller 경로가 동작하고 두 전체 실행이 통과하므로 이 경고들은 이번 태스크를 차단하지 않았다.

최근 실행·분석·중요 제어/기하 검토는 사용자 최신 지시에 따라 OpenAI를 사용했다. review119~122는 점유/위치 선택 가설과 수정안을 실제 재현·코드와 비교했고, review123은 main 통합을 검토했다. 무녹화019 후 review124로 최종 교차 검토한 뒤 코드 변경 없이 HQ020을 실행했다. 모델 의견만으로 성공 처리하거나 코드를 변경하지 않았다.

## 롤백

태스크 완전 종료 후 깨끗한 해당 checkout에서 전체 통합 변경은 다음으로 되돌린다. 검증 문서만 추가한 후속 커밋은 실행 코드에 영향을 주지 않으며 별도로 revert할 수 있다.

```text
git revert -m 1 cd2eb481ed61a728cb7b87b5c05a57b66da3bc45
```

선택적 롤백은 아래 원인별 커밋을 사용한다. 후속 변경과 의존 관계가 있으면 충돌 내용을 확인해 해결해야 하며 강제 reset/checkout/clean으로 처리하지 않는다. 기존 사용자 변경은 대상에 포함하지 않는다.

| 커밋 | 변경 | 선택적 롤백 |
|---|---|---|
| df620b3 | fix(task): prefer nearby clear packing volume and retain placement intent | `git revert df620b3f5b52c6cdfd027a0f51d9068ad68d420c` |
| c118fe5 | fix(task): preserve rigid container contact support under packed loads | `git revert c118fe5682ada0f7ec40cac4e9ceb99c26901ed9` |
| de096d3 | test(task): isolate packing ranking from the release corridor gate | `git revert de096d35f04562f7a57927a338838412a1a9cf2a` |
| c76bf5c | fix(execution): allow per-run OpenAI response token budgets | `git revert c76bf5c780657b25b00852de31288754a4546e13` |
| 9438e01 | fix(planner): refine inconclusive gravity release clearance checks | `git revert 9438e012226532fb9caef85fdd4727c2652adfec` |
| 2ef78d1 | fix(execution): preserve scene contact calibration across EE swaps | `git revert 2ef78d1d879fa5ccace5995872f1a31c3ff656c0` |
| 2d7c8ad | fix(planner): reject obstructed gravity release orientations | `git revert 2d7c8adf66dce462603e6f57731aebdf378a546a` |
| bc92ea4 | fix(execution): derive endpoint grasps from contact and support geometry | `git revert bc92ea49205d972606bedd559f3747d1e5bf5a32` |
| bc59e21 | fix(planner): honor requested clearance during scripted grasp approach | `git revert bc59e21894934955d0952146d407dc148defb7a5` |
| abe5da2 | fix(execution): synchronize attached object velocity with the hand | `git revert abe5da250b1f46aaefa8c034ba4653f9744d62ec` |
| 5f7c53a | fix(task): verify covers on container rims with packed contents | `git revert 5f7c53a79d7964dca6149fe9840fe70a1862a009` |
| b2cfaf9 | fix(planner): ground container place-on goals above the rim | `git revert b2cfaf9e703211b47b73e6b3f90974873d289a67` |
| 89e1286 | fix(planner): measure rigid vacuum release geometry | `git revert 89e12868b647f5bdeb62b93f4e1bd57ba9dfcd0a` |
| 1240ffe | fix(execution): synchronize attachment from current hand pose | `git revert 1240ffea7a5e93ce59e931ba613647cfdd128d14` |
| 747cfb3 | fix(task): preserve stable baguette contact before lift | `git revert 747cfb3c2427ca1886f6b08002e6e6473757e466` |
| 2c7c6ae | fix(execution): verify ambiguous attach penetration with convex geometry | `git revert 2c7c6ae58839f6504ee64059adeff7190e1a3491` |
| 84e5144 | fix(task): increase approved C4_2 box wall height | `git revert 84e51440ea3104fddf8dd1b1096606a3d9c63311` |
| 9d537bc | fix(task): derive edge grasps from open-hand release geometry | `git revert 9d537bcb56c04b75e8236c20b469fe2cdfd4a6a6` |
| 5fba16b | fix(planner): reject packing goals without collision-valid IK | `git revert 5fba16bc7aa558d4f754f6af44b518779dde6e17` |
| 4b8a460 | fix(planner): rank reachable packing orientations by occupied geometry | `git revert 4b8a4605c36101aa769edff0f1cb0cbbaa1d375d` |
| 7858e9a | fix(execution): account for bounded per-segment settling in run budget | `git revert 7858e9a265de043d3a4c5b121e9c0e42e0c654ba` |
| 6be84a5 | fix(planner): filter packing orientations by measured grasp reachability | `git revert 6be84a5170bb254ad2596fd54741f7ec97d5da7f` |
| 7fd70e8 | fix(task): attach whisk after stable bilateral contact | `git revert 7fd70e80e627e8efe23f73a448621a1efa237e99` |
| df1923b | fix(execution): await released container objects before completion | `git revert df1923b06834fe8f8676ef6950b4b8c7f367344f` |
| 0ddb242 | fix(planner): expose measured hand entry for container retreat | `git revert 0ddb242514da741d124c8ed92ce75efe1e74041d` |
| 9ec0b5a | fix(planner): converge open linkage geometry at joint bounds | `git revert 9ec0b5a496301b48f83c786edb07993092df2037` |
| 28258cd | fix(planner): derive open hand geometry from the executed endpoint | `git revert 28258cd53512063fd407bd3c16fbdad38d610d42` |
| 7b61a19 | fix(planner): reserve declared tracking tolerance above container rims | `git revert 7b61a1958578cebf3391482b590569fc28530c96` |
| e3e700f | fix(execution): recover missing constrained finger contact without unloading peers | `git revert e3e700f51b48297916f44e39e86db2627afe2caf` |
| 1c3d725 | fix(planner): include hand rim clearance in container transport | `git revert 1c3d725dc816251d65e727dcecd5e803ab612f0a` |
| 46ef453 | fix(planner): reject unreachable stable faces using measured grasp | `git revert 46ef453b66367b56a57fb612808b7bd80d622d58` |
| 22e869f | fix(planner): derive low stable container faces from collision geometry | `git revert 22e869fb686dbb5b8d12381227ed98bb674fd586` |
| f00feb5 | fix(planner): preserve measured orientation at held start anchors | `git revert f00feb56a326614868ee09a931a946ace7ae4ded` |
| 084d702 | fix(execution): identify lift support from measured static contacts | `git revert 084d70254d02a9e935253697876062e037445282` |
| 0e6c60a | fix(planner): ground container poses from declared collision geometry | `git revert 0e6c60ac8c2ce86dc27379d8bbfc414579b5bb69` |
| 6075d55 | fix(planner): honor assigned slot for the first region placement | `git revert 6075d55b8ed065c6a10d3b3850a5c2f646f17a99` |
| 7d6e656 | fix(execution): pass live grasp geometry to region place grounding | `git revert 7d6e656356511cb564efceac6cc266a06c176803` |
| cfa6dae | fix(planner): derive container release height from hand opening geometry | `git revert cfa6dae430ceb0d1022b8d79d18db7c9a216fdf8` |
| 7330cc1 | fix(execution): retain validated 3F closure on constrained objects | `git revert 7330cc1446e6fb97338661c686c6b57e1c95b790` |
| eaa117e | fix(execution): damp calibrated parallel gripper preshape oscillation | `git revert eaa117ed5114e095333a788ab80912694e072320` |
