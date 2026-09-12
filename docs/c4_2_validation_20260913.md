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

## CONTRIBUTING 규칙에 따른 PR 이력 정리

PR 작업 브랜치는 `fix/dain-c4-2-packing`, 대상은 `main`이다. 기존 검증 브랜치를 보존하고, 새 브랜치에서 커밋 제목만 50자 이내 Conventional Commits 형식으로 정리했다. 각 새 커밋의 파일 트리가 대응하는 원본과 동일함을 확인했다. 원본 커밋은 메시지의 `Original-commit` 및 외부 `dain_commit_mapping.json`으로 연결된다. 실제 영상에 기록된 코드 hash는 이 문서 상단의 원본 검증 hash 그대로이며, 재실행을 했다고 주장하지 않는다.

정리 후 코드 대응 커밋은 `f8df2713df3742d3e1ccf21d99d4cdf44a59e05c`이다. 이 기록을 추가하는 후속 변경은 검증 문서만 수정한다. 런타임 코드는 동일하다. 변경 Python 57개 파일의 AST·tabnanny 및 git diff --check를 다시 확인했다. 별도 프로젝트 lint/formatter/build 명령은 저장소에 정의되어 있지 않아 이 검사를 명시하며, 기존 실제 실행과 회귀 테스트 결과를 재사용한다.

권장 500줄을 넘는 PR이다. 파지 상대 자세, 열린 손 해제 기하, 접촉 지원과 최종 수납/뚜껑 판정이 연결된 실제 C4_2 검증 범위를 유지했다. 리뷰어는 아래 원인별 작은 커밋과 해당 회귀 테스트 순서로 검토할 수 있다. 작은 PR 권장 크기를 충족한다고 주장하지 않는다. 최소 한 명 승인 후 병합하는 규칙을 따르며 이 작업에서는 병합하지 않는다.

## 롤백

태스크 완전 종료 후 깨끗한 해당 checkout에서 전체 통합 변경은 다음으로 되돌린다. 검증 문서만 추가한 후속 커밋은 실행 코드에 영향을 주지 않으며 별도로 revert할 수 있다.

```text
git revert -m 1 f8df2713df3742d3e1ccf21d99d4cdf44a59e05c
```

선택적 롤백은 아래 원인별 커밋을 사용한다. 후속 변경과 의존 관계가 있으면 충돌 내용을 확인해 해결해야 하며 강제 reset/checkout/clean으로 처리하지 않는다. 기존 사용자 변경은 대상에 포함하지 않는다.

| 커밋 | 변경 | 선택적 롤백 |
|---|---|---|
| 8192890 | fix(task): prefer nearby clear packing volume and retain placement intent | `git revert 8192890fbee2178b5c7775c7e73aab7614f80418` |
| 3a892f4 | fix(task): preserve rigid container contact support under packed loads | `git revert 3a892f43a36f6492735d46a97800e900978b2e0b` |
| a840e1b | test(task): isolate packing ranking from the release corridor gate | `git revert a840e1b70a9a70d103b956e9b83f539691a325c3` |
| f60e2f7 | fix(execution): allow per-run OpenAI response token budgets | `git revert f60e2f7a2a48792641af97eaa44eda5de819405c` |
| 78d126c | fix(planner): refine inconclusive gravity release clearance checks | `git revert 78d126c60287bdd60a0cca6161ccd2494ebefbe9` |
| eca3860 | fix(execution): preserve scene contact calibration across EE swaps | `git revert eca386012fb923d178e45b5ec42000d87ac1fa72` |
| dd3e26d | fix(planner): reject obstructed gravity release orientations | `git revert dd3e26d9bbdf65e3324485319aae3f8c99eeda2b` |
| 238bd9f | fix(execution): derive endpoint grasps from contact and support geometry | `git revert 238bd9f02f343725474e41fc5e6c364042de96d4` |
| 1e97d6e | fix(planner): honor requested clearance during scripted grasp approach | `git revert 1e97d6eb60f6c0a08ba9b84c9567cd29027445d3` |
| 3c4dbd3 | fix(execution): synchronize attached object velocity with the hand | `git revert 3c4dbd36f44a1d110549b6b2c934d7e448613363` |
| 0ff14a2 | fix(task): verify covers on container rims with packed contents | `git revert 0ff14a2d564f940abf4589a4df71266e22ca0a38` |
| 757d274 | fix(planner): ground container place-on goals above the rim | `git revert 757d2741700158c86f386cc988597a8e25b363ac` |
| 079033d | fix(planner): measure rigid vacuum release geometry | `git revert 079033d038e38a20d11a12242d720b960d6f9847` |
| 8f76da5 | fix(execution): synchronize attachment from current hand pose | `git revert 8f76da5eb9840ceaf47811b47f335508a7ae9014` |
| 8f98b39 | fix(task): preserve stable baguette contact before lift | `git revert 8f98b39f607a6567c302454e63eff6c3fdc58ff6` |
| cac5590 | fix(execution): verify ambiguous attach penetration with convex geometry | `git revert cac5590600ab0bc0b76f44b424854ae2500788f2` |
| 91bc0f6 | fix(task): increase approved C4_2 box wall height | `git revert 91bc0f630b4ff69725372328788615f22685a6d8` |
| 85e277e | fix(task): derive edge grasps from open-hand release geometry | `git revert 85e277ee235082ea26b788263fedd0bead695905` |
| c4f3b75 | fix(planner): reject packing goals without collision-valid IK | `git revert c4f3b75162212b9e1fedcdb1f387cea945c29692` |
| df79ee6 | fix(planner): rank reachable packing orientations by occupied geometry | `git revert df79ee6ed962c8656d44ce02fb7c74ac055ba25c` |
| ad733ae | fix(execution): account for bounded per-segment settling in run budget | `git revert ad733ae2497c57e110bfb5fe6f9a19e983cabd6b` |
| 67e8aaf | fix(planner): filter packing orientations by measured grasp reachability | `git revert 67e8aaf04bac914bba4c1fb399dfa519f41e658d` |
| b2ff038 | fix(task): attach whisk after stable bilateral contact | `git revert b2ff038acedf18a0d4fc002ca5da1d72c3894d17` |
| 87f9c84 | fix(execution): await released container objects before completion | `git revert 87f9c842f1bd6c0ca9dce3ad7a92513b250be087` |
| 5893c93 | fix(planner): expose measured hand entry for container retreat | `git revert 5893c93de5d220705b832d1226740e61d6e2778b` |
| 0a1ae14 | fix(planner): converge open linkage geometry at joint bounds | `git revert 0a1ae149e25a0a99e87d777b8cf6c6f7971f71b2` |
| 8faedfd | fix(planner): derive open hand geometry from the executed endpoint | `git revert 8faedfd9ae29fcc09c4f518c9982bc0e81327c70` |
| 9118e61 | fix(planner): reserve declared tracking tolerance above container rims | `git revert 9118e612539a5c529cb4576f41694253bd78b2de` |
| f60379f | fix(execution): recover missing constrained finger contact without unloading peers | `git revert f60379f978e1f82b4531e6c2d6dbd6bc88aa55a0` |
| 3551c06 | fix(planner): include hand rim clearance in container transport | `git revert 3551c066a0e8ac2c2d5491b3fa26b7a2e487bf39` |
| e0e4476 | fix(planner): reject unreachable stable faces using measured grasp | `git revert e0e4476cebcf90b10fc1e3cc84b6cd7d623c3511` |
| 7da135e | fix(planner): derive low stable container faces from collision geometry | `git revert 7da135e46e29867f642dbd2c39363260d1407864` |
| 3058f39 | fix(planner): preserve measured orientation at held start anchors | `git revert 3058f39706a6d3255be34641f724081ae28c318f` |
| f03f532 | fix(execution): identify lift support from measured static contacts | `git revert f03f532e679f42c01c33d6ebc4fac8746869ce0f` |
| dee6bbd | fix(planner): ground container poses from declared collision geometry | `git revert dee6bbd50f4d5164b8243cd36eecdf58860da0e7` |
| 0e4dd28 | fix(planner): honor assigned slot for the first region placement | `git revert 0e4dd280bf3c31fad7685948340d2362d14ce969` |
| d4a5427 | fix(execution): pass live grasp geometry to region place grounding | `git revert d4a5427e23e7cd7a5edeaf0cdf89db652f31264c` |
| deb6647 | fix(planner): derive container release height from hand opening geometry | `git revert deb6647ebac8680f71a635f5a6feeaa8cf483cbd` |
| 9dd4c09 | fix(execution): retain validated 3F closure on constrained objects | `git revert 9dd4c097534b6faf27f4cb6a84448b710c786927` |
| dd827bc | fix(execution): damp calibrated parallel gripper preshape oscillation | `git revert dd827bc9622f133c657f97e8e544e9fb76191ba7` |
