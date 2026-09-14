# without_grounding 실행 모드

기존 `full`을 기본값으로 유지한다. `scripts/run.py`의 모드 분기로 별도 실행기
`scripts/run_without_grounding.py`와 `src/tuj/ablations/without_grounding.py`를 연결한다.
기존 M2 판단 로직, M4 검색 구현, M5 실행 구현은 수정하지 않는다.

## 실험 정의

`upper-level-visual-semantic-v1`은 **M2/M4 의사결정에 제공하는 명시적인 물리·기하
grounding 정보의 제거**다. M1 계산 비용 제거 또는 모든 모듈의 시각·공간 정보 제거
실험이 아니다. 첨부 논문의 정확한 ablation 정의를 재현했다고 주장하지 않는다.

| 경계 | 제공 정보 |
|---|---|
| M2/M4 | 원본 장면 이미지(M2), 객체 ID·canonical ID·이름·종류, 태스크, 로봇/EE 고유 사양 |
| 차단 | 위치·bbox·상세 geometry·표면·재질·질량·강성·마찰·신뢰도·caption·EE 판정·reachability·술어·관계·memory 출처 |
| M5 | 선택된 계획, 현재 로봇/객체 상태와 충돌 geometry, 공통 실행·복구 로직 |

허용 목록으로 새 scene을 생성하므로 중첩 geometry나 새 측정 필드도 기본적으로
통과하지 않는다. 원본 M1은 `execution/m1.json`에 보존되고 M2/M4에 전달하지 않는다.
객체 이름은 의도적으로 유지한다. 이름까지 제거하는 실험에는 익명 ID 정책이 별도로 필요하다.

M2는 같은 모델의 이미지 입력으로 (1) 의미적 서브골 분해, (2) 도구 선택과 객체별 EE
후보 추정을 수행한다. 호출별 잘못된 JSON/ID는 1회 재시도한다. API 오류는 그대로 실패한다.
EE 후보는 `visual_semantic_unverified` 출처를 기록하며, 빈 후보를 모든 EE로 채우지 않는다.
물리 술어는 `unknown`이고 기존 M4의 defer 정책을 사용한다. 사실을 `sat`로 조작하지 않는다.
기존 core의 동작 분해·상징적 순서·mutex를 재사용한다. 측정 기반 재분해·partition·배치
좌표 계산은 호출하지 않는다. 이미지에 근거한 순서는 모델이 결정한다.

M4는 이 후보 안에서 기존 비용·순서 검색을 수행한다. 스크립트 레지스트리의 객체별
지원 EE로 미리 좁히지 않는다. M5에는 기존 선택 계획 계약으로 전달하며, 자동으로
정답 도구/EE를 넣지 않는다. 동일 로봇 사양의 capability 검사는 유지된다.

## 사용법

기존 M1과 **동일 장면의 이미지**를 이용해 M4까지만 실행:

```powershell
python scripts/run.py c1_2 --grounding-mode without_grounding --m1-json output/c1_2/m1.json --scene-frame output/c1_2/frame.png --provider openai --model gpt-4o --stop-after m4
```

`OPENAI_API_KEY`는 환경변수로 제공한다. 명령문·설정 파일에 키를 저장하지 않는다.
`--m1-json`을 생략하면 기존 M1 실행기를 호출해 원본을 생성한 뒤 정보를 필터링한다.
기본 출력은 `output/without_grounding/c1_2/seed_0/`다. 반복 실험은 서로 다른
`--output-dir`을 지정한다. full 출력 폴더를 지정하면 실행을 거절한다.

해당 실행의 M4 산출물로 M5를 시작:

```powershell
python scripts/run.py c1_2 --grounding-mode without_grounding --provider openai --model gpt-4o --start-from m5 --m5-simulate controller
```

기존 실행과 task/seed/model/provider/로봇 사양이 일치해야 한다. 커스텀 출력 폴더를
썼다면 같은 `--output-dir`을 지정한다. M5 옵션은 full과 동일한 값으로 명시한다.
예를 들어 3F 경로 캐시 문제를 피하려고 `--m5-args --ee-attach-policy precomputed-or-plan`을
쓰면 full 대조군에도 같은 옵션을 사용해야 한다. 검증 허용치를 완화하지 않는다.

## 산출물과 재시작

- `m1.json`, `robot_decision.json`: 제한된 의사결정 입력.
- `execution/m1.json`, `frame.png`: 원본 M1과 사용 이미지.
- `m2.json`: 시각적 선택·EE 가설·unknown 조건·LLM 사용량.
- `m4_request.json`, `gk_bundle.json`, `m4.json`: 실제 M4 요청과 검색 결과.
- `ablation_manifest.json`: 정책·입력/산출물 해시·단계 상태·실패 위치·M5 옵션.
- `motion_attempt_NNN/`: 각 M5 시도의 계획 사본·캐시·summary·영상.

새 실행은 빈 디렉터리에서 시작한다. `--start-from`은 해당 모드의 manifest와
산출물 해시를 확인한다. 상위 단계를 재실행하면 하위 단계 재사용 체크포인트를
무효화한다. M5 재시도마다 출력과 keyframe 캐시를 분리하고 전역 캐시 설정은 복원한다.

v1은 로봇 사양에 있는 초기 EE/held-tool/hand-empty 상태를 사용한다. 임의의 접지 사실을
포함할 수 있는 `--initial-state`와 과거 `--m5-physical` 진입점은 지원하지 않는다.
통합 실행기에는 M5에서 M2/M4로 자동 되돌아가는 재계획 루프가 없으며,
이 모드는 그런 루프를 새로 도입하지 않는다. M5 내부 복구는 공통 코드를 따른다.

## 검증과 결과 해석

테스트는 가짜 모델 응답과 저장 월드를 사용해 정보 차단, 비정상 선택 보존,
빈 EE 후보의 실패, 실제 M4 검색, M5 요청 변환, full 기본 분기, 재시작/캐시 격리를 확인한다.
저장 월드는 입력 변환 검사에만 사용하며 현재 native-flex 물리 실행의 근거로 쓰지 않는다.

이 모드 구현/테스트 통과가 SR 측정 결과는 아니다. 실제 API와 시뮬레이션으로 paired
실험을 별도로 수행해야 한다. full과 동일 장면·seed·모델·예산을 사용하고 LLM 호출량도
보고한다. 전용 시각 판단 프롬프트는 대체 경로의 일부이며 full과 완전히 동일하지 않다.

현재 full 통합 실행기는 스크립트 지원 EE를 사전 필터링하지만 이 모드는 하지 않는다.
이 차이는 반드시 보고해야 하며, **스크립트 미지원 실패를 물리적 EE 불가능(HV)으로
집계하면 안 된다.** HV 비교에는 두 조건에 동일 실행 지원 범위/평가 기준이 필요하다.
API/응답 계약 오류, 스크립트 지원 오류, 물리 실패를 분리하고, 전체 실패를 곧바로
grounding ablation의 인과적 효과라고 해석하지 않는다. PE/W의 정의와 분모를 정한 뒤
공통 평가기로 계산해야 한다; 이 구현은 새로운 SR/HV/PE/W 평가기를 포함하지 않는다.
