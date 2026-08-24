# 새 태스크 러너 만들기 (c1_2 / c2_2 등)

씬 환경(`environments/<task>.py`)이 준비되면 아래 3곳만 갈아끼우면 됨.
c1_1 → c2_1 이식이 정확히 이 패턴으로 진행됨.

## run_m0_<task>.py

1. **파일 상단**
   ```python
   TASK = "c1_2"           # 또는 "c2_2"
   ```
2. **suite.make**
   ```python
   env_name="C1_2_XXX",    # environments/<file>.py의 클래스 이름
   ```
3. **class_of(inst)** — 씬의 인스턴스명을 M0 노드 클래스로 매핑.
   (컨테이너·랙·distractor 포함 여부는 태스크 목적에 따라 결정)
4. **object_bound_points(env)** — auto-fit 카메라가 프레임 안에 넣을
   대상 리스트. 보통 `env.<target_objects> + env.<containers>` 형태.

## run_m2_<task>.py

1. `TASK = "..."` 만 바꾸고
2. `<Task>Backend(MockBackend)` — 클래스별 mock 물성 (SiPhy 백엔드
   쓰면 안 씀). 재질·밀도·E 값은 실측 정합보다 결정 경계를 만드는
   방향으로 (예: vac payload 근처에 걸치는 쌍).

## 씬 준비 안 됐을 때

`suite.make`가 "no such env" 에러 → environments/ 파일이 아직 없다는 뜻.
씬 담당 팀원에게 요청. 러너 자체는 c1_1/c2_1 스타일 그대로 미리 두어도 됨.
