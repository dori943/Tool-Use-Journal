# Scene Graph → Task Planner 인터페이스

Task Planner가 직접 사용하는 scene 정보는 symbolic planning과 task-level 자원
검증에 필요한 최소 필드다.

| 필드 | 용도 |
|---|---|
| object id와 현재 region | symbolic precondition/effect |
| `mass_kg` | EE payload suitability |
| obstacle/world payload | Motion Planner pass-through |
| scene signature | lazy geometry cache identity |

정확한 물체 pose와 장애물 mesh는 Task Planner가 해석하지 않는다. Scene Graph가
Motion Planner의 world model을 초기화하거나 `WorldSnapshot`의 opaque payload로
전달한다.

서브골 실행 후 Task Planner는 symbolic effect를 적용해 새로운 scene signature와
causal history를 만든다. Motion Planner는 그 history를 replay해 예상 장면을
재구성한다.

외부에서 제공되는 작업 좌표의 권위와 좌표계 변환 책임은 Scene Graph/Motion
Planner 계약에 있다. Task Planner는 좌표 후보를 생성하거나 선택하지 않는다.
