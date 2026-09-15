# Real-5 GT 출처와 물리적 의미

GT의 단일 원본 설정은 [`configs/c4_real5_gt.json`](../../../configs/c4_real5_gt.json)이다. 이 준비 작업은 그 파일을 덮어쓰거나 Table 4의 synthetic `scene_properties`를 사용하지 않았다. 출처는 [Kandukuri et al. arXiv:2309.15703v3, Appendix B.5/Table 5](https://arxiv.org/html/2309.15703)이며 공식 공개 원본은 [MPI Keeper EV-RealPhys](https://keeper.mpdl.mpg.de/d/5ec213b655b44e40a382/)이다. 로컬 압축본 `data/external/kandukuri_ev_realphys/ev-realphys.tar.gz` SHA-256은 `4c484ec1e5f3f66ccb6a930437404ba7c67a1bb5590791027ced9741e1136eab`; 공개 README에는 별도 dataset release tag가 없으므로 이 해시를 정확한 데이터 버전으로 잠근다. `dataset_info.md`는 이 폴더가 D455로 촬영한 real-world `{val,test}_sliding`이라고 명시한다. `test_sliding`의 scene_properties와 모든 scene_gt `obj_id`를 검증해 BOP 2/5/11/12/14를 각각 YCB 객체 003/006/019/021/025에 대응시켰다. Tuna Fish Can은 포함되지 않는다. 본문 주요 분석은 4종이고, Cracker Box는 논문 보충 분석과 Table 5에 있으며 공개 test split에는 다섯 장면이 있다.

| BOP ID | Table 5 물리 객체 | 질량 kg | 객체–테이블 결합 μ | GT 횟수 |
| ---: | --- | ---: | ---: | ---: |
| 2 | 003 Cracker Box | 0.0565 | 0.280 | 10 |
| 5 | 006 Mustard Bottle | 0.3180 | 0.159 | 10 |
| 11 | 019 Pitcher | 0.7120 | 0.220 | 10 |
| 12 | 021 Bleach Cleanser | 0.4030 | 0.169 | 10 |
| 14 | 025 Mug | 0.1030 | 0.110 | 10 |

원 논문은 Mustard/Pitcher/Bleach를 모래로 부분 충전한 **그 실물**을 저울로 측정했다. 영상 입력이나 이름으로는 내부 모래의 양을 식별할 수 없고, 관리용 충전 메타데이터를 모델에 제공해서도 안 된다. 마찰 GT는 표면에 마커를 단 테이블을 기울여 각 객체를 10번 활주시킨 MoCap 궤적에서 속도 기울기와 테이블 각도 α로 `μ_combined = tan(α) - a/9.81`을 구한 **중앙값**이다. 이는 10회의 모델 추론이 아니다. 논문은 Cracker Box 바닥의 이방성과 비균일 접촉도 지적한다.

공식 split와 객체 ID가 Table 5의 동일 실물 실험을 가리킨다는 출처 연결을 확인했다. 공개 per-scene 메타데이터에는 독립 시리얼 번호나 모래 충전량이 없으므로 그것까지 영상별로 재계측·대조했다는 주장은 하지 않는다. 따라서 GT 사용 범위는 위 아카이브의 공식 `test_sliding`에만 묶어 둔다.

공개 test 영상은 촬영 시 RGB-D 60 Hz, MoCap 240 Hz였지만 timestamp를 맞춰 **30 Hz로 downsample**했다고 논문 §5.1이 명시한다. 따라서 저장소의 `dt=1/30 s`는 export sequence의 명목 시간 간격에 대응한다. BOP의 동일 번호·848×480 크기·프레임별 `cam_K`/depth scale은 검사했지만 픽셀 수준 RGB/depth registration과 30 Hz 잔여 지터는 독립 검증하지 않았다. 기존 RGB-D 궤적식 `probe_mu_from_track`은 수평, 외력이 없는 순수 평면 병진 활주에서 감속도/중력가속도로 같은 결합 접촉 μ를 겨냥한다. `FrictionHead.stage0`는 material/RMS에 따른 **gripper slip/contact proxy**로 별개이며 Table 5와 비교할 수 없다. 실제 시험 장면의 회전·외력·테이블 경사·occlusion이 남으면 감속 기반 숫자는 별개의 effective drag가 되므로 [`qc_policy.yaml`](qc_policy.yaml)의 `TARGET_SEMANTICS_MISMATCH`/BLOCKED 게이트를 적용한다.
