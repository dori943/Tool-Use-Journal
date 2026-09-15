# Static Real-5 입력 QC

이 선택은 GT와 기존 예측을 읽지 않고 RGB, depth, 기존 M1/SiPhy crop 경로만 사용했다.
자유 활주/감속도/시간축/회전은 정지 이미지 QC에 사용하지 않았다.
원본: https://keeper.mpdl.mpg.de/d/5ec213b655b44e40a382/ · archive SHA-256 `4c484ec1e5f3f66ccb6a930437404ba7c67a1bb5590791027ced9741e1136eab`
source manifest SHA-256 `07752e7d89b244db2210041d8fe308a2761d0a99dc166b98210ad4198ab6b8ee` · policy `static_real5_v1_20260914`

선정 50/50, 객체별 {'019_pitcher_base': 10, '006_mustard_bottle': 10, '021_bleach_cleanser': 10, '003_cracker_box': 10, '025_mug': 10}.
초기 프레임 교체 1건. 후보 실패 1건.
자동 QC 단계의 경고 입력 2건(수동 검토 전). 경고는 자동 제외하지 않았다.

동일 RGB 중복은 파일 SHA-256으로 검사했다. RGB와 depth는 같은 프레임 번호, 848×480 크기와 per-frame K를 사용한다.
마스크는 depth 지지 평면 분할→RGB GrabCut→M1 점군 필터로 생성했다. 공개 GT mask, pose, 물성값은 사용하지 않았다.
원본 RGB/depth와 추출 mask overlay를 별도 보존했다. crop/context는 모든 객체에 동일 padding 규칙을 적용했다.
약한 배경 혼입과 로봇 노출은 contact sheet 수동 점검 후 WARNING에 기록한다.

## 교체·경고

- 000001_000040: weak background/mask contamination; visual review required
- 000017_000035: 대체 프레임; weak background/mask contamination; visual review required

## Contact sheet 수동 검토 (GT·기존 prediction 미참조)

50개 원본 RGB, black-background mass crop, 객체+테이블 context를 모두 확인했다. 객체 식별 불가능·심한 절단/가림·배경만 있는 crop·테이블이 보이지 않는 context는 선정본에 없었다. mask의 작은 table/robot fragment는 원본을 고치지 않고 WARNING으로만 표시했다. 자동 정책에 의해 `000017` 두 번째 슬롯의 frame 40은 객체 mask가 너무 작아 frame 35로 대체되었다. `000015` 두 프레임은 객체가 보이고 표면 context가 유효하므로 유지했으며, 질량 crop의 배경 잔여만 경고한다. `000009` 두 프레임도 유지했다; 과거 궤적 품질은 이 정지 평가의 제외 근거가 아니다.

최종: 50/50 입력, 객체별 {'019_pitcher_base': 10, '006_mustard_bottle': 10, '021_bleach_cleanser': 10, '003_cracker_box': 10, '025_mug': 10}, WARNING 13, 프레임 교체 1, 최종 제외 0.

- `000001_000040`: weak background/mask contamination; visual review required
- `000003_000040`: minor table/robot fragment in mass mask
- `000005_000040`: minor background fragment in mass mask
- `000008_000010`: minor background fragment in mass mask
- `000009_000040`: minor background fragment in mass mask
- `000010_000010`: robot fragment in mass mask
- `000010_000040`: minor background fragment in mass mask
- `000012_000040`: minor table fragment in mass mask
- `000015_000010`: minor table fragment in mass mask
- `000015_000040`: visible table streak in mass mask; object remains identifiable
- `000017_000035`: weak background/mask contamination; visual review required
- `000018_000010`: minor background fragment in mass mask
- `000021_000040`: minor background fragment in mass mask

RGB/depth 동일 index·해상도 및 per-frame camera K는 확인했다. 픽셀 단위의 물리적 정합 오차 (예: reprojection residual)는 별도 실측 calibration 없이 증명하지 못한다. 원본 mask는 생성하지 않았으며 M1 depth/RGB 분할 결과 overlay를 파생물로 보존했다. mask 수동 수정 없음.

실측 GT는 동일 EV-RealPhys 실물 5개에만 대응하며 일반 YCB 이미지로 옮기지 않는다. `source_trial_count=10`은 원 논문의 마찰 측정 활주 횟수로, 이 manifest의 객체별 정지 이미지 10장이나 추후 모델 5회 반복과 다른 수다. Mustard Bottle, Pitcher, Bleach Cleanser의 숨겨진 모래 충전 상태는 단일 이미지에서 식별하기 어렵고 모델 입력으로 제공하지 않는다. 정지 시각 마찰도 감속도 기반 물리 측정치가 아니라 시각 prior로만 해석한다.
