# RoboCasa static-to-sim audit

- commit: `4f8a2980def75a55dff96b990745b83540425f09`
- dataset/assets downloaded: no

## EV-RealPhys object mapping

| object_id | aliases found in source/docs | exact mapping status |
|---|---|---|
| 003_cracker_box | boxed_food | UNVERIFIED |
| 006_mustard_bottle | mustard | UNVERIFIED |
| 019_pitcher_base | pitcher | UNVERIFIED |
| 021_bleach_cleanser | cleaner | UNVERIFIED |
| 025_mug | mug | UNVERIFIED |

## Capability audit

| capability | status |
|---|---|
| object_scale | SUPPORTED_IN_SOURCE |
| object_world_pose | AVAILABLE_AT_REPLAY |
| camera_pose | SUPPORTED_IN_SOURCE |
| table_plane_contact | SUPPORTED_BY_MUJOCO |
| scene_replay | SUPPORTED_IN_DATASET |
| vacuum_suction_pressure | UNVERIFIED |

RoboCasa source support does not prove that an EV-RealPhys image is the same object instance. Exact mapping requires downloaded episode/object asset metadata and image-to-episode matching.
