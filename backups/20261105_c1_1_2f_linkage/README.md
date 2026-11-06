# Backup — C1_1 2F linkage initialization (2026-11-05)

Change: C1_1 + 2F now runs the corrected 2F linkage path (repair_spoon_parallel_2f_xml,
linkage self-contact exclusion, gripper joints zeroed after reset) instead of the
native-hand early return. Goal: remove the PRESHAPE GRIPPER_JOINT_LIMIT_EXCEEDED
(left_inner_knuckle q=-0.196 rad) seen on every C1_1 2F probe.

Risk: the C1_1 plate 2F production route uses the same profile.

Undo (pick one):
  1. No code change:  $env:TUJ_C1_1_2F_LINKAGE_INIT = "0"   (PowerShell, current window)
  2. Restore file:    copy backups\20261105_c1_1_2f_linkage\profiles.py.before src\tuj\m5_motion\scripted_grasps\profiles.py
  3. Back to git HEAD (also drops the earlier catalog_hand_correction fix):
                      git checkout -- src/tuj/m5_motion/scripted_grasps/profiles.py

Files:
  profiles.py.before    working copy right before this change (includes catalog_hand_correction fix)
  profiles.py.git_HEAD  committed version at HEAD
