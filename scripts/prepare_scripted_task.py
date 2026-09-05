"""Replan existing Gemini M1-M3 outputs with the scripted grasp EE contract.

Writes separate M4 files. It neither changes the task instruction nor calls a
model. Use the resulting m4-scripted.json with scripts/run_m5.py.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT))


def main():
    from task_registry import TASK_ENVS
    from tuj.m4_taskplanner.gk_adapter import build_request_from_gk
    from tuj.m4_taskplanner.planner import plan
    from tuj.m5_motion.scripted_grasps.task_constraints import constrain_task_request, scene_id_aliases
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('task',choices=sorted(TASK_ENVS))
    parser.add_argument('--input-dir',type=Path,required=True)
    parser.add_argument('--initial-ee',choices=['2F','3F','vac'],required=True)
    args=parser.parse_args()
    folder=args.input_dir.resolve()
    def read(path): return json.loads(path.read_text(encoding='utf-8'))
    m1=read(folder/'m1.json')
    aliases=scene_id_aliases(m1)
    request=build_request_from_gk(read(folder/'gk_bundle.json'),read(folder/'m2.json'),
        m1_payload=m1,robot_spec_payload=read(ROOT/'configs/robot_spec.json'),id_aliases=aliases)
    request.task_graph.initial_state.current_ee=args.initial_ee
    request,changes=constrain_task_request(request,TASK_ENVS[args.task])
    result=plan(request)
    (folder/'m4-scripted-request.json').write_text(request.model_dump_json(indent=2),encoding='utf-8')
    (folder/'m4-scripted.json').write_text(result.model_dump_json(indent=2),encoding='utf-8')
    (folder/'m4-scripted-contract.json').write_text(json.dumps({'initial_ee':args.initial_ee,
        'environment':TASK_ENVS[args.task],'changes':changes,'id_aliases':aliases,
        'status':result.status.value},ensure_ascii=False,indent=2),encoding='utf-8')
    print(result.status.value)
    return 0 if result.status.value=='SUCCESS' else 2


if __name__=='__main__':
    raise SystemExit(main())
