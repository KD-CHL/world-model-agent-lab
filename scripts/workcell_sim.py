"""Inspect or validate the twelve dataset-inspired G1 workcell worlds."""
import argparse
from pathlib import Path
import time

import numpy as np
import mujoco

from wmal.envs.workcell import TASKS, build_workcell
from wmal.logging.manifest import atomic_json


def snapshot(scene, model, data):
    return {**scene.metadata(),'sim_time_s':float(data.time),
            'objects':{name:{'position':data.body(name).xpos.tolist(),
                             'quaternion':data.body(name).xquat.tolist()} for name in scene.objects},
            'dimensions':{'nq':model.nq,'nu':model.nu,'nflex':model.nflex},
            'finite':bool(np.isfinite(data.qpos).all() and np.isfinite(data.qvel).all()),
            'warnings':{str(i):int(w.number) for i,w in enumerate(data.warning) if w.number}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--task',choices=TASKS,default='stack_block')
    parser.add_argument('--all',action='store_true',help='Validate all scenes without a window')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--steps',type=int,default=1000)
    parser.add_argument('--seed',type=int,default=0)
    parser.add_argument('--output',default='runs/workcells')
    parser.add_argument('--render',action='store_true',help='Export three camera PNGs per scene (requires Pillow)')
    args = parser.parse_args(argv)
    if args.steps < 1:
        parser.error('--steps must be positive')
    output = Path(args.output)
    failures = []
    for task in TASKS if args.all else [args.task]:
        scene,model,data = build_workcell(task,args.seed)
        initial_qpos,initial_ctrl = data.qpos.copy(),data.ctrl.copy()
        if args.headless or args.all:
            mujoco.mj_step(model,data,nstep=args.steps)
        else:
            from mujoco import viewer as mj_viewer
            reset = []
            def key_callback(key):
                if key in (82,114):
                    reset.append(True)
            with mj_viewer.launch_passive(model,data,key_callback=key_callback) as viewer:
                viewer.cam.lookat[:] = [.75,0.,.85]
                viewer.cam.distance,viewer.cam.azimuth,viewer.cam.elevation = 2.8,135,-30
                print('R: reset world. Close window to exit. Robot holds posture; objects remain physical.')
                while viewer.is_running():
                    start = time.monotonic()
                    if reset:
                        reset.clear()
                        mujoco.mj_resetData(model,data)
                        data.qpos[:],data.ctrl[:] = initial_qpos,initial_ctrl
                        mujoco.mj_forward(model,data)
                    mujoco.mj_step(model,data,nstep=10)
                    viewer.sync()
                    time.sleep(max(0.,.02-(time.monotonic()-start)))
        report = snapshot(scene,model,data)
        report['objects_above_floor'] = all(data.body(name).xpos[2] > 0 for name in scene.objects)
        if not report['finite'] or report['warnings'] or not report['objects_above_floor']:
            failures.append(task)
        atomic_json(output / task / 'scene.json',report)
        if args.render:
            from PIL import Image
            with mujoco.Renderer(model,height=480,width=640) as renderer:
                for camera in scene.metadata()['cameras']:
                    renderer.update_scene(data,camera=camera)
                    Image.fromarray(renderer.render()).save(output / task / (camera+'.png'))
        print(task, 'finite=',report['finite'], 'warnings=',report['warnings'],flush=True)
    return 1 if failures else 0


if __name__ == '__main__':
    raise SystemExit(main())
