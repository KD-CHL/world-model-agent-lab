"""Collect bounded G1 simulation actions into explicitly separated episodes."""
from dataclasses import asdict
from pathlib import Path
import numpy as np
from wmal.logging.manifest import atomic_json, build_manifest
from wmal.locomotion.contracts import G1VelocityAction


def collect_motion(output, *, train_episodes=8, validation_episodes=3, test_episodes=3,
                   steps=100, duration_s=.5, seed=0, progress=None):
    counts = {'train':train_episodes,'validation':validation_episodes,'test':test_episodes}
    if any(type(n) is not int or n<1 for n in (*counts.values(),steps)) or type(seed) is not int or seed<0:
        raise ValueError('Positive episode/step counts and a nonnegative seed are required')
    G1VelocityAction(0.,0.,0.,duration_s)
    if Path(output).exists():
        raise FileExistsError('Choose a new output; collection does not overwrite existing data')
    from wmal.envs.g1_session import G1MuJoCoSession
    from wmal.envs.g1_locomotion import POLICY_ONNX, MODEL_XML
    manifest = build_manifest('motion_collection',seed,{'policy':POLICY_ONNX,'robot_xml':MODEL_XML},
                              {'episodes':counts,'steps':steps,'duration_s':duration_s},
                              data_source='mujoco_interaction')
    rng, rows, episodes, failures = np.random.default_rng(seed), [], [], []
    emit = progress or (lambda row:None)
    for split,count in counts.items():
        for index in range(count):
            completed = 0
            with G1MuJoCoSession(viewer=False,realtime=False) as session:
                episode_id = session.observe().episode_id
                for _ in range(steps):
                    before = session.observe()
                    action = G1VelocityAction(float(rng.uniform(-.2,.3)),float(rng.uniform(-.1,.1)),
                                              float(rng.uniform(-.3,.3)),duration_s)
                    try:
                        after = session.step(action)
                    except (RuntimeError, ValueError) as exc:
                        failures.append({'episode_id':episode_id,'split':split,'step_id':before.step_id,
                                         'action':asdict(action),'reason':type(exc).__name__,
                                         'outcome':'partial_or_failed_simulation_action'})
                        break
                    rows.append({'split':split,'before':asdict(before),'action':asdict(action),'after':asdict(after)})
                    completed += 1
            episodes.append({'episode_id':episode_id,'split':split,'transitions':completed})
            atomic_json(output,{'schema':'wmal.g1.transitions.v1','manifest':manifest,'rows':rows,
                                'episodes':episodes,'failed_episodes':failures})
            emit({'phase':'collection','split':split,'episode':index,'transitions':completed,
                  'total_transitions':len(rows),'failed_actions':len(failures)})
    return {'output':str(output),'transitions':len(rows),'failed_actions':len(failures),'episodes':counts}
