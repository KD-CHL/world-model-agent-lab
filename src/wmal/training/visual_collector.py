"""Collect genuine rendered action consequences under a fixed G1 joint servo."""
from pathlib import Path
import numpy as np

from wmal.datasets.visual_sequences import assign_four_splits, write_episode, write_manifest


def collect_visual_workcell(output, *, task='stack_block',episodes=40,steps=16,image_size=64,seed=0):
    from wmal.envs.visual_workcell import VisualWorkcellSession, TARGET_LOW, TARGET_HIGH, MAX_DELTA
    output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Collection output is nonempty')
    if type(steps) is not int or steps<2:
        raise ValueError('Need at least two transitions per episode')
    split=assign_four_splits(episodes,seed)
    rng=np.random.default_rng(seed)
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    with VisualWorkcellSession(task,image_size=image_size,seed=seed) as session:
        for episode in range(episodes):
            before=session.reset(rng.uniform(TARGET_LOW+.1,TARGET_HIGH-.1))
            rgb=[np.rint(before.rgb.transpose(1,2,0)*255).astype('uint8')]
            states,actions=[before.state],[]
            for _ in range(steps):
                target=before.state[2:]
                low=np.maximum(-MAX_DELTA,TARGET_LOW+.005-target)
                high=np.minimum(MAX_DELTA,TARGET_HIGH-.005-target)
                action=rng.uniform(low,high).astype('float32')
                after=session.execute_actions(action[None],episode_id=before.episode_id,step_id=before.step_id)
                actions.append(action)
                states.append(after.state)
                rgb.append(np.rint(after.rgb.transpose(1,2,0)*255).astype('uint8'))
                before=after
            name=f'episode_{episode:06d}.npz'
            write_episode(output/name,np.stack(rgb),np.stack(states),np.stack(actions))
            rows.append({'episode_id':f'workcell_{seed}_{episode}','split':split[episode],'path':name})
        semantics=session.semantics
        source={'kind':'mujoco_g1_arm_servo','task':task,'seed':seed,'scene':session.scene.metadata(),
                'target_envelope_rad':[TARGET_LOW.tolist(),TARGET_HIGH.tolist()],
                'max_target_delta_rad':MAX_DELTA,
                'limitations':'Joint motion only; no grasp success/contact safety labels; fixed base'}
    return write_manifest(output/'manifest.json',rows,**semantics,source=source)
