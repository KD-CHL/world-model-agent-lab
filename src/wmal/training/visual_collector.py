"""Collect genuine rendered action consequences under a fixed G1 joint servo."""
from pathlib import Path
from hashlib import sha256
import json
import numpy as np

from wmal.datasets.visual_sequences import assign_four_splits, write_episode, write_manifest


def collect_visual_workcell(output, *, task='stack_block',episodes=40,steps=16,image_size=64,seed=0,
                           controller='legacy',target_low=None,target_high=None):
    from wmal.envs.visual_workcell import VisualWorkcellSession, TARGET_LOW, TARGET_HIGH, MAX_DELTA
    output=Path(output).resolve()
    if output.exists() and any(output.iterdir()):
        raise ValueError('Collection output is nonempty')
    if type(steps) is not int or steps<2:
        raise ValueError('Need at least two transitions per episode')
    if (target_low is None)!=(target_high is None):
        raise ValueError('Local collection requires both target bounds')
    scoped=target_low is not None
    lower=TARGET_LOW if not scoped else np.asarray(target_low,dtype=float)
    upper=TARGET_HIGH if not scoped else np.asarray(target_high,dtype=float)
    if (lower.shape!=(2,) or upper.shape!=(2,) or not np.isfinite(lower).all()
            or not np.isfinite(upper).all() or np.any(upper-lower<=.01)
            or np.any(lower<TARGET_LOW) or np.any(upper>TARGET_HIGH)):
        raise ValueError('Collection scope must be inside the fixed joint envelope')
    scope={'command_lower':lower.tolist(),'command_upper':upper.tolist(),
           'measured_margin':.15,'max_action_delta':MAX_DELTA} if scoped else None
    split=assign_four_splits(episodes,seed)
    rng=np.random.default_rng(seed)
    output.mkdir(parents=True,exist_ok=True)
    rows=[]
    with VisualWorkcellSession(task,image_size=image_size,seed=seed,controller=controller) as session:
        for episode in range(episodes):
            before=session.reset(rng.uniform(lower if scoped else lower+.1,upper if scoped else upper-.1))
            rgb=[np.rint(before.rgb.transpose(1,2,0)*255).astype('uint8')]
            states,actions=[before.state],[]
            for _ in range(steps):
                target=before.state[2:]
                low=np.maximum(-MAX_DELTA,lower+.005-target)
                high=np.minimum(MAX_DELTA,upper-.005-target)
                action=rng.uniform(low,high).astype('float32')
                after=session.execute_actions(action[None],episode_id=before.episode_id,step_id=before.step_id)
                actions.append(action)
                states.append(after.state)
                rgb.append(np.rint(after.rgb.transpose(1,2,0)*255).astype('uint8'))
                before=after
            name=f'episode_{episode:06d}.npz'
            write_episode(output/name,np.stack(rgb),np.stack(states),np.stack(actions))
            namespace='workcell' if controller=='legacy' else 'workcell_gc2'
            if scoped:
                namespace+='_'+sha256(json.dumps(scope,sort_keys=True).encode()).hexdigest()[:8]
            rows.append({'episode_id':f'{namespace}_{seed}_{episode}','split':split[episode],'path':name})
        semantics=session.semantics
        source={'kind':'mujoco_g1_arm_servo','task':task,'seed':seed,'scene':session.scene.metadata(),
                'controller_profile':session.controller_profile,
                'target_envelope_rad':[lower.tolist(),upper.tolist()],
                'max_target_delta_rad':MAX_DELTA,
                'limitations':'Joint motion only; no grasp success/contact safety labels; fixed base'}
        if scoped:
            source['prediction_scope']=scope
    return write_manifest(output/'manifest.json',rows,**semantics,source=source)
