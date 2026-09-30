"""Neural G1 motion ensemble with a differentiable pose integration layer."""
from dataclasses import replace
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile

import numpy as np

from wmal.locomotion.contracts import G1_ACTION_SCHEMA, G1_STATE_SCHEMA, G1Prediction
from wmal.models.neural_dynamics import _torch_modules

STATE_FIELDS = ('x', 'y', 'yaw', 'vx', 'vy', 'yaw_rate', 'roll', 'pitch', 'pelvis_height')
STATE_SCALES = (.1, .1, .2, .3, .3, .5, .2, .2, .1)
SCHEMA = 'wmal.motion_network.v1'
LATENT_SCHEMA = 'wmal.motion_network.v2'


def build_network(hidden, architecture='physical'):
    _, nn = _torch_modules()
    if not hidden or any(type(width) is not int or width < 4 for width in hidden):
        raise ValueError('Hidden widths must be integers >=4')
    if architecture == 'latent_state':
        # A state representation, action-conditioned transition, and physical head.
        # Checkpoints store tensor dictionaries rather than pickled model objects.
        torch, _ = _torch_modules()
        def mlp(input_width, output_width):
            layers, width = [], input_width
            for next_width in hidden:
                layers.extend([nn.Linear(width, next_width), nn.SiLU()])
                width = next_width
            layers.append(nn.Linear(width, output_width))
            return nn.Sequential(*layers)
        class StateTransitionNetwork(nn.Module):
            def __init__(self):
                super().__init__()
                self.encoder = nn.Sequential(mlp(6, hidden[-1]), nn.LayerNorm(hidden[-1]))
                self.transition = mlp(hidden[-1]+4, hidden[-1])
                self.readout = mlp(2*hidden[-1]+4, 9)
                self.reconstruction = nn.Linear(hidden[-1], 6)

            def advance(self, latent, normalized_action):
                joined = torch.cat([latent, normalized_action], dim=-1)
                next_latent = latent + self.transition(joined)
                output = self.readout(torch.cat([latent, next_latent, normalized_action], dim=-1))
                return next_latent, output

            def forward(self, x):
                return self.advance(self.encoder(x[..., :6]), x[..., 6:])[1]
        return StateTransitionNetwork()
    if architecture != 'physical':
        raise ValueError('Unknown motion network architecture')
    layers, width = [], 10
    for next_width in hidden:
        layers.extend([nn.Linear(width, next_width), nn.SiLU()])
        width = next_width
    layers.append(nn.Linear(width, 6))
    return nn.Sequential(*layers)


def state_vector(state):
    return [getattr(state, key) for key in STATE_FIELDS]


def features(states, actions):
    torch, _ = _torch_modules()
    c, s = torch.cos(states[...,2]), torch.sin(states[...,2])
    return torch.stack([c*states[...,3]+s*states[...,4], -s*states[...,3]+c*states[...,4],
        states[...,5], states[...,6], states[...,7], states[...,8],
        actions[...,0], actions[...,1], actions[...,2], actions[...,3]], dim=-1)


def target_outputs(before, after, actions, architecture='physical'):
    torch, _ = _torch_modules()
    c, s = torch.cos(before[...,2]), torch.sin(before[...,2])
    dx, dy, duration = after[...,0]-before[...,0], after[...,1]-before[...,1], actions[...,3]
    yaw = torch.atan2(torch.sin(after[...,2]-before[...,2]), torch.cos(after[...,2]-before[...,2]))
    values = [(c*dx+s*dy)/duration, (-s*dx+c*dy)/duration, yaw/duration,
              after[...,6], after[...,7], after[...,8]]
    if architecture == 'latent_state':
        # Terminal velocity uses the *terminal* body frame; displacement uses the initial frame.
        ac, ass = torch.cos(after[...,2]), torch.sin(after[...,2])
        values.extend([ac*after[...,3]+ass*after[...,4],
                       -ass*after[...,3]+ac*after[...,4], after[...,5]])
    return torch.stack(values, dim=-1)


def integrate(states, outputs, actions):
    torch, _ = _torch_modules()
    c, s = torch.cos(states[...,2]), torch.sin(states[...,2])
    vx, vy = c*outputs[...,0]-s*outputs[...,1], s*outputs[...,0]+c*outputs[...,1]
    yaw = states[...,2]+outputs[...,2]*actions[...,3]
    if outputs.shape[-1] == 9:
        terminal_vx = torch.cos(yaw)*outputs[...,6]-torch.sin(yaw)*outputs[...,7]
        terminal_vy = torch.sin(yaw)*outputs[...,6]+torch.cos(yaw)*outputs[...,7]
        terminal_rate = outputs[...,8]
    else:
        terminal_vx, terminal_vy, terminal_rate = vx, vy, outputs[...,2]
    return torch.stack([states[...,0]+vx*actions[...,3], states[...,1]+vy*actions[...,3],
        torch.atan2(torch.sin(yaw), torch.cos(yaw)), terminal_vx, terminal_vy, terminal_rate,
        outputs[...,3], outputs[...,4], outputs[...,5]], dim=-1)


def atomic_torch_save(path, payload):
    torch, _ = _torch_modules()
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(dir=path.parent, prefix=path.name+'.')
    os.close(descriptor)
    try:
        torch.save(payload, temporary)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def cpu_states(models):
    return [{k:v.detach().cpu().clone() for k,v in model.state_dict().items()} for model in models]


def content_version(payload):
    identity = {key:payload[key] for key in ('hidden','duration_s','normalization')}
    if payload.get('schema') == LATENT_SCHEMA:
        identity.update(schema=LATENT_SCHEMA, architecture=payload['architecture'])
    digest = hashlib.sha256(json.dumps(identity,
                                      sort_keys=True, allow_nan=False).encode())
    for member in payload['state_dicts']:
        for name, value in sorted(member.items()):
            digest.update(name.encode())
            digest.update(value.detach().cpu().contiguous().numpy().tobytes())
    return 'motion-' + digest.hexdigest()[:16]


class MotionNetwork:
    state_schema, action_schema = G1_STATE_SCHEMA, G1_ACTION_SCHEMA

    def __init__(self, payload, device='cpu'):
        torch, _ = _torch_modules()
        if payload.get('schema') not in (SCHEMA,LATENT_SCHEMA) or len(payload.get('state_dicts', [])) < 2:
            raise ValueError('Invalid neural motion checkpoint')
        self.schema = payload['schema']
        self.architecture = payload.get('architecture', 'physical')
        if (self.schema == LATENT_SCHEMA) != (self.architecture == 'latent_state'):
            raise ValueError('Checkpoint schema/architecture mismatch')
        self.hidden = tuple(payload['hidden'])
        self.duration_s = payload['duration_s']
        if not math.isfinite(self.duration_s) or not .05 <= self.duration_s <= 1:
            raise ValueError('Invalid model action duration')
        self.normalization = payload['normalization']
        output_width = 9 if self.architecture == 'latent_state' else 6
        for key, width in (('x_mean',10),('x_scale',10),('y_mean',output_width),('y_scale',output_width)):
            values = np.asarray(self.normalization[key])
            if values.shape != (width,) or not np.isfinite(values).all() or ('scale' in key and np.any(values <= 0)):
                raise ValueError('Invalid model normalization')
        self.device = torch.device(device)
        self.stats = {key:torch.tensor(value,dtype=torch.float32,device=self.device)
                      for key,value in self.normalization.items()}
        self.models = []
        for weights in payload['state_dicts']:
            model = build_network(self.hidden,self.architecture).to(self.device)
            model.load_state_dict(weights)
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError('Nonfinite network weights')
            self.models.append(model.eval())
        self.version = content_version(payload)
        if payload.get('version', self.version) != self.version:
            raise ValueError('Checkpoint content/version mismatch')
        self.metadata = payload.get('metadata', {})

    def _validate(self, state, action, duration_s):
        if (not math.isfinite(duration_s) or abs(duration_s-self.duration_s)>1e-9
                or abs(duration_s-action.duration_s)>1e-9):
            raise ValueError('Prediction duration differs from training')
        if set(state.joint_positions) != set(self.metadata.get('joint_order', state.joint_positions)):
            raise ValueError('Observation joint schema differs from training')

    def rollout(self, state, candidates):
        """Keep each member's own state and latent across the entire candidate horizon."""
        from wmal.locomotion.rollouts import EnsembleRollout
        torch, _ = _torch_modules()
        if not candidates or not candidates[0] or any(len(row)!=len(candidates[0]) for row in candidates):
            raise ValueError('Rollout requires a rectangular nonempty action batch')
        for row in candidates:
            for action in row:
                self._validate(state,action,action.duration_s)
        actions = torch.tensor([[[a.vx,a.vy,a.yaw_rate,a.duration_s] for a in row] for row in candidates],
                               dtype=torch.float32,device=self.device)
        initial = torch.tensor([state_vector(state)]*len(candidates),dtype=torch.float32,device=self.device)
        trajectories = []
        with torch.inference_mode():
            for model in self.models:
                current, latent, path = initial, None, []
                for step in range(actions.shape[1]):
                    action = actions[:,step]
                    x = (features(current,action)-self.stats['x_mean'])/self.stats['x_scale']
                    if self.architecture == 'latent_state':
                        if latent is None:
                            latent = model.encoder(x[...,:6])
                        latent, output = model.advance(latent,x[...,6:])
                    else:
                        output = model(x)
                    current = integrate(current,output*self.stats['y_scale']+self.stats['y_mean'],action)
                    path.append(current)
                trajectories.append(torch.stack(path,dim=1))
            outcomes = torch.stack(trajectories).cpu().numpy()
        return EnsembleRollout(self.version, state, tuple(tuple(row) for row in candidates), outcomes)

    def predict(self, state, action, duration_s):
        self._validate(state,action,duration_s)
        return self.rollout(state, [[action]]).prediction(0,0)

    def save(self, path):
        payload = {'schema':self.schema,'hidden':list(self.hidden),'duration_s':self.duration_s,
                   'normalization':self.normalization,'state_dicts':cpu_states(self.models),
                   'metadata':self.metadata}
        if self.architecture == 'latent_state':
            payload['architecture'] = self.architecture
        payload['version'] = content_version(payload)
        atomic_torch_save(path,payload)


def load(config):
    torch, _ = _torch_modules()
    payload = torch.load(config['checkpoint'],map_location='cpu',weights_only=True)
    return MotionNetwork(payload,device=config.get('device','cpu'))
