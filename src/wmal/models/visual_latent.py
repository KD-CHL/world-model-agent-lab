"""Compact RGB-conditioned recurrent ensemble, not a UniFoLM/Dreamer replica.

Each independently trained member imagines from the SAME real observation.
No target frames are consumed during candidate rollout. No policy is trained.
"""
from dataclasses import asdict, dataclass
import hashlib
import json

import numpy as np
import torch
from torch import nn

from wmal.models.motion_network import atomic_torch_save

SCHEMA='wmal.visual_latent.v2'


@dataclass(frozen=True)
class NetworkConfig:
    state_dim: int
    action_dim: int
    image_size: int = 64
    latent_dim: int = 64
    hidden_dim: int = 128
    event_dim: int = 0
    architecture: str = 'deterministic'
    stoch: int = 8
    classes: int = 8
    unimix: float = .01

    def __post_init__(self):
        if (any(type(v) is not int or v<1 for v in
                (self.state_dim,self.action_dim,self.latent_dim,self.hidden_dim))
                or self.image_size not in (32,64,128) or type(self.event_dim) is not int or self.event_dim<0):
            raise ValueError('Invalid visual network dimensions')
        if (self.architecture not in ('deterministic','categorical_rssm')
                or any(type(v) is not int or v<2 for v in (self.stoch,self.classes))
                or not np.isfinite(self.unimix) or not 0<self.unimix<1):
            raise ValueError('Invalid RSSM architecture or categorical dimensions')


class VisualLatentMember(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config=config
        z,h=config.latent_dim,config.hidden_dim
        self.encoder=nn.Sequential(nn.Conv2d(3,16,4,2,1),nn.SiLU(),
                                  nn.Conv2d(16,32,4,2,1),nn.SiLU(),
                                  nn.Conv2d(32,32,4,2,1),nn.SiLU(),
                                  nn.AdaptiveAvgPool2d((4,4)),nn.Flatten(),
                                  nn.Linear(512,z),nn.LayerNorm(z))
        self.initial=nn.Sequential(nn.Linear(z+config.state_dim,h),nn.Tanh())
        self.transition=nn.GRUCell(z+config.state_dim+config.action_dim,h)
        self.latent=nn.Sequential(nn.Linear(h,z),nn.LayerNorm(z))
        self.state_head=nn.Sequential(nn.Linear(h+z,h),nn.SiLU(),nn.Linear(h,config.state_dim))
        self.event_head=nn.Linear(h+z,config.event_dim) if config.event_dim else None
        self.decoder_input=nn.Linear(z+h,32*4*4)
        layers=[]
        size, channels=4,32
        while size<config.image_size:
            output=3 if size*2==config.image_size else 16
            layers.extend([nn.ConvTranspose2d(channels,output,4,2,1),
                           nn.Sigmoid() if output==3 else nn.SiLU()])
            size*=2
            channels=output
        self.decoder=nn.Sequential(*layers)

    def imagine(self, rgb, state, actions):
        latent=self.encoder(rgb)
        hidden=self.initial(torch.cat([latent,state],-1))
        frames, states, latents, events=[],[],[],[]
        for action in actions.unbind(1):
            hidden=self.transition(torch.cat([latent,state,action],-1),hidden)
            latent=latent+.1*self.latent(hidden)
            features=torch.cat([hidden,latent],-1)
            state=state+self.state_head(features)
            residual=2.*self.decoder(self.decoder_input(torch.cat([latent,hidden],-1)).reshape(-1,32,4,4))-1.
            frames.append((rgb+residual).clamp(0.,1.))
            states.append(state)
            latents.append(latent)
            if self.event_head is not None:
                events.append(self.event_head(features))
        return {'rgb':torch.stack(frames,1),'state':torch.stack(states,1),
                'latent':torch.stack(latents,1),
                'event_logits':torch.stack(events,1) if events else None}


def make_member(config):
    if config.architecture=='categorical_rssm':
        from wmal.models.categorical_rssm import CategoricalRSSMMember
        return CategoricalRSSMMember(config)
    return VisualLatentMember(config)


def _version(config, semantics, normalization, states):
    configuration=asdict(config)
    if config.architecture=='deterministic':
        # v2 checksums predate these fields: don't invalidate existing calibration.
        for key in ('architecture','stoch','classes','unimix'):
            configuration.pop(key)
    digest=hashlib.sha256(json.dumps({'config':configuration,'semantics':semantics,
                                     'normalization':normalization},sort_keys=True).encode())
    for state in states:
        for key,tensor in sorted(state.items()):
            digest.update(key.encode())
            digest.update(tensor.detach().cpu().contiguous().numpy().tobytes())
    return 'visual-'+digest.hexdigest()[:20]


class VisualWorldModel:
    def __init__(self, config, semantics, normalization, members, *, device='cpu', metadata=None):
        if len(members)<2 or len(semantics['state_order'])!=config.state_dim or len(semantics['action_order'])!=config.action_dim:
            raise ValueError('World model requires >=2 members and matching named semantics')
        if semantics['image_size']!=config.image_size:
            raise ValueError('Camera resolution mismatch')
        for prefix,width in [('state',config.state_dim),('action',config.action_dim)]:
            for key in ('mean','scale'):
                value=np.asarray(normalization[prefix+'_'+key])
                if value.shape!=(width,) or not np.isfinite(value).all() or (key=='scale' and np.any(value<=0)):
                    raise ValueError('Invalid model normalization')
        self.config,self.semantics,self.normalization=config,dict(semantics),normalization
        self.members=[member.to(device).eval() for member in members]
        self.device,self.metadata=device,metadata or {}
        self.version=_version(config,semantics,normalization,[m.state_dict() for m in members])

    def predict(self, rgb, state, actions):
        rgb,state,actions=np.asarray(rgb),np.asarray(state),np.asarray(actions)
        size=self.config.image_size
        if (rgb.shape!=(3,size,size) or state.shape!=(self.config.state_dim,)
                or actions.ndim!=2 or actions.shape[1]!=self.config.action_dim or len(actions)<1
                or not all(np.isfinite(v).all() for v in (rgb,state,actions))
                or rgb.min()<0 or rgb.max()>1):
            raise ValueError('Expected finite CHW RGB [0,1], state and [H,A] candidate actions')
        if len(actions)>self.metadata.get('max_horizon',len(actions)):
            raise ValueError('Candidate horizon exceeds trained horizon')
        norm=self.normalization
        normalized_state=(state-np.asarray(norm['state_mean']))/np.asarray(norm['state_scale'])
        normalized_actions=(actions-np.asarray(norm['action_mean']))/np.asarray(norm['action_scale'])
        inputs=[torch.as_tensor(v,dtype=torch.float32,device=self.device)[None]
                for v in (rgb,normalized_state,normalized_actions)]
        states,frames,events,diagnostics=[],[],[],[]
        with torch.inference_mode():
            for member in self.members:
                output=member.imagine(*inputs)
                states.append(output['state'][0].cpu().numpy()*norm['state_scale']+norm['state_mean'])
                frames.append(output['rgb'][0].cpu().numpy())
                if output['event_logits'] is not None:
                    events.append(output['event_logits'][0].sigmoid().cpu().numpy())
                if 'prior_entropy' in output:
                    diagnostics.append({'prior_entropy':output['prior_entropy'][0].cpu().tolist(),
                                        'posterior_entropy':float(output['posterior_entropy'][0])})
        if not all(np.isfinite(v).all() for v in states+frames+events):
            raise RuntimeError('Nonfinite world-model prediction; refuse candidate')
        return {'model_version':self.version,'states':np.stack(states),'frames':np.stack(frames),
                'event_probabilities':np.stack(events) if events else None,
                'diagnostics':{'architecture':self.config.architecture,
                               'inference_mode':'categorical_probability_proxy' if diagnostics else 'deterministic',
                               'uncertainty_kind':'ensemble_spread_not_calibrated',
                               'members':diagnostics}}

    def predict_video(self, observation_history, action_sequence):
        """VideoPredictionProvider interface: history entries contain rgb/state/semantics."""
        observation=observation_history[-1]
        if observation.get('semantics')!=self.semantics:
            raise ValueError('Observation/control semantics differ from model')
        rgb=np.asarray(observation['rgb'])
        if rgb.dtype!=np.uint8 or rgb.shape!=(self.config.image_size,self.config.image_size,3):
            raise ValueError('History must contain HWC uint8 RGB')
        result=self.predict(rgb.transpose(2,0,1).astype('float32')/255.,observation['state'],action_sequence)
        return np.rint(result['frames'].mean(0).transpose(0,2,3,1)*255).clip(0,255).astype('uint8')

    def save(self, path):
        states=[{key:value.detach().cpu() for key,value in member.state_dict().items()} for member in self.members]
        atomic_torch_save(path,{'schema':SCHEMA,'network':asdict(self.config),'semantics':self.semantics,
                               'normalization':self.normalization,'members':states,'metadata':self.metadata,
                               'model_version':self.version})

    @classmethod
    def load(cls,path,*,device='cpu'):
        payload=torch.load(path,map_location='cpu',weights_only=True)
        if payload.get('schema')!=SCHEMA:
            raise ValueError('Checkpoint is not a WMAL visual network; WMA weights need external adapter')
        config=NetworkConfig(**payload['network'])
        members=[]
        for weights in payload['members']:
            if any(not torch.isfinite(value).all() for value in weights.values()):
                raise ValueError('Nonfinite checkpoint weights')
            member=make_member(config)
            member.load_state_dict(weights,strict=True)
            members.append(member)
        result=cls(config,payload['semantics'],payload['normalization'],members,
                   device=device,metadata=payload['metadata'])
        if result.version!=payload['model_version']:
            raise ValueError('Checkpoint content/version mismatch')
        return result
