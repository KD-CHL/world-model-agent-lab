"""Compare batched and per-candidate inference with the same member-preserving semantics."""
import argparse
import time
import numpy as np
from wmal.datasets.motion_sequences import load_motion_episodes
from wmal.locomotion.contracts import G1VelocityAction
from wmal.logging.manifest import atomic_json,build_manifest
from wmal.models.motion_network import load


def main():
    import torch
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset',required=True)
    parser.add_argument('--checkpoint',required=True)
    parser.add_argument('--output',required=True)
    parser.add_argument('--samples',type=int,default=48)
    parser.add_argument('--horizon',type=int,default=3)
    parser.add_argument('--repeats',type=int,default=20)
    parser.add_argument('--threads',type=int,default=1)
    args=parser.parse_args()
    if min(args.samples,args.horizon,args.repeats,args.threads)<1:
        parser.error('Counts must be positive')
    torch.set_num_threads(args.threads)
    splits,duration=load_motion_episodes(args.dataset)
    state=next(iter(splits['validation'].values()))[0][0]
    model=load({'checkpoint':args.checkpoint,'device':'cpu'})
    rng=np.random.default_rng(7)
    candidates=[[G1VelocityAction(float(rng.uniform(-.2,.3)),float(rng.uniform(-.1,.1)),
                 float(rng.uniform(-.3,.3)),duration) for _ in range(args.horizon)] for _ in range(args.samples)]
    def batch(): return model.rollout(state,candidates).members
    def serial(): return np.concatenate([model.rollout(state,[row]).members for row in candidates],axis=1)
    np.testing.assert_allclose(batch(),serial(),atol=2e-6,rtol=1e-5)
    timings={key:[] for key in ('batch','serial')}
    # Warm both paths; alternate order to reduce systematic timing bias.
    batch(); serial()
    for index in range(args.repeats):
        order=(('batch',batch),('serial',serial)) if index%2==0 else (('serial',serial),('batch',batch))
        for name,call in order:
            started=time.perf_counter()
            call()
            timings[name].append((time.perf_counter()-started)*1000.)
    report=build_manifest('motion_rollout_latency',7,{'dataset':args.dataset,'checkpoint':args.checkpoint},
        {'device':'cpu','threads':args.threads,'samples':args.samples,'horizon':args.horizon,'repeats':args.repeats})
    report.update(model_version=model.version,scope='inference only; excludes scene scoring, ROS 2 and physical execution',
        equivalent_outputs_checked=True,latency_ms={key:{'p50':float(np.percentile(v,50)),
            'p95':float(np.percentile(v,95)),'samples':v} for key,v in timings.items()},
        batch_model_step_calls=len(model.models)*args.horizon,
        serial_model_step_calls=len(model.models)*args.horizon*args.samples,
        model_imagination_steps=len(model.models)*args.horizon*args.samples)
    atomic_json(args.output,report)
    print({key:value for key,value in report['latency_ms'].items() if key in ('batch','serial')})


if __name__=='__main__': main()
