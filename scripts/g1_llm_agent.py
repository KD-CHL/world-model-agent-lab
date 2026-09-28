"""Natural-language G1 navigation via an authenticated chat-completions API."""
import argparse
import json
from pathlib import Path

from wmal.agents.api_client import ApiModelClient
from wmal.agents.g1_coordinator import LanguageMissionPlanner,G1Coordinator
from wmal.envs.indoor_scene import IndoorScene
from wmal.envs.g1_session import G1MuJoCoSession
from wmal.locomotion.world_model import load_world_model
from wmal.locomotion.planner import G1RolloutPlanner
from wmal.locomotion.navigation import NavigationPlanner
from wmal.locomotion.feedback import ResidualFeedback
from wmal.logging.events import EventLog


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',default='configs/g1_indoor.json')
    parser.add_argument('--instruction',help='One task; omit for persistent interactive session')
    parser.add_argument('--headless',action='store_true')
    parser.add_argument('--log',default='runs/g1_llm/events.jsonl')
    args=parser.parse_args(argv)
    try:
        client=ApiModelClient.from_env()
        config=json.loads(Path(args.config).read_text())
        model=load_world_model(config['world_model']['factory'],config['world_model'].get('config',{}))
        if config['world_model'].get('expected_version',model.version)!=model.version:
            raise ValueError('Unexpected prediction model version')
        scene=IndoorScene()
        log=EventLog(args.log)
        planner=NavigationPlanner(G1RolloutPlanner(model,**config['planner']),scene)
        coordinator=G1Coordinator(LanguageMissionPlanner(client),planner,planner.scene,model.version,
                                  log=log,feedback=ResidualFeedback(**config.get('feedback',{})),
                                  max_cycles=config.get('agent',{}).get('max_cycles_per_goal',100))
        with G1MuJoCoSession(viewer=not args.headless,realtime=not args.headless,scene=scene) as session:
            while session.is_running:
                try:
                    instruction=args.instruction or input('任务（quit 退出）> ')
                except (EOFError,KeyboardInterrupt):
                    break
                if instruction.strip().lower()=='quit':
                    break
                result=coordinator.run(instruction,session)
                print(json.dumps(result,ensure_ascii=False))
                if args.instruction:
                    return 0 if result['status']=='succeeded' else 1
                if coordinator.faulted:
                    break
        return 0
    except (ValueError,RuntimeError,OSError,KeyError,TypeError,ImportError) as exc:
        parser.exit(2,f'Agent startup failed: {type(exc).__name__}; check API environment and model configuration.\n')


if __name__=='__main__':
    raise SystemExit(main())
