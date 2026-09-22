"""Run after installing this package or setting PYTHONPATH=src."""
from wmal.agents.cli import run_agent

if __name__ == '__main__':
    raise SystemExit(run_agent())
