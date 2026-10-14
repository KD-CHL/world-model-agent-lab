#!/usr/bin/env python3
"""Run directly with the wmal Python environment; no installation required."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from wmal.training.embodied_cli import main

if __name__ == '__main__':
    raise SystemExit(main())
