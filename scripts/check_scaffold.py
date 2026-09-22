"""Check design artifacts, syntax and JSON; does not test robotics."""
import ast
import json
from pathlib import Path

root = Path(__file__).resolve().parents[1]
required = ["README.md", "pyproject.toml", "docs/01_literature.md", "docs/02_research_and_route.md", "docs/03_architecture.md", "docs/04_contracts_and_runtime.md", "docs/05_training_pseudocode.md", "docs/06_experiments_and_data.md", "docs/07_delivery_plan.md"]
for name in required:
    assert (root / name).is_file(), name
for path in root.rglob("*.py"):
    ast.parse(path.read_text(), filename=str(path))
for path in root.rglob("*.json"):
    json.loads(path.read_text())
print("PASS: design files, Python syntax and JSON. Training/simulation NOT implemented or tested.")
