"""Check local syntax/contracts; does not claim ROS 2 end-to-end execution."""
import ast
import json
from pathlib import Path
import xml.etree.ElementTree as ET

root = Path(__file__).resolve().parents[1]
required = ["README.md", "pyproject.toml", "docs/01_literature.md", "docs/02_research_and_route.md", "docs/03_architecture.md", "docs/04_contracts_and_runtime.md", "docs/05_training_pseudocode.md", "docs/06_experiments_and_data.md", "docs/07_delivery_plan.md", "docs/09_agent_ros2_design.md", "ros2/wmal_interfaces/package.xml", "ros2/wmal_interfaces/CMakeLists.txt"]
for name in required:
    assert (root / name).is_file(), name
for folder in ("src", "scripts", "tests"):
    for path in (root / folder).rglob("*.py"):
        ast.parse(path.read_text(), filename=str(path))
for folder in ("configs", "examples"):
    for path in (root / folder).rglob("*.json"):
        json.loads(path.read_text())
ET.parse(root / "ros2/wmal_interfaces/package.xml")
print("PASS: files, Python syntax, JSON and interface package XML. ROS 2 end-to-end not checked by this script.")
