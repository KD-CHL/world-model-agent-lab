"""Small, dependency-free trajectory interchange layer.

The schema is deliberately close to common robotics datasets: one episode is
an ordered sequence of observations, actions and optional images.  Adapters
for LeRobot, robomimic and simulator output can convert into this schema
without making the planner depend on any of those projects.
"""
from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Any, Iterable


@dataclass(frozen=True)
class Transition:
    episode_id: str
    step_id: int
    observation: dict[str, float]
    action: dict[str, float]
    next_observation: dict[str, float]
    instruction: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def validate(self) -> None:
        if not self.episode_id or self.step_id < 0:
            raise ValueError("Invalid transition identity")
        if not self.observation or set(self.observation) != set(self.next_observation):
            raise ValueError("Observation schemas must match")
        if not set(self.action).issubset(self.observation):
            raise ValueError("Action contains unknown joints")
        values = list(self.observation.values()) + list(self.action.values()) + list(self.next_observation.values())
        if not all(isinstance(value, (int, float)) for value in values):
            raise ValueError("Trajectory values must be numeric")

    def to_dict(self) -> dict[str, Any]:
        self.validate()
        return {"episode_id": self.episode_id, "step_id": self.step_id,
                "observation": self.observation, "action": self.action,
                "next_observation": self.next_observation,
                "instruction": self.instruction, "metadata": self.metadata}


def write_jsonl(path: str | Path, transitions: Iterable[Transition]) -> int:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with destination.open("w", encoding="utf-8") as handle:
        for transition in transitions:
            handle.write(json.dumps(transition.to_dict(), sort_keys=True) + "\n")
            count += 1
    return count


def read_jsonl(path: str | Path) -> list[Transition]:
    rows = []
    with Path(path).open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            try:
                payload = json.loads(line)
                row = Transition(**payload)
                row.validate()
                rows.append(row)
            except (TypeError, ValueError, json.JSONDecodeError) as exc:
                raise ValueError(f"Invalid trajectory row {line_number}") from exc
    return rows
