"""Named, explicit skill capabilities shared by robot-specific runtimes."""
from dataclasses import dataclass
from typing import Callable


@dataclass(frozen=True)
class SkillSpec:
    name: str
    parameter_schema: str
    state_schema: str
    precondition: Callable
    execute: Callable
    success: Callable
    max_cycles: int = 100

    def __post_init__(self):
        if (not self.name or not self.parameter_schema or not self.state_schema
                or type(self.max_cycles) is not int or self.max_cycles < 1
                or not all(callable(f) for f in (self.precondition, self.execute, self.success))):
            raise ValueError('Invalid skill specification')


class SkillRegistry:
    def __init__(self):
        self._skills = {}

    def register(self, spec):
        if not isinstance(spec, SkillSpec) or spec.name in self._skills:
            raise ValueError('Invalid or duplicate skill')
        self._skills[spec.name] = spec

    def get(self, name):
        if name not in self._skills:
            raise ValueError('Unregistered skill: ' + str(name))
        return self._skills[name]

    def capabilities(self):
        return [{'name': s.name, 'parameter_schema': s.parameter_schema,
                 'state_schema': s.state_schema, 'max_cycles': s.max_cycles}
                for s in self._skills.values()]
