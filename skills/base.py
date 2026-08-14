"""Base classes for skills."""

from __future__ import annotations

from dataclasses import dataclass

from robot_agent.core.types import ExecutionContext, SkillResult


@dataclass(slots=True)
class BaseSkill:
    name: str
    description: str
    keywords: tuple[str, ...]

    def can_handle(self, task: str) -> bool:
        lowered = task.lower()
        return any(keyword in lowered for keyword in self.keywords)

    def run(self, context: ExecutionContext) -> SkillResult:
        raise NotImplementedError
