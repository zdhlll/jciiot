"""Record-trajectory skill — saves accumulated trajectory data as JSON."""

from __future__ import annotations

import logging
from pathlib import Path

from robot_agent.core.types import ExecutionContext, SkillResult
from robot_agent.skills.base import BaseSkill

logger = logging.getLogger(__name__)

DEFAULT_OUTPUT = Path(__file__).resolve().parents[4] / "recordings" / "trajectory.json"


class RecordTrajectorySkill(BaseSkill):
    """Save the backend's accumulated trajectory frames to a JSON file.

    This skill should run as the **last** step in a pipeline, after all
    navigation / manipulation actions have completed.  It reads the
    trajectory data buffered by ``RobosuiteBackend`` and writes a JSON
    file compatible with external replay tools.
    """

    def __init__(self, *, backend, output_path: str | Path | None = None) -> None:
        super().__init__(
            name="record_trajectory",
            description="Record and save task trajectory to JSON file",
            keywords=("record", "save", "trajectory", "log", "capture"),
        )
        self._backend = backend
        self._output = Path(output_path) if output_path else DEFAULT_OUTPUT

    def run(self, context: ExecutionContext) -> SkillResult:
        try:
            rec_dir = self._output.parent
            rec_dir.mkdir(exist_ok=True)
            path = self._backend.save_trajectory(self._output)
            return SkillResult(
                skill_name=self.name,
                success=True,
                message=f"Trajectory saved: {path} ({len(self._backend.get_trajectory())} frames)",
                payload={"action": "record_trajectory", "path": str(path)},
            )
        except Exception as exc:
            logger.error("Failed to save trajectory: %s", exc)
            return SkillResult(
                skill_name=self.name,
                success=False,
                message=f"Trajectory save failed: {exc}",
                payload={"action": "record_trajectory", "error": str(exc)},
            )
