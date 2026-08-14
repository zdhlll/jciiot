"""Place-down skill — release a held object at target via backend."""

from __future__ import annotations

import logging

import numpy as np

from robot_agent.core.scene_context import SceneContext
from robot_agent.core.types import ExecutionContext, SkillResult
from robot_agent.skills.base import BaseSkill
from robot_agent.skills.pick_up import _resolve_station_name

logger = logging.getLogger(__name__)


class PlaceDownSkill(BaseSkill):
    """Release a held object at the target through the environment backend.

    Resolves natural-language target descriptions to known station names
    via ``SceneContext`` (same algorithm as ``PickUpSkill``).
    """

    def __init__(self, *, backend, scene_context: SceneContext | None = None) -> None:
        super().__init__(
            name="place_down",
            description="Place down or drop an object",
            keywords=(
                "place", "put", "drop", "release",
                "place", "drop", "put", "release", "unload",
            ),
        )
        self._backend = backend
        self._scene = scene_context

    def run(self, context: ExecutionContext) -> SkillResult:
        inputs = context.metadata.get("inputs", {})
        raw_target: str = (
            inputs.get("target")
            or context.task
        )
        target = raw_target
        if self._scene is not None:
            target = _resolve_station_name(raw_target, self._scene)
            logger.info("place_down target: %r → %r", raw_target, target)

        # Physics place (only mode — no teleport fallback)
        # The live env exposes only output_1..output_4, while the semantic
        # map and the L3-L5 tasks also use output_5 / output_6. Use the
        # repository's generic Siemens support-surface placement helper for
        # those extended stations.
        live_outputs = {}
        if getattr(self._backend, "env", None) is not None:
            live_outputs = getattr(self._backend.env, "output_ports", {}) or {}
        held_name = getattr(self._backend, "_held_crate_name", None)
        scene_station = self._scene.output_ports.get(target) if self._scene is not None else None
        extended_output = bool(held_name and scene_station is not None and target not in live_outputs)
        placement_xy = inputs.get("place_xy") if extended_output else None
        if extended_output:
            if placement_xy is None:
                placement_xy = scene_station.center[:2]
            placement_xy = [float(placement_xy[0]), float(placement_xy[1])]
            try:
                ok = self._place_extended_output(target, placement_xy)
                return SkillResult(
                    skill_name=self.name,
                    success=ok,
                    message=f"Extended physics place {'OK' if ok else 'FAIL'}: {target}",
                    payload={
                        "action": "place_down", "target": target,
                        "method": "constraint_release_then_gravity",
                        "placement_xy": placement_xy, "ok": ok,
                    },
                )
            except Exception as exc:
                logger.exception("extended physics place crashed")
                return SkillResult(
                    skill_name=self.name, success=False,
                    message=f"Extended physics place error: {exc}",
                    payload={"action": "place_down", "target": target, "error": str(exc)},
                )

        if hasattr(self._backend, "place_object_physics"):
            try:
                ok = self._backend.place_object_physics(target)
                msg = f"Physics place {'OK' if ok else 'FAIL'}: {target}"
                if not ok:
                    _held = getattr(self._backend, "_held_crate_name", None)
                    _ports = list(self._backend.env.output_ports.keys()) if hasattr(self._backend, 'env') and self._backend.env else []
                    logger.warning("place_down: failed target=%s held=%s avail_out=%s", target, _held, _ports)
                    msg += f" held={_held} out_ports={_ports}"
                return SkillResult(
                    skill_name=self.name,
                    success=ok,
                    message=msg,
                    payload={
                        "action": "place_down",
                        "target": target,
                        "method": "extended_physics" if extended_output else "physics",
                        "placement_xy": placement_xy,
                        "ok": ok,
                    },
                )
            except Exception as exc:
                logger.exception("physics place crashed")
                return SkillResult(
                    skill_name=self.name, success=False,
                    message=f"Physics place error: {exc}",
                    payload={"action": "place_down", "target": target, "error": str(exc)},
                )

        # No physics configured — teleport only
        try:
            self._backend.place_object(target)
        except Exception:
            pass
        return SkillResult(
            skill_name=self.name, success=True,
            message=f"Placed (snap): {target}",
            payload={"action": "place_down", "target": target, "raw_target": raw_target, "method": "teleport"},
        )

    def _place_extended_output(self, target: str, placement_xy: list[float]) -> bool:
        """Continuously turn, open the grippers, and let gravity place the tote."""
        from robot_agent.environments.robosuite_backend import (
            _capture_upper_body_posture,
            _restore_upper_body_posture,
        )
        from robosuite.environments.factory_sorting.place_on_table import gripper_release_action
        from robosuite.environments.factory_sorting.transport_attachment import (
            clear_transport_attachment,
            sync_transport_attachment,
        )
        from robosuite.environments.factory_sorting.turn_to_station import turn_to_face_xy

        raw = self._backend.env
        held_name = getattr(self._backend, "_held_crate_name", None)
        if raw is None or not held_name:
            return False

        posture = _capture_upper_body_posture(raw, raw.robots[0])

        def _record_turn_frame() -> None:
            _restore_upper_body_posture(raw, posture)
            self._backend._record_trajectory_frame()

        turn_params = self._backend._rp["turn"]
        result = turn_to_face_xy(
            env=raw,
            target_xy=np.asarray(placement_xy, dtype=float),
            tolerance=turn_params["tolerance"],
            max_iters=turn_params["max_iters"],
            turn_steps=turn_params["turn_steps"],
            settle_steps=turn_params["settle_steps"],
            render=not self._backend._headless,
            render_sleep=0.0,
            sync_attachment=True,
            post_step_callback=_record_turn_frame,
        )
        if not result.get("success", False):
            return False

        sync_transport_attachment(raw)
        release_action = gripper_release_action(raw)
        release_steps = int(self._backend._rp["place"]["release_steps"])
        gripper_qpos_idx: list[int] = []
        gripper_qvel_idx: list[int] = []
        for joint_list in raw.robots[0].gripper_joints.values():
            for joint_name in joint_list:
                gripper_qpos_idx.append(raw.sim.model.get_joint_qpos_addr(joint_name))
                gripper_qvel_idx.append(raw.sim.model.get_joint_qvel_addr(joint_name))

        def _hold_upper_body_keep_grippers() -> None:
            qpos = np.array(raw.sim.data.qpos[gripper_qpos_idx], dtype=float)
            qvel = np.array(raw.sim.data.qvel[gripper_qvel_idx], dtype=float)
            _restore_upper_body_posture(raw, posture)
            raw.sim.data.qpos[gripper_qpos_idx] = qpos
            raw.sim.data.qvel[gripper_qvel_idx] = qvel
            raw.sim.forward()

        clear_transport_attachment(raw)
        self._backend._held_crate_name = None
        self._backend._held_crate_body_id = None
        for _ in range(release_steps):
            raw.step(release_action)
            _hold_upper_body_keep_grippers()
            self._backend._record_trajectory_frame()
            if not self._backend._headless:
                raw.render()
        logger.info("Released %s at extended output %s", held_name, target)
        return True
