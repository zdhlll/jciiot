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
        raw_target: str = inputs.get("target") or context.task
        target = raw_target
        if self._scene is not None:
            target = _resolve_station_name(raw_target, self._scene)
            logger.info("place_down target: %r → %r", raw_target, target)

        # 语义地图里有 output_5/6(L3-L5 的放置目标),但场景的输出端口
        # 表只登记了 output_1..4。地图里有、端口表里没有的目标走扩展
        # 放置:在给定坐标转向→同步附着→松爪,靠重力自然落到桌面。
        live_outputs = getattr(getattr(self._backend, "env", None), "output_ports", {}) or {}
        held_name = getattr(self._backend, "_held_crate_name", None)
        scene_station = self._scene.output_ports.get(target) if self._scene is not None else None
        needs_extended = bool(held_name and scene_station is not None and target not in live_outputs)
        placement_xy = inputs.get("place_xy") if needs_extended else None
        if needs_extended:
            if placement_xy is None:
                placement_xy = scene_station.center[:2]
            placement_xy = [float(placement_xy[0]), float(placement_xy[1])]
            try:
                ok = self._place_at_extended_xy(target, placement_xy)
                return SkillResult(
                    skill_name=self.name,
                    success=ok,
                    message=f"Extended physics place {'OK' if ok else 'FAIL'}: {target}",
                    payload={
                        "action": "place_down", "target": target,
                        "method": "gravity_release_after_turn",
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
                    _ports = (
                        list(self._backend.env.output_ports.keys())
                        if hasattr(self._backend, "env") and self._backend.env else []
                    )
                    logger.warning("place_down: failed target=%s held=%s avail_out=%s", target, _held, _ports)
                    msg += f" held={_held} out_ports={_ports}"
                return SkillResult(
                    skill_name=self.name,
                    success=ok,
                    message=msg,
                    payload={
                        "action": "place_down",
                        "target": target,
                        "method": "extended_physics" if needs_extended else "physics",
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

        # 没有物理接口时不假装成功:缺少物理放置能力是显式失败,
        # 不做瞬移兜底(那会绕过比赛的真实执行与评分语义)。
        return SkillResult(
            skill_name=self.name,
            success=False,
            message=f"Physics place backend is unavailable: {target}",
            payload={
                "action": "place_down",
                "target": target,
                "raw_target": raw_target,
                "method": "unavailable",
            },
        )

    def _place_at_extended_xy(self, target: str, xy: list[float]) -> bool:
        """转向目标桌位,同步附着后松爪,让箱子自然落到桌面。

        携物转向期间锁住上肢姿态,防止箱子/手臂扫落邻物;松爪阶段
        只允许夹爪自由度变化,上肢继续锁在快照上;结束后清空运输
        附着并把 held 状态归零。
        """
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

        # 放置转向:旋转已由 move 层沿网格安全腿完成(goal_yaw
        # 注入),这里原地转向只剩零角度残量;未注入时兜底照旧。
        posture = _capture_upper_body_posture(raw, raw.robots[0])

        def _turn_step() -> None:
            _restore_upper_body_posture(raw, posture)
            self._backend._record_trajectory_frame()

        turn_cfg = self._backend._rp["turn"]
        turned = turn_to_face_xy(
            env=raw,
            target_xy=np.asarray(xy, dtype=float),
            tolerance=turn_cfg["tolerance"],
            max_iters=turn_cfg["max_iters"],
            turn_steps=turn_cfg["turn_steps"],
            settle_steps=turn_cfg["settle_steps"],
            render=not self._backend._headless,
            render_sleep=0.0,
            sync_attachment=True,
            post_step_callback=_turn_step,
        )
        _restore_upper_body_posture(raw, posture)
        if not turned.get("success", False):
            return False

        sync_transport_attachment(raw)

        # 夹爪自由度地址:松爪循环里夹爪可以动,其余上肢锁回快照
        finger_addrs: list[int] = []
        finger_vel_addrs: list[int] = []
        for joints in raw.robots[0].gripper_joints.values():
            for name in joints:
                finger_addrs.append(raw.sim.model.get_joint_qpos_addr(name))
                finger_vel_addrs.append(raw.sim.model.get_joint_qvel_addr(name))

        def _keep_body_locked() -> None:
            finger_q = np.array(raw.sim.data.qpos[finger_addrs], dtype=float)
            finger_v = np.array(raw.sim.data.qvel[finger_vel_addrs], dtype=float)
            _restore_upper_body_posture(raw, posture)
            raw.sim.data.qpos[finger_addrs] = finger_q
            raw.sim.data.qvel[finger_vel_addrs] = finger_v
            raw.sim.forward()

        clear_transport_attachment(raw)
        self._backend._held_crate_name = None
        self._backend._held_crate_body_id = None
        release = gripper_release_action(raw)
        for _ in range(int(self._backend._rp["place"]["release_steps"])):
            raw.step(release)
            _keep_body_locked()
            self._backend._record_trajectory_frame()
            if not self._backend._headless:
                raw.render()
        logger.info("Released %s at extended output %s", held_name, target)
        return True
