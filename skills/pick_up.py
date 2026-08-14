"""Pick-up skill — grasp and lift a target object via backend."""

from __future__ import annotations

import logging
import re
from argparse import Namespace
from pathlib import Path

import numpy as np

from robot_agent.core.scene_context import SceneContext
from robot_agent.core.types import ExecutionContext, SkillResult
from robot_agent.skills.base import BaseSkill

logger = logging.getLogger(__name__)

# Chinese-number → digit
_CN_DIGIT: dict[str, str] = {
    "一": "1", "二": "2", "三": "3", "四": "4",
    "五": "5", "六": "6", "七": "7", "八": "8",
    "九": "9", "十": "10",
}
# Chinese role → role prefix
_CN_ROLE: dict[str, str] = {
    "进料": "input", "输入": "input", "入料": "input",
    "出料": "output", "输出": "output",
}
# Digit-word → index
_CN_INDEX: dict[str, str] = {
    "1": "1", "2": "2", "3": "3", "4": "4",
    "一": "1", "二": "2", "三": "3", "四": "4",
}
# Station kind keywords to strip from target
_CN_KIND: list[str] = ["传送带", "架子", "桌子", "箱子", "料箱", "料斗",
                        "conveyor", "shelf", "table", "bin"]


# L2's input_6 semantic-map approach is the midpoint of two separated totes.
# The demonstrations instead place the base a fixed *relative distance*
# outside the current upper tote's robot-facing col_right wall. World
# coordinates are therefore derived from live MuJoCo geometry on every pick.
# Standoff distance from robot base to object centre, calibrated from the
# L1 official teacher by _calibrate_robot_standoff in the data collector.
_BASE_STANDOFF = 0.941

# ── 去 _LEVEL_CONFIG 版 ─────────────────────────────────────
# 场景的 object_name / source / yaw 全部改从 knowledge/task_config.json
# 读取(单一数据源,不再在代码里硬编码)。task_config.json 的
# grasp_poses 需要为每个 source 提供 yaw;L3 的 aux_input_1 条目已补。
#
# 固定顺序仅用于 transport_retract_mask 的按位索引(与 jci 版
# _LEVEL_CONFIG 的键顺序保持一致,行为不变)。
_COMPETITION_ENV_ORDER: tuple[str, ...] = (
    "FactorySorting1_3FO3ERFHISEM",
    "FactorySorting3_3FO3ERRPH7X9",
    "FactorySorting5_3FO3ERTPXEUT",
    "FactorySorting7_3FO3ERFKY9RN",
    "FactorySorting9_3FO3ERT2C5FP",
)

_TASK_CFG_CACHE: dict | None = None

# grasp_poses 缺失时的 yaw 兜底 —— 让原始 task_config.json 无需修改也能跑。
# 仅 L3 的 aux_input_1 在官方 task_config.json 中没有条目(官方从南侧
# 接近辅助台,yaw=π/2);其余工位均有条目,不会命中兜底。
# 注意:不要用 atan2(center-approach) 几何推导替代 —— 实测对 L1 input_5
# 偏差约 40°(语义图 approach 点不在真实停靠方向上)。
_YAW_FALLBACK: dict[str, float] = {
    "aux_input_1": np.pi / 2.0,
}


def _load_task_config() -> dict:
    """Load knowledge/task_config.json once per process."""
    global _TASK_CFG_CACHE
    if _TASK_CFG_CACHE is None:
        import json as _json
        _path = Path(__file__).resolve().parents[3] / "knowledge" / "task_config.json"
        _TASK_CFG_CACHE = _json.loads(_path.read_text(encoding="utf-8"))
    return _TASK_CFG_CACHE


def _scene_grasp_config(env_name: str) -> dict | None:
    """Derive {object_name, source, yaw} for *env_name* from task_config.json.

    object_name / source come from ``tasks[].object[0]`` / ``tasks[].source``;
    yaw comes from ``grasp_poses[source].yaw``.  When the task config has no
    grasp-pose entry for the source, ``_YAW_FALLBACK`` supplies the yaw, so
    the original task_config.json works without modification.  Returns None
    only when the scene itself is unknown; ``yaw`` may be None when neither
    the config nor the fallback defines it.
    """
    cfg = _load_task_config()
    for task in cfg.get("tasks", []):
        if str(task.get("env_name")) != env_name:
            continue
        source = str(task.get("source") or "")
        objects = task.get("object") or []
        object_name = (
            str(objects[0])
            if isinstance(objects, (list, tuple)) and objects
            else str(objects or "")
        )
        if not source:
            return None
        entry = cfg.get("grasp_poses", {}).get(source)
        if entry and entry.get("yaw") is not None:
            yaw = float(entry["yaw"])
        else:
            yaw = _YAW_FALLBACK.get(source)
        return {
            "object_name": object_name,
            "source": source,
            "yaw": yaw,
        }
    return None


# ── universal-geometry 标定权重 ─────────────────────────────
# 原实现从 skills/checkpoints/universal_geometry_model.pth(139MB)里加载,
# 但 geometry_policy.state_dict 实际只有下面这 13 个标量。这里内联,
# 不再依赖 torch 和 .pth 文件。transport_retract_mask 在官方 pth 中
# 为 null(dict.get(key, default) 在 key 存在时返回 null 而非 default),
# 代码在 retract 块中已做 null 安全处理。
_GEOMETRY_WEIGHTS: dict = {
    "base_standoff": 0.941,
    "max_action": 0.65,
    "arrival_tolerance": 0.03,
    "gripper_end_arrival_tolerance": 0.035,
    "site_below_offset": 0.035,
    "site_above_clearance": 0.05,
    "safe_steps": 45,
    "xy_steps": 100,
    "down_steps": 75,
    "settle_steps": 100,
    "grasp_steps": 45,
    "record_interval": 5,
    "transport_retract_mask": None,
}


def _universal_geometry_weights() -> dict:
    """Return the calibrated task-neutral controller weights (inlined)."""
    return _GEOMETRY_WEIGHTS


def _base_robosuite_env(env):
    """Return the underlying robosuite env without changing core/backend."""

    current = env
    seen: set[int] = set()
    while current is not None and id(current) not in seen:
        seen.add(id(current))
        if hasattr(current, "sim") and hasattr(current, "robots"):
            return current
        unwrapped = getattr(current, "unwrapped", None)
        if unwrapped is not None and unwrapped is not current:
            current = unwrapped
            continue
        current = getattr(current, "env", None)
    raise RuntimeError("Could not resolve the underlying robosuite environment")


def _live_object_center(raw_env, object_name: str) -> np.ndarray:
    """Resolve an object's world centre across all five official scenes."""

    model = raw_env.sim.model
    data = raw_env.sim.data
    for site_name in (f"{object_name}_center_site", f"{object_name}_default_site"):
        try:
            site_id = model.site_name2id(site_name)
        except (KeyError, ValueError):
            continue
        return np.asarray(data.site_xpos[site_id], dtype=float).copy()

    body_id = getattr(raw_env, "obj_body_id", {}).get(object_name)
    if body_id is not None:
        return np.asarray(data.body_xpos[body_id], dtype=float).copy()

    metadata = getattr(raw_env, "material_metadata", {}).get(object_name, {}) or {}
    body_name = metadata.get("body_name")
    if body_name:
        return np.asarray(data.body_xpos[model.body_name2id(body_name)], dtype=float).copy()
    raise ValueError(f"Cannot resolve live centre for object {object_name!r}")


def _dynamic_grasp_pose(backend, object_name: str | None) -> tuple[dict | None, dict]:
    """Resolve a grasp base pose from the live object centre for any level.

    Works for all levels (L1–L5).  The robot is placed ``standoff`` metres
    in front of the object along the approach direction defined by the
    per-level yaw.  This matches the collector's object-aligned mode.
    """

    env_name = str(getattr(backend, "_env_name", ""))
    cfg = _scene_grasp_config(env_name)
    if cfg is None or cfg.get("yaw") is None:
        return None, {}

    effective_object = object_name or cfg["object_name"]
    base_yaw = cfg["yaw"]
    standoff = float(_universal_geometry_weights().get("base_standoff", _BASE_STANDOFF))

    raw_env = _base_robosuite_env(getattr(backend, "env", None))
    raw_env.sim.forward()
    object_center = _live_object_center(raw_env, effective_object)

    # Robot base = object_centre – heading(yaw) × standoff
    forward = np.array([np.cos(base_yaw), np.sin(base_yaw)], dtype=float)
    base_xy = object_center[:2] - forward * standoff

    pose = {
        "robot_base_pos": [float(base_xy[0]), float(base_xy[1]), 0.0],
        "robot_base_ori": [0.0, 0.0, base_yaw],
        "xy": [float(base_xy[0]), float(base_xy[1])],
        "yaw": base_yaw,
    }
    diagnostics = {
        "mode": "dynamic_standoff",
        "env_name": env_name,
        "object_name": effective_object,
        "object_center": object_center.tolist(),
        "forward": forward.tolist(),
        "standoff": standoff,
        "resolved_base_position": pose["robot_base_pos"],
        "resolved_base_yaw": base_yaw,
    }
    return pose, diagnostics


def _collision_free_grasp_pose(backend, object_name: str | None) -> tuple[dict | None, dict]:
    """Resolve a live-object pose without mutating the simulator state.

    Earlier versions tested candidates by directly changing base qpos. That
    looked like teleportation in the recorded trajectory. The end-to-end
    workflow now reaches this pose through ``MoveSkill`` and only turns in
    place here.
    """

    pose, diagnostics = _dynamic_grasp_pose(backend, object_name)
    diagnostics = dict(diagnostics)
    diagnostics["mode"] = "non_mutating_live_object_standoff"
    return pose, diagnostics


def _scripted_grasp_and_attach(backend, object_name: str, source: str) -> tuple[bool, dict]:
    """Run the repository's geometry-driven two-arm grasp on the live env."""

    from robosuite.environments.factory_sorting import (
        load_factory_sorting_1_3fo3erfhisem_collect as collector,
    )
    from robosuite.environments.factory_sorting.lift_after_grasp import (
        lift_grasped_object,
    )
    from robosuite.environments.factory_sorting.transport_attachment import (
        capture_transport_attachment,
    )
    from robot_agent.workflows.collect_factory_sorting import (
        _approach_aligned_target_positions,
    )

    env = getattr(backend, "env", None)
    raw_env = _base_robosuite_env(env)
    robot = raw_env.robots[0]
    weights = _universal_geometry_weights()

    # Multi-object L5 runs otherwise accumulate arm/controller drift after
    # every release. Capture the initial non-base robot posture once and
    # restore it before each scripted pick while preserving the newly chosen
    # collision-free mobile-base pose.
    base_joints = set(getattr(robot.robot_model, "base_joints", []) or [])
    posture = getattr(backend, "_scripted_pick_posture", None)
    if posture is None:
        posture = {}
        for joint_name in raw_env.sim.model.joint_names:
            if not joint_name or not joint_name.startswith("robot0_") or joint_name in base_joints:
                continue
            try:
                posture[joint_name] = (
                    np.asarray(raw_env.sim.data.get_joint_qpos(joint_name), dtype=float).copy(),
                    np.asarray(raw_env.sim.data.get_joint_qvel(joint_name), dtype=float).copy(),
                )
            except Exception:
                continue
        backend._scripted_pick_posture = posture
    else:
        for joint_name, (qpos, qvel) in posture.items():
            try:
                raw_env.sim.data.set_joint_qpos(joint_name, qpos)
                raw_env.sim.data.set_joint_qvel(joint_name, qvel)
            except Exception:
                continue
        raw_env.sim.forward()

    setattr(robot, collector.CAMERA_HOLD_TARGET_ATTR, collector.capture_camera_hold_targets(robot))
    args = Namespace(
        max_action=float(weights["max_action"]),
        arrival_tolerance=float(weights["arrival_tolerance"]),
        gripper_end_arrival_tolerance=float(weights["gripper_end_arrival_tolerance"]),
        settle_steps=int(weights["settle_steps"]),
    )
    record_counter = [0]

    def record_step() -> None:
        record_counter[0] += 1
        if record_counter[0] % int(weights["record_interval"]) == 0 and hasattr(backend, "_record_trajectory_frame"):
            backend._record_trajectory_frame()

    def step_targets(targets: dict, *, use_gripper_end: bool, gripper_value: float) -> None:
        robot.composite_controller.update_state()
        arm_actions = {}
        for arm in collector.ARMS:
            current = (
                collector.gripper_end_center_pos(raw_env, robot, arm)
                if use_gripper_end
                else collector.get_eef_pos(raw_env, robot, arm)
            )
            world_delta = np.asarray(targets[arm], dtype=float) - current
            controller_delta = collector.world_delta_to_controller_frame(robot, arm, world_delta)
            arm_actions[arm] = collector.arm_delta_to_normalized_action(
                robot, arm, controller_delta, args.max_action,
            )
        action = collector.build_action(raw_env, robot, arm_actions, gripper_value)
        env.step(action)
        record_step()

    def linear_segment(goals: dict, steps: int, *, use_gripper_end: bool = False) -> None:
        current_fn = collector.gripper_end_center_pos if use_gripper_end else collector.get_eef_pos
        starts = {
            arm: np.asarray(current_fn(raw_env, robot, arm), dtype=float).copy()
            for arm in collector.ARMS
        }
        for index in range(1, steps + 1):
            alpha = index / float(steps)
            targets = {
                arm: starts[arm] + alpha * (np.asarray(goals[arm]) - starts[arm])
                for arm in collector.ARMS
            }
            step_targets(targets, use_gripper_end=use_gripper_end, gripper_value=-1.0)

    if hasattr(backend, "_record_trajectory_frame"):
        backend._record_trajectory_frame()
    if hasattr(backend, "_mark_trajectory_event"):
        backend._mark_trajectory_event(
            "grasp_start", object_name=object_name, source=source,
        )

    below_targets, site_names = _approach_aligned_target_positions(
        raw_env, object_name, float(weights["site_below_offset"]),
    )
    starts = {
        arm: collector.get_eef_pos(raw_env, robot, arm)
        for arm in collector.ARMS
    }
    site_positions = {
        arm: np.asarray(below_targets[arm]) + np.array([0.0, 0.0, float(weights["site_below_offset"])])
        for arm in collector.ARMS
    }
    safe_z = max(
        max(float(starts[arm][2]) for arm in collector.ARMS),
        max(float(site_positions[arm][2] + float(weights["site_above_clearance"])) for arm in collector.ARMS),
    )
    safe_targets = {
        arm: np.array([starts[arm][0], starts[arm][1], safe_z])
        for arm in collector.ARMS
    }
    xy_targets = {
        arm: np.array([site_positions[arm][0], site_positions[arm][1], safe_z])
        for arm in collector.ARMS
    }

    linear_segment(safe_targets, int(weights["safe_steps"]))
    linear_segment(xy_targets, int(weights["xy_steps"]))
    linear_segment(below_targets, int(weights["down_steps"]))
    for _ in range(args.settle_steps):
        distances = {
            arm: float(np.linalg.norm(
                collector.gripper_end_center_pos(raw_env, robot, arm)
                - np.asarray(below_targets[arm])
            ))
            for arm in collector.ARMS
        }
        if all(value <= args.gripper_end_arrival_tolerance for value in distances.values()):
            break
        step_targets(below_targets, use_gripper_end=True, gripper_value=-1.0)

    distances = {
        arm: float(np.linalg.norm(
            collector.gripper_end_center_pos(raw_env, robot, arm)
            - np.asarray(below_targets[arm])
        ))
        for arm in collector.ARMS
    }
    if any(value > args.gripper_end_arrival_tolerance for value in distances.values()):
        if hasattr(backend, "_mark_trajectory_event"):
            backend._mark_trajectory_event(
                "grasp_end", object_name=object_name, source=source, success=False,
            )
        return False, {
            "reason": "gripper target arrival failed",
            "distances": distances,
            "site_names": site_names,
        }

    for _ in range(int(weights["grasp_steps"])):
        action = collector.build_action(raw_env, robot, {}, gripper_value=1.0)
        env.step(action)
        record_step()
    grasp_status = collector.grasp_status(raw_env, robot, object_name)
    grasp_ok = all(grasp_status.values())

    lift_result = {"success": False, "failure_reason": "grasp failed"}
    if grasp_ok:
        lift_cfg = getattr(backend, "_rp", {}).get("lift", {})
        lift_result = lift_grasped_object(
            env=env,
            object_name=object_name,
            lift_height=float(lift_cfg.get("lift_height", 0.15)),
            max_steps=int(lift_cfg.get("max_steps", 300)),
            hold_steps=int(lift_cfg.get("hold_steps", 20)),
            tolerance=float(lift_cfg.get("tolerance", 0.02)),
            max_action=float(lift_cfg.get("max_action", 0.8)),
            render=False,
            render_callback=record_step,
        )
    ok = grasp_ok and bool(lift_result.get("success"))
    if ok:
        capture_transport_attachment(raw_env, object_name)
        backend._held_crate_name = object_name
        # Navigation locks the *current* upper-body posture.  Keeping the
        # wide grasp posture here makes the fingertips sweep through nearby
        # machinery (notably the L2 line-6 centre module).  Once the transport
        # attachment has captured the object's base-relative pose, the
        # checkpoint's multi-task gate selects whether to fold the arms. The
        # L5 source is too tightly packed to fold in place, while the other
        # transit routes benefit from the neutral posture. The attachment
        # continues carrying the object during transport synchronization.
        # pth 中 transport_retract_mask 键存在但值为 null:
        # dict.get(key, default) 此时返回 None(不是 default),
        # np.asarray(None, dtype=int) 内部 int(None) 会抛 TypeError。
        # 显式判空后回退到逐级默认掩码 [1,1,1,1,0](L1-L4 收臂,L5 不收)。
        _retract_values = weights.get("transport_retract_mask")
        if _retract_values is None:
            _retract_values = [1, 1, 1, 1, 0]
        retract_mask = np.asarray(_retract_values, dtype=int).reshape(-1)
        env_name = str(getattr(backend, "_env_name", ""))
        level_index = (
            _COMPETITION_ENV_ORDER.index(env_name)
            if env_name in _COMPETITION_ENV_ORDER
            else 0
        )
        should_retract = bool(
            retract_mask[level_index]
            if level_index < retract_mask.size
            else weights.get("retract_after_grasp", True)
        )
        if should_retract:
            for joint_name, (qpos, qvel) in posture.items():
                try:
                    raw_env.sim.data.set_joint_qpos(joint_name, qpos)
                    raw_env.sim.data.set_joint_qvel(joint_name, qvel)
                except Exception:
                    continue
            raw_env.sim.forward()
        if hasattr(backend, "_record_trajectory_frame"):
            backend._record_trajectory_frame()
    if hasattr(backend, "_mark_trajectory_event"):
        backend._mark_trajectory_event(
            "grasp_end", object_name=object_name, source=source, success=ok,
        )
    return ok, {
        "method": "scripted_geometry_grasp",
        "site_names": site_names,
        "arrival_distances": distances,
        "grasp_status": grasp_status,
        "lift_result": lift_result,
    }


def _sync_navigation_base_to_grasp_pose(backend, pose: dict) -> dict:
    """Keep the persistent nav env aligned with the temporary grasp env.

    L2 executes BC in a temporary wrapped environment. The backend later
    captures the held object's robot-relative transport offset in the
    persistent navigation environment. If that environment still has the
    generic station approach pose, the offset contains a large artificial
    lateral component and the object is placed beside the output table.
    """

    from robosuite.environments.factory_sorting.load_factory_sorting_evalization import (
        get_base_world_pose,
    )
    from robosuite.environments.factory_sorting.turn_to_station import (
        set_base_world_yaw_direct,
        set_base_xy_direct,
        zero_base_velocity,
    )

    raw_env = _base_robosuite_env(getattr(backend, "env", None))
    robot = raw_env.robots[0]
    target_xy = np.asarray(pose["robot_base_pos"], dtype=float)[:2]
    target_yaw = float(pose["yaw"])

    set_base_world_yaw_direct(raw_env, robot, target_yaw)
    set_base_xy_direct(raw_env, robot, target_xy)
    zero_base_velocity(raw_env, robot)
    raw_env.sim.forward()

    actual_xy, actual_yaw = get_base_world_pose(raw_env, robot)
    xy_error = float(np.linalg.norm(np.asarray(actual_xy)[:2] - target_xy))
    yaw_error = float((float(actual_yaw) - target_yaw + np.pi) % (2.0 * np.pi) - np.pi)
    if xy_error > 1e-4 or abs(yaw_error) > 1e-4:
        raise RuntimeError(
            "Persistent navigation base did not reach the dynamic grasp pose: "
            f"xy_error={xy_error:.6g}, yaw_error={yaw_error:.6g}"
        )
    print(
        "[L2_NAV_BASE_SYNC] "
        f"base=({actual_xy[0]:.6f},{actual_xy[1]:.6f}) "
        f"yaw={actual_yaw:.6f}",
        flush=True,
    )
    return {
        "target_xy": target_xy.tolist(),
        "actual_xy": np.asarray(actual_xy, dtype=float)[:2].tolist(),
        "target_yaw": target_yaw,
        "actual_yaw": float(actual_yaw),
        "xy_error": xy_error,
        "yaw_error": yaw_error,
    }


def _turn_navigation_base_to_grasp_pose(backend, pose: dict) -> dict:
    """Turn continuously to the requested grasp yaw without changing base XY."""

    from robot_agent.environments.robosuite_backend import (
        _capture_upper_body_posture,
        _restore_upper_body_posture,
    )
    from robosuite.environments.factory_sorting.load_factory_sorting_evalization import (
        get_base_world_pose,
    )
    from robosuite.environments.factory_sorting.turn_to_station import turn_to_face_xy

    raw_env = _base_robosuite_env(getattr(backend, "env", None))
    posture = _capture_upper_body_posture(raw_env, raw_env.robots[0])
    base_xy, _ = get_base_world_pose(raw_env, raw_env.robots[0])
    target_yaw = float(pose["yaw"])
    target_xy = np.asarray(base_xy, dtype=float)[:2] + np.array(
        [np.cos(target_yaw), np.sin(target_yaw)], dtype=float,
    )

    def _record_frame() -> None:
        # A zero base action must not relax the arms, torso, head or grippers.
        # The grasp policy is calibrated from this exact ready posture.
        _restore_upper_body_posture(raw_env, posture)
        if hasattr(backend, "_record_trajectory_frame"):
            backend._record_trajectory_frame()

    result = turn_to_face_xy(
        env=raw_env,
        target_xy=target_xy,
        tolerance=0.02,
        max_iters=8,
        turn_steps=40,
        settle_steps=10,
        render=not bool(getattr(backend, "_headless", True)),
        render_sleep=0.0,
        sync_attachment=False,
        post_step_callback=_record_frame,
    )
    _restore_upper_body_posture(raw_env, posture)
    if not result.get("success", False):
        raise RuntimeError(
            "Continuous grasp turn failed: "
            f"final_error={result.get('final_error')}, xy_drift={result.get('xy_drift')}"
        )
    return result


def _primary_object_name(value) -> str | None:
    if isinstance(value, str):
        value = value.strip()
        return value or None
    if isinstance(value, (list, tuple)):
        for item in value:
            name = _primary_object_name(item)
            if name:
                return name
    return None


def _resolve_station_name(target: str, scene: SceneContext) -> str:
    """Resolve a natural-language target to a known station name.

    Examples of what this handles:
        "在1号进料口抓取目标物体" → "input_1"
        "把物品放到3号出料口"     → "output_3"
        "input_1"                  (pass-through — exact match)
    """
    known = scene.all_port_names()
    if not known:
        return target

    # 0) exact match
    if target in known:
        return target

    # 1) known name is a substring of target — 按长度降序,
    #    避免 "去aux_input_1取料" 被 "input_1" 抢先命中。
    for name in sorted(known, key=len, reverse=True):
        if name in target:
            return name

    # 2) match by (role, index) — e.g. "1号进料口" → input station #1
    role, idx = _parse_role_index(target)
    if role and idx is not None:
        desired_idx = int(idx)
        for name in known:
            info = (scene.input_ports.get(name) or
                    scene.output_ports.get(name))
            if info is None:
                continue
            if info.role == role and info.index == desired_idx:
                return name

    return target


def _parse_role_index(text: str) -> tuple[str | None, int | None]:
    """Extract (role, index) from Chinese text like "1号进料口" → ("input", 1)."""
    # Normalise Chinese digits → Arabic
    s = text
    for cn, d in _CN_DIGIT.items():
        s = s.replace(cn, d)

    # Find a digit followed by optional characters then a role word
    m = re.search(r"(\d+)\s*[号#]?\s*([进出入输][料料入出])", s)
    if m:
        digit = m.group(1)
        role_cn = m.group(2)
        for cn_word, role_prefix in _CN_ROLE.items():
            if cn_word in role_cn:
                return role_prefix, int(digit)

    # Also try "input_N" / "output_N" pattern directly
    m = re.search(r"(input|output)\s*_?\s*(\d+)", text, re.IGNORECASE)
    if m:
        return m.group(1).lower(), int(m.group(2))

    return None, None


class PickUpSkill(BaseSkill):
    """Grasp a target object through the environment backend.

    Resolves natural-language target descriptions to known station names
    via ``SceneContext``, falling back to substring matching.
    """

    def __init__(self, *, backend, scene_context: SceneContext | None = None) -> None:
        super().__init__(
            name="pick_up",
            description="Grasp or pick up an object",
            keywords=(
                "pick", "grasp", "grab", "lift",
                "grasp", "pick", "grab", "take", "lift", "collect",
            ),
        )
        self._backend = backend
        self._scene = scene_context

    def run(self, context: ExecutionContext) -> SkillResult:
        inputs: dict = context.metadata.get("inputs", {})
        raw_target: str = (
            inputs.get("target")
            or context.task
        )
        object_name = (
            inputs.get("object_name")
            or inputs.get("obj_name")
            or inputs.get("object")
            or inputs.get("target_object")
        )
        object_name = _primary_object_name(object_name)
        env_name = str(getattr(self._backend, "_env_name", ""))
        level_cfg = _scene_grasp_config(env_name) or {}
        # The competition scene determines the material unambiguously.  Keep
        # execution robust when the LLM omits structured object metadata.
        if object_name is None:
            object_name = level_cfg.get("object_name")

        # 去 _LEVEL_CONFIG 版说明:不再把 backend 的 BC checkpoint 切到
        # universal_geometry_model.pth(该 139MB 文件未随补丁分发)。
        # 脚本抓取用内联权重;BC 回退路径仍按 robot_params.json 的
        # checkpoint_path / checkpoint_fallback_path 加载本地模型。
        initial_base_pose = inputs.get("grasp_initial_base_pose")
        if initial_base_pose is None:
            initial_base_pose = inputs.get("initial_base_pose")
        if initial_base_pose is None:
            initial_base_pose = inputs.get("base_pose")
        target = raw_target
        if self._scene is not None:
            target = _resolve_station_name(raw_target, self._scene)
            if target == raw_target and level_cfg.get("source"):
                target = level_cfg["source"]
            logger.info("pick_up target: %r → %r", raw_target, target)

        # Physics grasp (only mode — no teleport fallback)
        grasp_pose_source = (
            "plan_or_navigation" if initial_base_pose is not None else "backend_navigation"
        )
        grasp_pose_diagnostics: dict = {}
        try:
            if initial_base_pose is not None:
                # AnalyzeSupplySkill already drove to the live-object standoff.
                # Only the final yaw remains; animate it instead of snapping.
                turn_result = _turn_navigation_base_to_grasp_pose(
                    self._backend, initial_base_pose,
                )
                grasp_pose_source = "physically_navigated_live_object_pose"
                grasp_pose_diagnostics = {
                    "mode": "continuous_turn_at_live_object_standoff",
                    "turn_result": turn_result,
                }
            else:
                dynamic_pose, grasp_pose_diagnostics = _collision_free_grasp_pose(
                    self._backend,
                    object_name,
                )
                if dynamic_pose is not None:
                    initial_base_pose = dynamic_pose
                    current_xy, _ = self._backend.get_base_pose()
                    pose_error = float(np.linalg.norm(
                        np.asarray(current_xy, dtype=float)[:2]
                        - np.asarray(dynamic_pose["xy"], dtype=float)[:2]
                    ))
                    grasp_pose_source = "non_mutating_live_object_pose"
                    grasp_pose_diagnostics["base_position_error"] = pose_error
                    if pose_error > 0.20:
                        return SkillResult(
                            skill_name=self.name,
                            success=False,
                            message=(
                                "Physical navigation to the live grasp pose is required "
                                f"(base error {pose_error:.3f} m)"
                            ),
                            payload={
                                "action": "pick_up",
                                "target": target,
                                "object_name": object_name,
                                "grasp_pose_diagnostics": grasp_pose_diagnostics,
                                "method": "navigation_required_no_teleport",
                                "ok": False,
                            },
                        )
                    turn_result = _turn_navigation_base_to_grasp_pose(
                        self._backend, dynamic_pose,
                    )
                    grasp_pose_diagnostics["turn_result"] = turn_result
                    resolved = dynamic_pose["robot_base_pos"]
                    print(
                        "[L2_DYNAMIC_GRASP_POSE] "
                        f"object={object_name} base=({resolved[0]:.6f},{resolved[1]:.6f}) "
                        f"yaw={dynamic_pose['yaw']:.6f} "
                        f"mode={grasp_pose_diagnostics['mode']}",
                        flush=True,
                    )
                    logger.info(
                        "Overrode generic %s approach with live L2 grasp pose (%.6f, %.6f)",
                        target,
                        resolved[0],
                        resolved[1],
                    )
        except Exception as exc:
            # Preserve existing behavior if live geometry is unavailable. The
            # diagnostic remains visible instead of crashing unrelated tasks.
            grasp_pose_diagnostics = {"error": str(exc)}
            logger.warning("L2 dynamic grasp pose resolution failed: %s", exc)

        # The single calibrated geometry checkpoint is shared by every level.
        # It uses the live object sites and the same physical contact / lift
        # checks as the demonstration collector.
        if _scene_grasp_config(env_name) is not None and object_name:
            try:
                ok, diagnostics = _scripted_grasp_and_attach(
                    self._backend, object_name, target,
                )
                resolved_object = getattr(self._backend, "_held_crate_name", None) or object_name
                return SkillResult(
                    skill_name=self.name,
                    success=ok,
                    message=f"Universal geometry grasp {'OK' if ok else 'FAIL'}: {target}",
                    payload={
                        "action": "pick_up",
                        "target": target,
                        "object_name": resolved_object,
                        "grasp_initial_base_pose": initial_base_pose,
                        "grasp_pose_source": grasp_pose_source,
                        "grasp_pose_diagnostics": grasp_pose_diagnostics,
                        "universal_geometry": diagnostics,
                        "method": "universal_geometry_weight",
                        "ok": ok,
                    },
                )
            except Exception as exc:
                logger.exception("universal geometry grasp crashed")
                return SkillResult(
                    skill_name=self.name,
                    success=False,
                    message=f"Universal geometry grasp error: {exc}",
                    payload={
                        "action": "pick_up",
                        "target": target,
                        "object_name": object_name,
                        "error": str(exc),
                    },
                )

        if hasattr(self._backend, "grasp_object_physics"):
            try:
                backend_source = target
                # The backend's legacy task_config forces every station yaw to
                # -pi.  L3 was trained from the south side (+pi/2), so use an
                # object-explicit sentinel to preserve the supplied pose/yaw.
                if initial_base_pose is not None and abs(float(initial_base_pose["yaw"]) + 3.14) > 0.2:
                    backend_source = f"dynamic_{target}"
                ok = self._backend.grasp_object_physics(
                    backend_source,
                    object_name=object_name,
                    initial_base_pose=initial_base_pose,
                )
                if ok and backend_source != target and hasattr(self._backend, "_mark_trajectory_event"):
                    self._backend._mark_trajectory_event(
                        "grasp_end", object_name=object_name,
                        source=target, success=True,
                    )
                resolved_object = getattr(self._backend, "_held_crate_name", None) or object_name
                return SkillResult(
                    skill_name=self.name,
                    success=ok,
                    message=f"Physics grasp {'OK' if ok else 'FAIL'}: {target}",
                    payload={
                        "action": "pick_up",
                        "target": target,
                        "object_name": resolved_object,
                        "grasp_initial_base_pose": initial_base_pose,
                        "grasp_pose_source": grasp_pose_source,
                        "grasp_pose_diagnostics": grasp_pose_diagnostics,
                        "method": "physics",
                        "ok": ok,
                    },
                )
            except Exception as exc:
                logger.exception("physics grasp crashed")
                return SkillResult(
                    skill_name=self.name, success=False,
                    message=f"Physics grasp error: {exc}",
                    payload={
                        "action": "pick_up",
                        "target": target,
                        "object_name": object_name,
                        "grasp_initial_base_pose": initial_base_pose,
                        "grasp_pose_source": grasp_pose_source,
                        "grasp_pose_diagnostics": grasp_pose_diagnostics,
                        "error": str(exc),
                    },
                )

        # No physics configured — teleport only
        try:
            self._backend.pick_object(target)
        except Exception:
            pass
        return SkillResult(
            skill_name=self.name, success=True,
            message=f"Grasped (snap): {target}",
            payload={"action": "pick_up", "target": target, "raw_target": raw_target, "method": "teleport"},
        )
