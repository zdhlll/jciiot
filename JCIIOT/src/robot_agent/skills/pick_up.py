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


# 站位距离:底盘到物体中心的标称间距(0.941m)。该值由官方 L1
# 采集教师标定(数据采集器里的 _calibrate_robot_standoff),所有
# 关卡共用。L2 的 input_6 语义图 approach 点是两个并排 tote 的
# 中点,与真实停靠位不符,所以站位一律从运行中的 MuJoCo 几何
# 现算,不读地图坐标。
_BASE_STANDOFF = 0.941

# ── 抓取路由开关 (jci_test 专用:切换脚本/BC 两种抓取) ─────
# "bc"       → 所有关卡走 backend.grasp_object_physics(官方 BC 管线,
#              临时 wrapped env 里跑 grasp_policy.checkpoint_path 模型)。
# "scripted" → 走 _staged_scripted_grasp(阶段表脚本抓取)。
_GRASP_MODE: str = "scripted"

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


# 手臂动作的 OSC 增量限幅(归一化输入上限 ±1 对应此幅度)
_MAX_ACTION = 0.65


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
    """读取物体当前世界中心;按 site → body 映射 → 材质表逐级回退。"""

    model, data = raw_env.sim.model, raw_env.sim.data

    def _site_xyz(site_name: str) -> np.ndarray | None:
        try:
            site_id = model.site_name2id(site_name)
        except (KeyError, ValueError):
            return None
        return np.asarray(data.site_xpos[site_id], dtype=float).copy()

    for candidate in (f"{object_name}_center_site", f"{object_name}_default_site"):
        position = _site_xyz(candidate)
        if position is not None:
            return position

    mapping = getattr(raw_env, "obj_body_id", {}) or {}
    body_id = mapping.get(object_name)
    if body_id is not None:
        return np.asarray(data.body_xpos[body_id], dtype=float).copy()

    material = (getattr(raw_env, "material_metadata", {}) or {}).get(object_name, {}) or {}
    body_name = material.get("body_name")
    if body_name:
        return np.asarray(data.body_xpos[model.body_name2id(body_name)], dtype=float).copy()
    raise ValueError(f"Cannot resolve live centre for object {object_name!r}")


def _live_standoff_pose(backend, object_name: str | None) -> tuple[dict | None, dict]:
    """由运行中的物体中心现算抓取站位,不改动仿真状态。

    站位 = 物体中心 − 朝向(关卡 yaw)×0.941m。底盘通过 MoveSkill
    物理开到位,这里只负责计算;早期直接改 base qpos 的写法会让
    轨迹记录看起来像瞬移,已废弃。
    """

    env_name = str(getattr(backend, "_env_name", ""))
    cfg = _scene_grasp_config(env_name)
    if cfg is None or cfg.get("yaw") is None:
        return None, {}

    effective_object = object_name or cfg["object_name"]
    base_yaw = cfg["yaw"]

    raw_env = _base_robosuite_env(getattr(backend, "env", None))
    raw_env.sim.forward()
    object_center = _live_object_center(raw_env, effective_object)

    heading = np.array([np.cos(base_yaw), np.sin(base_yaw)], dtype=float)
    base_xy = object_center[:2] - heading * _BASE_STANDOFF

    pose = {
        "robot_base_pos": [float(base_xy[0]), float(base_xy[1]), 0.0],
        "robot_base_ori": [0.0, 0.0, base_yaw],
        "xy": [float(base_xy[0]), float(base_xy[1])],
        "yaw": base_yaw,
    }
    diagnostics = {
        "mode": "live_standoff",
        "env_name": env_name,
        "object_name": effective_object,
        "object_center": object_center.tolist(),
        "heading": heading.tolist(),
        "standoff": _BASE_STANDOFF,
        "resolved_base_position": pose["robot_base_pos"],
        "resolved_base_yaw": base_yaw,
    }
    return pose, diagnostics


# ── 分阶段脚本抓取(重写版,替换原 universal-geometry 实现) ──
# 相位表驱动:每个相位先按步数插值逼近,再以有界闭环收敛;
# 每阶段独立报告成败,失败时指出具体阶段。
# 步数采用加速档(40/80/80/50/40);此前线上验证过的保守档为
# 45/100/75/100/45(settle 100),回归异常时切回。
# 相位合并:原"抬升(40)+平移(80)"两段合并为一段斜线(80)。
# 斜线中间点高度线性上升,不低于两端;起点过低时(见 _LIFT_GUARD_MARGIN)
# 仍先补一段 20 步纯抬升,防止从低姿态斜插扫到桌面/货架。
_PHASE_TABLE: tuple[tuple[str, int], ...] = (
    ("diagonal_approach", 80),
    ("vertical_descent", 70),
)
_LIFT_GUARD_MARGIN = 0.12
_SETTLE_MAX_STEPS = 30
_SETTLE_TOLERANCE = 0.035
_CLOSE_STEPS = 25
_RETRACT_BY_LEVEL = (1, 1, 1, 1, 0)


def _geometry_collector():
    from robosuite.environments.factory_sorting import (
        load_factory_sorting_1_3fo3erfhisem_collect as collector,
    )
    return collector


def _rotate_site_template_to_approach(env, robot, object_name: str) -> tuple[dict, dict]:
    """把官方抓取 site 对绕物体中心旋转到正对底盘侧,再降到夹持深度。

    场景给每个物体一对固定夹点;夹点连线的中点可能在物体任意一侧。
    这里以"底盘→物体"方向为近轴重建夹点对:中点沿近轴推出模板进深,
    两点沿垂直轴按模板半跨分开,高度取 site 高度下方 site_below_offset。
    """
    model, data = env.sim.model, env.sim.data
    site_of = {
        arm: np.asarray(
            data.site_xpos[model.site_name2id(f"{object_name}_{arm}_grasp_site")],
            dtype=float,
        ).copy()
        for arm in ("right", "left")
    }
    centre = _live_object_center(env, object_name)
    midpoint = (site_of["right"] + site_of["left"]) / 2.0
    span = site_of["right"][:2] - site_of["left"][:2]
    half_span = float(np.linalg.norm(span) / 2.0)
    near_offset = float(np.linalg.norm(midpoint[:2] - centre[:2]))

    base_xy = np.asarray(
        data.site_xpos[
            model.site_name2id(robot.robot_model.base.correct_naming("center"))
        ],
        dtype=float,
    )[:2]
    near = base_xy - centre[:2]
    norm = float(np.linalg.norm(near))
    if norm <= 1e-9:
        raise RuntimeError("grasp staging: base overlaps object centre")
    near /= norm
    lateral = np.array([-near[1], near[0]], dtype=float)

    span_unit = span / (2.0 * half_span) if half_span > 1e-9 else np.zeros(2)
    usable = (
        half_span > 1e-9
        and abs(float(np.dot(span_unit, lateral))) >= 0.75
        and abs(float(np.dot(midpoint[:2] - centre[:2], near))) >= near_offset * 0.75
    )
    target_z = float((site_of["right"][2] + site_of["left"][2]) / 2.0 - 0.035)

    def _assign_sides(side_a: np.ndarray, side_b: np.ndarray) -> dict:
        """夹点对落在横轴两端;按当前手部位置就近分配给左右臂。"""
        gripper_xy = {
            arm: _geometry_collector().gripper_end_center_pos(env, robot, arm)[:2]
            for arm in ("right", "left")
        }

        def _cost(candidate: dict) -> float:
            return float(sum(
                np.linalg.norm(candidate[arm] - gripper_xy[arm]) ** 2
                for arm in ("right", "left")
            ))

        return min(
            (
                {"right": side_a, "left": side_b},
                {"right": side_b, "left": side_a},
            ),
            key=_cost,
        )

    if usable:
        sign = 1.0 if float(np.dot(span_unit, lateral)) >= 0.0 else -1.0
        half_axis = lateral * (half_span * sign)
        aligned_mid = centre[:2] + near * near_offset
        assignment = _assign_sides(aligned_mid + half_axis, aligned_mid - half_axis)
        below = {
            arm: np.array([assignment[arm][0], assignment[arm][1], target_z], dtype=float)
            for arm in ("right", "left")
        }
        return below, {"target_source": "site-template rotation", "half_span": half_span}

    # 兜底:模板朝向与底盘侧不一致(L2 的 tote 就属此类)时,
    # 取朝向机器人的那面碰撞墙:夹点进深 = 墙在近轴上的投影,
    # 夹点半跨 = 墙在横轴上的半宽,再按"模板半跨/物体整体半宽"
    # 的比例缩放(与官方 site 模板的宽度比例保持一致)。
    geom_records = []
    for geom_name in _geometry_collector().object_collision_geoms(env, object_name):
        gid = model.geom_name2id(geom_name)
        gpos = np.asarray(data.geom_xpos[gid], dtype=float)[:2]
        gmat = np.asarray(data.geom_xmat[gid], dtype=float).reshape(3, 3)
        gsize = np.asarray(model.geom_size[gid], dtype=float)

        def _half_extent(axis_dir: np.ndarray) -> float:
            d3 = np.array([axis_dir[0], axis_dir[1], 0.0], dtype=float)
            return float(sum(abs(float(np.dot(gmat[:, k], d3))) * gsize[k] for k in range(3)))

        geom_records.append({
            "near_proj": float(np.dot(gpos - centre[:2], near)),
            "span_proj": float(np.dot(gpos - centre[:2], span_unit)),
            "near_half": _half_extent(near),
            "lateral_half": _half_extent(lateral),
            "span_half": _half_extent(span_unit),
        })
    if not geom_records:
        raise RuntimeError("grasp staging: no collision geometry for fallback")
    wall = max(geom_records, key=lambda g: g["near_proj"])
    object_span_extent = max(
        abs(g["span_proj"]) + g["span_half"] for g in geom_records
    )
    if object_span_extent <= 1e-6:
        raise RuntimeError("grasp staging: cannot resolve object extent")
    span_ratio = min(1.0, half_span / object_span_extent)
    fallback_mid = centre[:2] + near * wall["near_proj"]
    fallback_half = wall["lateral_half"] * span_ratio
    assignment = _assign_sides(
        fallback_mid + lateral * fallback_half,
        fallback_mid - lateral * fallback_half,
    )
    below = {
        arm: np.array([assignment[arm][0], assignment[arm][1], target_z], dtype=float)
        for arm in ("right", "left")
    }
    return below, {"target_source": "collision-geometry fallback", "half_span": fallback_half}


def _issue_arm_step(env, robot, targets, *, use_gripper_centers: bool, gripper_value: float) -> None:
    """单步 OSC 增量动作:位置差→控制器系→归一化→env.step。"""
    collector = _geometry_collector()
    robot.composite_controller.update_state()
    arm_actions: dict = {}
    if targets is not None:
        for arm in ("right", "left"):
            current = (
                collector.gripper_end_center_pos(env, robot, arm)
                if use_gripper_centers
                else collector.get_eef_pos(env, robot, arm)
            )
            delta = np.asarray(targets[arm], dtype=float) - current
            controller_delta = collector.world_delta_to_controller_frame(robot, arm, delta)
            arm_actions[arm] = collector.arm_delta_to_normalized_action(
                robot, arm, controller_delta, 0.65,
            )
    env.step(collector.build_action(env, robot, arm_actions, gripper_value))


def _drive_arms_to(env, robot, targets, *, steps: int, gripper_value: float, tolerance,
                   record_cb=None):
    """插值逼近目标,再以有界闭环收敛;返回 (ok, 各臂距离)。"""
    collector = _geometry_collector()
    starts = {arm: collector.get_eef_pos(env, robot, arm) for arm in ("right", "left")}
    for index in range(1, max(1, int(steps)) + 1):
        alpha = index / float(max(1, int(steps)))
        _issue_arm_step(
            env, robot,
            {arm: starts[arm] + alpha * (np.asarray(targets[arm]) - starts[arm])
             for arm in ("right", "left")},
            use_gripper_centers=False, gripper_value=gripper_value,
        )
        if record_cb is not None:
            record_cb()
    distances = {
        arm: float(np.linalg.norm(collector.get_eef_pos(env, robot, arm) - targets[arm]))
        for arm in ("right", "left")
    }
    if tolerance is None:
        return True, distances
    for _ in range(100):
        if all(d <= tolerance for d in distances.values()):
            return True, distances
        _issue_arm_step(env, robot, targets, use_gripper_centers=False, gripper_value=gripper_value)
        if record_cb is not None:
            record_cb()
        distances = {
            arm: float(np.linalg.norm(collector.get_eef_pos(env, robot, arm) - targets[arm]))
            for arm in ("right", "left")
        }
    return all(d <= tolerance for d in distances.values()), distances


def _settle_gripper_centers_at(env, robot, targets, record_cb=None) -> tuple[bool, dict]:
    """用 gripper 末端中心闭环收敛到目标(提前退出)。"""
    collector = _geometry_collector()
    for _ in range(_SETTLE_MAX_STEPS):
        distances = {
            arm: float(np.linalg.norm(collector.gripper_end_center_pos(env, robot, arm) - targets[arm]))
            for arm in ("right", "left")
        }
        if all(d <= _SETTLE_TOLERANCE for d in distances.values()):
            return True, distances
        _issue_arm_step(env, robot, targets, use_gripper_centers=True, gripper_value=-1.0)
        if record_cb is not None:
            record_cb()
    distances = {
        arm: float(np.linalg.norm(collector.gripper_end_center_pos(env, robot, arm) - targets[arm]))
        for arm in ("right", "left")
    }
    return all(d <= _SETTLE_TOLERANCE for d in distances.values()), distances



def _staged_scripted_grasp(backend, object_name: str, source: str) -> tuple[bool, dict]:
    """分阶段脚本抓取:L1-L5 通用;底盘位姿由调用方负责。"""
    from robosuite.environments.factory_sorting.lift_after_grasp import (
        lift_grasped_object,
    )
    from robosuite.environments.factory_sorting.transport_attachment import (
        capture_transport_attachment,
    )

    collector = _geometry_collector()
    env = _base_robosuite_env(getattr(backend, "env", None))
    robot = env.robots[0]
    stages: list = []

    # 多箱关卡(L5)连续抓取会累积漂移:首抓前快照上肢姿态,每次开抓前恢复。
    posture = getattr(backend, "_grasp_posture_snapshot", None)
    if posture is None:
        posture = {}
        base_joints = set(getattr(robot.robot_model, "base_joints", []) or [])
        for joint_name in env.sim.model.joint_names:
            if not joint_name or not joint_name.startswith("robot0_") or joint_name in base_joints:
                continue
            try:
                posture[joint_name] = (
                    np.asarray(env.sim.data.get_joint_qpos(joint_name), dtype=float).copy(),
                    np.asarray(env.sim.data.get_joint_qvel(joint_name), dtype=float).copy(),
                )
            except Exception:
                continue
        backend._grasp_posture_snapshot = posture
    else:
        for joint_name, (qpos, qvel) in posture.items():
            try:
                env.sim.data.set_joint_qpos(joint_name, qpos)
                env.sim.data.set_joint_qvel(joint_name, qvel)
            except Exception:
                continue
        env.sim.forward()

    setattr(robot, collector.CAMERA_HOLD_TARGET_ATTR, collector.capture_camera_hold_targets(robot))
    record_counter = [0]

    def record_frame() -> None:
        record_counter[0] += 1
        if record_counter[0] % 5 == 0 and hasattr(backend, "_record_trajectory_frame"):
            backend._record_trajectory_frame()

    if hasattr(backend, "_record_trajectory_frame"):
        backend._record_trajectory_frame()
    if hasattr(backend, "_mark_trajectory_event"):
        backend._mark_trajectory_event("grasp_start", object_name=object_name, source=source)

    def fail(stage: str) -> tuple[bool, dict]:
        if hasattr(backend, "_mark_trajectory_event"):
            backend._mark_trajectory_event(
                "grasp_end", object_name=object_name, source=source, success=False,
            )
        return False, {"method": "staged_scripted_grasp", "failed_stage": stage, "stages": stages}

    try:
        below_targets, target_meta = _rotate_site_template_to_approach(env, robot, object_name)
    except Exception as exc:
        return fail(f"targets: {exc}")

    site_positions = {
        arm: below_targets[arm] + np.array([0.0, 0.0, 0.035])
        for arm in ("right", "left")
    }
    starts = {arm: collector.get_eef_pos(env, robot, arm) for arm in ("right", "left")}
    safe_z = max(
        max(float(p[2]) for p in starts.values()),
        max(float(p[2] + 0.05) for p in site_positions.values()),
    )
    waypoints = {
        "safe_vertical": {
            arm: np.array([starts[arm][0], starts[arm][1], safe_z], dtype=float)
            for arm in ("right", "left")
        },
        "diagonal_approach": {
            arm: np.array([site_positions[arm][0], site_positions[arm][1], safe_z], dtype=float)
            for arm in ("right", "left")
        },
        "vertical_descent": below_targets,
    }

    # 起点手部高度足够时直接斜线接近;起点过低(L5 第二/三箱,
    # 刚放完箱手臂在低位)先补一段短抬升,防斜线扫过桌面。
    phase_plan = list(_PHASE_TABLE)
    lowest_start_z = min(float(p[2]) for p in starts.values())
    if lowest_start_z < safe_z - _LIFT_GUARD_MARGIN:
        phase_plan.insert(0, ("safe_vertical", 20))

    for label, steps in phase_plan:
        ok, distances = _drive_arms_to(
            env, robot, waypoints[label],
            steps=steps, gripper_value=-1.0, tolerance=None,
            record_cb=record_frame,
        )
        stages.append({"stage": label, "success": ok, "distances": distances})
        if not ok:
            return fail(label)

    settled, distances = _settle_gripper_centers_at(
        env, robot, below_targets, record_cb=record_frame,
    )
    stages.append({"stage": "center_settle", "success": settled, "distances": distances})
    if not settled:
        return fail("center_settle")

    for _ in range(_CLOSE_STEPS):
        _issue_arm_step(env, robot, None, use_gripper_centers=False, gripper_value=1.0)
        record_frame()

    grasp_status = collector.grasp_status(env, robot, object_name)
    grasped = all(bool(v) for v in grasp_status.values())
    stages.append({"stage": "close", "success": grasped, "grasp_status": grasp_status})

    lift_result: dict = {"success": False, "failure_reason": "grasp failed"}
    if grasped:
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
            render_callback=record_frame,
        )
    stages.append({"stage": "lift_verify", "success": bool(lift_result.get("success")), "lift": lift_result})
    if not grasped or not lift_result.get("success"):
        return fail("lift_verify" if grasped else "close")

    capture_transport_attachment(env, object_name)
    backend._held_crate_name = object_name

    env_name = str(getattr(backend, "_env_name", ""))
    level_index = (
        _COMPETITION_ENV_ORDER.index(env_name)
        if env_name in _COMPETITION_ENV_ORDER
        else 0
    )
    should_retract = bool(
        _RETRACT_BY_LEVEL[level_index] if level_index < len(_RETRACT_BY_LEVEL) else True
    )
    if should_retract:
        # 不再一次性 qpos 直设回 ready 姿态(轨迹里表现为手臂瞬移),
        # 而是把目标姿态挂到后端: 下一段导航腿的前 12 步内由
        # skills/fused_retract.py 边行驶边逐帧插值(融合回缩,
        # 底盘离站与收臂并行, 无原地等待, 官方驱动层零改动)。
        from robot_agent.skills.fused_retract import posture_index_from_named

        try:
            backend._pending_retract_posture = posture_index_from_named(
                env, posture,
            )
        except Exception:
            backend._pending_retract_posture = None

    if hasattr(backend, "_record_trajectory_frame"):
        backend._record_trajectory_frame()
    if hasattr(backend, "_mark_trajectory_event"):
        backend._mark_trajectory_event(
            "grasp_end", object_name=object_name, source=source, success=True,
        )
    return True, {
        "method": "staged_scripted_grasp",
        "stages": stages,
        "target_meta": target_meta,
        "lift": lift_result,
    }
class _PostureLock:
    """转向期间锁住上肢姿态。

    底盘零动作时手臂/躯干/头部不允许松劲——抓取模板标定的就是
    这套 ready 姿态,转向全程需要逐帧恢复快照。
    """

    def __init__(self, backend, raw_env) -> None:
        from robot_agent.environments.robosuite_backend import (
            _capture_upper_body_posture,
            _restore_upper_body_posture,
        )

        self._backend = backend
        self._raw_env = raw_env
        self._restore = _restore_upper_body_posture
        self._posture = _capture_upper_body_posture(raw_env, raw_env.robots[0])

    def hold_and_record(self) -> None:
        self._restore(self._raw_env, self._posture)
        if hasattr(self._backend, "_record_trajectory_frame"):
            self._backend._record_trajectory_frame()

    def release(self) -> None:
        self._restore(self._raw_env, self._posture)


def _combined_or_inplace_turn(backend, pose: dict) -> dict:
    """原地转向到抓取 yaw。

    边开边转的旋转已由 move 层沿 A* 网格安全腿完成(见 move.py 的
    goal_yaw 注入),到这里只剩零角度残量,原地转向会立即提前退出。
    """
    return _turn_to_grasp_yaw(backend, pose)


def _turn_to_grasp_yaw(backend, pose: dict) -> dict:
    """原地连续转向到抓取 yaw,底盘 XY 保持不变。"""

    from robosuite.environments.factory_sorting.load_factory_sorting_evalization import (
        get_base_world_pose,
    )
    from robosuite.environments.factory_sorting.turn_to_station import turn_to_face_xy

    raw_env = _base_robosuite_env(getattr(backend, "env", None))
    lock = _PostureLock(backend, raw_env)
    base_xy, _ = get_base_world_pose(raw_env, raw_env.robots[0])
    target_yaw = float(pose["yaw"])
    look_at = np.asarray(base_xy, dtype=float)[:2] + np.array(
        [np.cos(target_yaw), np.sin(target_yaw)], dtype=float,
    )

    result = turn_to_face_xy(
        env=raw_env,
        target_xy=look_at,
        tolerance=0.02,
        max_iters=8,
        turn_steps=40,
        settle_steps=10,
        render=not bool(getattr(backend, "_headless", True)),
        render_sleep=0.0,
        sync_attachment=False,
        post_step_callback=lock.hold_and_record,
    )
    lock.release()
    if not result.get("success", False):
        raise RuntimeError(
            "Grasp turn failed: "
            f"final_error={result.get('final_error')}, xy_drift={result.get('xy_drift')}"
        )
    return result


def _primary_object_name(value) -> str | None:
    """从任务输入的物体字段里取第一个非空名字(兼容 str/list/tuple)。"""
    if isinstance(value, str):
        stripped = value.strip()
        return stripped or None
    if isinstance(value, (list, tuple)):
        for item in value:
            resolved = _primary_object_name(item)
            if resolved:
                return resolved
    return None


def _resolve_station_name(target: str, scene: SceneContext) -> str:
    """把自然语言目标映射到语义地图里的站位名。

    示例:"在1号进料口抓取目标物体" → "input_1";
    "input_1" 原样返回。子串匹配按名字长度降序,保证
    "去aux_input_1取料" 命中 "aux_input_1" 而不是 "input_1"。
    """
    known = scene.all_port_names()
    if not known or target in known:
        return target

    # 子串匹配:长名优先
    for name in sorted(known, key=len, reverse=True):
        if name in target:
            return name

    # (角色, 编号) 匹配:"1号进料口" → input #1
    role, idx = _parse_role_index(target)
    if role and idx is not None:
        for name in known:
            info = (scene.input_ports.get(name) or scene.output_ports.get(name))
            if info is not None and info.role == role and info.index == int(idx):
                return name

    return target


def _parse_role_index(text: str) -> tuple[str | None, int | None]:
    """从文本里提取 (角色, 编号),如 "1号进料口" → ("input", 1)。"""
    normalized = str(text)
    for cn_digit, arabic in _CN_DIGIT.items():
        normalized = normalized.replace(cn_digit, arabic)

    # 中文形式:数字 + 可选"号/#" + 角色字
    match = re.search(r"(\d+)\s*[号#]?\s*([进出入输][料料入出])", normalized)
    if match:
        role_chars = match.group(2)
        for cn_word, role in _CN_ROLE.items():
            if cn_word in role_chars:
                return role, int(match.group(1))

    # 英文形式:"input_5" / "output_3"
    match = re.search(r"(input|output)\s*_?\s*(\d+)", text, re.IGNORECASE)
    if match:
        return match.group(1).lower(), int(match.group(2))

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
        # 关卡场景唯一确定物料;LLM 漏给结构化物体字段时保持可用。
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
                # 编排层已把底盘开到 live 站位;这里只差最后转向,
                # 用动画转向而不是直接 snap。
                turn_result = _combined_or_inplace_turn(
                    self._backend, initial_base_pose,
                )
                grasp_pose_source = "physically_navigated_live_object_pose"
                grasp_pose_diagnostics = {
                    "mode": "continuous_turn_at_live_object_standoff",
                    "turn_result": turn_result,
                }
            else:
                dynamic_pose, grasp_pose_diagnostics = _live_standoff_pose(
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
                    turn_result = _combined_or_inplace_turn(
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
            # live 几何不可用时保持原有行为:只记录诊断,不拖垮任务。
            grasp_pose_diagnostics = {"error": str(exc)}
            logger.warning("L2 dynamic grasp pose resolution failed: %s", exc)

        # 所有关卡共用同一套脚本抓取:目标点来自 live site 几何,
        # 接触与抬升验证沿用官方采集器的判定逻辑。
        if _GRASP_MODE == "scripted" and _scene_grasp_config(env_name) is not None and object_name:
            try:
                ok, diagnostics = _staged_scripted_grasp(
                    self._backend, object_name, target,
                )
                resolved_object = getattr(self._backend, "_held_crate_name", None) or object_name
                return SkillResult(
                    skill_name=self.name,
                    success=ok,
                    message=f"Staged scripted grasp {'OK' if ok else 'FAIL'}: {target}",
                    payload={
                        "action": "pick_up",
                        "target": target,
                        "object_name": resolved_object,
                        "grasp_initial_base_pose": initial_base_pose,
                        "grasp_pose_source": grasp_pose_source,
                        "grasp_pose_diagnostics": grasp_pose_diagnostics,
                        "staged_grasp": diagnostics,
                        "method": "staged_scripted_grasp",
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
                # 后端旧逻辑把所有站位 yaw 压成 -pi,而 L3 是从南侧
                # (+pi/2)训练的;用 object 前缀哨兵保住传入位姿。
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
