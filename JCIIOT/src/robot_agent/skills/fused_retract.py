"""Fused arm-retract leg segment (skills-side, legal modification zone).

The post-grasp arm retract (L1-L4) and the pre-grasp posture restore
(L5 empty-return legs) are animated across the first steps of the next
navigation leg instead of snapping the joints in a single qpos write:
the base follows the A* path as usual while the upper body interpolates
toward the target posture, so the retract costs no in-place waiting.

After the fused steps the remaining path is handed back to the official
``follow_path`` driver, which locks the arms at the (already reached)
target posture — the official ``environments/`` code stays byte-identical.
This module only imports helpers from the backend (use ≠ modification).
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)

# 融合回缩步数:最大关节行程 ~50° ÷ 12 步 ≈ 4°/步,视觉自然。
_RETRACT_ANIM_STEPS = 12


def posture_index_from_named(env, named: dict) -> dict:
    """Convert a ``{joint_name: (qpos, qvel)}`` snapshot into the index
    format used by the backend's posture helpers."""
    names = [str(j) for j in named.keys()]
    return {
        "joint_names": names,
        "qpos_indexes": [env.sim.model.get_joint_qpos_addr(j) for j in names],
        "qvel_indexes": [env.sim.model.get_joint_qvel_addr(j) for j in names],
        "qpos": np.asarray(
            [np.asarray(named[j][0], dtype=float).reshape(-1) for j in names],
            dtype=float,
        ).reshape(-1),
        "qvel": np.asarray(
            [np.asarray(named[j][1], dtype=float).reshape(-1) for j in names],
            dtype=float,
        ).reshape(-1),
    }


def _blend_posture(env, start: dict, target: dict, alpha: float) -> None:
    """Set the upper body to the linear blend of *start* and *target*.

    The two posture sets can differ in joint coverage (the retract target
    snapshot excludes gripper joints, the leg-start capture includes them),
    so joints are matched by name and only the shared ones are blended —
    joints present only in the target fall to the target value.
    """
    start_by_name = dict(zip(start["joint_names"], start["qpos"]))
    indexes = target["qpos_indexes"]
    for i, joint_name in enumerate(target["joint_names"]):
        start_value = start_by_name.get(joint_name)
        target_value = float(target["qpos"][i])
        value = (
            target_value
            if start_value is None
            else float(start_value) + (target_value - float(start_value)) * float(alpha)
        )
        env.sim.data.qpos[indexes[i]] = value
    env.sim.data.qvel[target["qvel_indexes"]] = target["qvel"]
    env.sim.forward()


def run_fused_retract_segment(backend, path) -> list[np.ndarray]:
    """Drive the first ``_RETRACT_ANIM_STEPS`` steps of *path* while the
    upper body interpolates to ``backend._pending_retract_posture``.

    Returns the remaining path (current base position + not-yet-reached
    waypoints) for the official ``follow_path`` to continue.  When no
    pending posture is set, *path* is returned unchanged.
    """
    target = getattr(backend, "_pending_retract_posture", None)
    if not target:
        return path
    backend._pending_retract_posture = None

    from robot_agent.environments.robosuite_backend import (
        _capture_upper_body_posture,
        _get_base_pose,
        _set_base_xy_direct,
        _should_stop_for_collision,
        _try_sync_transport,
    )

    env = backend.env
    robot = env.robots[0]
    start = _capture_upper_body_posture(env, robot)
    points = [np.asarray(p, dtype=float)[:2] for p in path]
    if not points:
        return path

    nav = backend._rp["navigation"]
    tolerance = float(nav.get("waypoint_tolerance", 0.25))
    max_step = float(backend._max_linear) / float(backend._control_freq)
    warmup = int(getattr(backend, "_collision_warmup_steps", 5))
    max_pairs = int(getattr(backend, "_max_collision_pairs", 8))
    ignore = list(getattr(backend, "_ignore_collision_geom", ()))
    idle_action = np.zeros_like(env.action_spec[0])

    waypoint = 0
    for step in range(_RETRACT_ANIM_STEPS):
        base_xy, _ = _get_base_pose(env)
        goal = points[waypoint]
        delta = goal - base_xy
        distance = float(np.linalg.norm(delta))
        if distance < tolerance:
            waypoint += 1
            if waypoint >= len(points):
                break
            continue
        step_xy = base_xy + delta / max(distance, 1e-6) * min(distance, max_step)
        _set_base_xy_direct(env, robot, step_xy)
        _try_sync_transport(env)
        env.step(idle_action)
        _blend_posture(env, start, target, (step + 1) / _RETRACT_ANIM_STEPS)
        _try_sync_transport(env)
        if _should_stop_for_collision(env, robot, ignore, step, warmup, max_pairs):
            # 与官方驱动同策略:记录并继续行驶
            logger.info("collision logged at fused-retract step %d (continues)", step)
        if hasattr(backend, "_update_held_crate_position"):
            backend._update_held_crate_position()
        if hasattr(backend, "_record_trajectory_frame"):
            backend._record_trajectory_frame()

    base_xy, _ = _get_base_pose(env)
    current = np.array([float(base_xy[0]), float(base_xy[1])], dtype=float)
    return [current, *points[waypoint:]]
