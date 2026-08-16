"""Deterministic end-to-end transport skill for the competition scenes."""

from __future__ import annotations

import logging
import json
from pathlib import Path

import numpy as np

from robot_agent.core.scene_context import SceneContext
from robot_agent.core.types import ExecutionContext, SkillResult
from robot_agent.skills.base import BaseSkill
from robot_agent.skills.move import MoveSkill
from robot_agent.skills.pick_up import PickUpSkill, _live_standoff_pose, _resolve_station_name
from robot_agent.skills.place_down import PlaceDownSkill

logger = logging.getLogger(__name__)

_FALLBACK_COMPETITION_TASKS = {
    "FactorySorting1_3FO3ERFHISEM": ("input_5", "output_4", ["line_5_container_h01_near"]),
    "FactorySorting3_3FO3ERRPH7X9": ("input_6", "output_4", ["green_tote_b01_upper"]),
    "FactorySorting5_3FO3ERTPXEUT": ("aux_input_1", "output_5", ["blue_tote_b01_far_right"]),
    "FactorySorting7_3FO3ERFKY9RN": ("input_2", "output_5", ["blue_container_h01_back_upper"]),
    "FactorySorting9_3FO3ERT2C5FP": (
        "input_1",
        "aux_output_1",
        [
            "white_tote_b01_left_center",
            "white_tote_b01_left_front",
            "white_tote_b01_left_back",
        ],
    ),
}


def _task_spec(env_name: str) -> tuple[str, str, list[str]] | None:
    """任务表统一走 pick_up 的 task_config 读取,不重复读盘。

    L1-L4 的候选物体任取其一;L5 必须按前→中→后顺序取完三个
    (先取中间,两臂包络会在后拉时把后面箱子的支撑扫走,避免它
    在自己那轮翻倒)。
    """
    from robot_agent.skills.pick_up import _load_task_config

    try:
        tasks = _load_task_config().get("tasks", [])
    except Exception:
        logger.exception("task config unavailable for %s", env_name)
        return _FALLBACK_COMPETITION_TASKS.get(env_name)

    for task in tasks:
        if str(task.get("env_name")) != env_name:
            continue
        raw_objects = task.get("object", [])
        object_names = (
            [str(raw_objects)] if isinstance(raw_objects, str)
            else [str(name) for name in raw_objects if name]
        )
        if env_name == "FactorySorting9_3FO3ERT2C5FP":
            ranks = {"front": 0, "center": 1, "back": 2}

            def _rank(name: str) -> int:
                for token, position in ranks.items():
                    if token in name:
                        return position
                return 99

            object_names.sort(key=_rank)
        else:
            object_names = object_names[:1]
        return str(task["source"]), str(task["target"]), object_names
    return _FALLBACK_COMPETITION_TASKS.get(env_name)


def _held_offset_radius(backend) -> float:
    """携物时箱子相对底盘的偏移半径;附着数据缺失时取保守值。"""

    from robosuite.environments.factory_sorting.transport_attachment import (
        TRANSPORT_ATTACHMENT_ATTR,
    )
    from robot_agent.skills.pick_up import _base_robosuite_env

    attachment = getattr(
        _base_robosuite_env(getattr(backend, "env", None)),
        TRANSPORT_ATTACHMENT_ATTR,
        {},
    ) or {}
    offset = np.asarray(attachment.get("relative_xy", []), dtype=float)
    radius = float(np.linalg.norm(offset)) if offset.size == 2 else 0.0
    return radius if radius > 0.1 else 0.94


def _place_slots(scene, target: str, object_index: int, object_count: int, radius: float):
    """按箱数把放置点沿桌面长轴排开,并给出底盘的站位目标。

    槽位表:count 个箱子均匀分布,间距 _SLOT_SPACING(0.59m)。L5 的
    辅助桌沿世界 x 轴较长,南侧接近时最后的原地转向会把箱子转
    90°,0.48m 宽的箱子按 0.59m 中心距摆放互不重叠。填充顺序从
    右到左:携带箱在底盘西侧,转向时扫过西/北象限,这个顺序让它
    始终背向已经上桌的箱子。
    """
    station = scene.output_ports[target]
    center = np.asarray(station.center[:2], dtype=float)
    release_xy = center.copy()
    if object_count > 1:
        offsets = [
            (object_count - 1) / 2.0 - index for index in range(object_count)
        ]
        release_xy[0] += offsets[object_index] * _SLOT_SPACING

    outward = np.asarray(station.approach[:2], dtype=float) - center
    outward_norm = float(np.linalg.norm(outward))
    if outward_norm < 1e-6:
        outward = np.array([-1.0, 0.0], dtype=float)
    else:
        outward /= outward_norm
    base_goal = release_xy + outward * float(radius)
    return base_goal, release_xy


def _l5_aisle_waypoint(scene, source: str, target: str) -> tuple[float, float]:
    """L5 班车的走廊中线 y 与上料口东侧的安全出站 x。

    仅看底盘足迹,A* 最短路径会直接切进 input_1 与北侧两张桌子
    之间的过道;但携带箱悬在底盘西侧约 0.95m,必须先向东拉出
    货架、再进过道。两个坐标全部由语义地图的站位几何现算,地图
    重新生成后依然有效。

    走廊窗口由两个边沿夹出:目标桌南沿(approach[1]+0.50m)与
    input_1 北沿(中心+0.84m 半长)。两沿之间的中点即走廊中线。
    """
    source_station = scene.input_ports[source]
    target_station = scene.output_ports[target]
    source_center = np.asarray(source_station.center[:2], dtype=float)

    south_edge = float(target_station.approach[1] + 0.50)
    north_edge = float(source_center[1] + 0.84)
    if south_edge <= north_edge:
        raise ValueError("L5 north corridor is not open in the semantic map")

    corridor_y = (south_edge + north_edge) / 2.0
    # 出站 x:此前验证过的路线在 input_1 东侧 x≈-12.8 通过;
    # 同样的净空相对站位边沿表达为 approach 点东移 0.27m。
    source_clear_x = float(source_station.approach[0] + 0.27)
    return corridor_y, source_clear_x


# ── rack departure corridor ───────────────────────────────
# Several pick racks hold the interchangeable candidate totes side by side
# (L2: green_tote_b01_upper/lower, L3: blue_tote_b01_far/near_right).  The
# held tote rides ``relative_xy`` from the base (about 0.95 m to the side),
# so the base-only A* shortest path hugs the rack and sweeps the held tote
# straight through the neighbour, knocking it off the shelf (observed in
# both 2026-08-13 L2 runs for green_tote_b01_lower and the L3 run for
# blue_tote_b01_near_right).  The route below pulls the base backward first
# so the tote slides off the rack toward the robot's own approach side, then
# travels along the rack row until the tote clears every other material
# object, then hands over to A* for the final approach.  Clearances are
# derived from live MuJoCo geometry (neighbour/material collision geoms,
# held-tote offset, scene AABB proxies); the fallback constants were
# measured on the official map.
_L3_CLEAR_MARGIN = 0.30
_L3_TRAVEL_MARGIN = 0.50
_L3_MIN_PULL = 0.30
_L3_FALLBACK_PULL = 0.90
_L3_FALLBACK_TRAVEL = 2.50
_L3_FALLBACK_TOTE_HALF = 0.45

# L5 多箱放置的桌面槽位中心距
_SLOT_SPACING = 0.59


def _alternate_object_name(env_name: str, primary: str) -> str | None:
    """Return the other interchangeable candidate object for *env_name*, if any."""

    config_path = Path(__file__).resolve().parents[3] / "knowledge" / "task_config.json"
    try:
        tasks = json.loads(config_path.read_text(encoding="utf-8")).get("tasks", [])
        for task in tasks:
            if str(task.get("env_name")) != env_name:
                continue
            objects = task.get("object", [])
            if isinstance(objects, str):
                objects = [objects]
            others = [str(name) for name in objects if str(name) != primary]
            return others[0] if others else None
    except Exception:
        logger.exception("Could not resolve the alternate object for %s", env_name)
    return None


def _geom_aabb(raw_env, geom_names) -> tuple[float, float, float, float] | None:
    """World (x0, x1, y0, y1) AABB of *geom_names*, honouring geom rotation.

    Box half-extents are in the geom's local frame; the world half-extent
    along each axis is ``|xmat| @ size`` (exact for axis-aligned rotations,
    which the tote wall geoms use).
    """

    model = raw_env.sim.model
    data = raw_env.sim.data
    x0 = y0 = np.inf
    x1 = y1 = -np.inf
    found = False
    for geom_name in geom_names:
        try:
            geom_id = model.geom_name2id(geom_name)
        except Exception:
            continue
        pos = data.geom_xpos[geom_id]
        half = np.asarray(model.geom_size[geom_id], dtype=float)
        rot = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        world_half = np.abs(rot) @ half
        x0 = min(x0, float(pos[0] - world_half[0]))
        x1 = max(x1, float(pos[0] + world_half[0]))
        y0 = min(y0, float(pos[1] - world_half[1]))
        y1 = max(y1, float(pos[1] + world_half[1]))
        found = True
    return (x0, x1, y0, y1) if found else None


def _geom_z_range(raw_env, geom_names) -> tuple[float, float] | None:
    """World (z0, z1) extent of *geom_names*, honouring geom rotation."""

    model = raw_env.sim.model
    data = raw_env.sim.data
    z0, z1 = np.inf, -np.inf
    found = False
    for geom_name in geom_names:
        try:
            geom_id = model.geom_name2id(geom_name)
        except Exception:
            continue
        half = np.asarray(model.geom_size[geom_id], dtype=float)
        rot = np.asarray(data.geom_xmat[geom_id], dtype=float).reshape(3, 3)
        world_half = np.abs(rot) @ half
        z0 = min(z0, float(data.geom_xpos[geom_id][2] - world_half[2]))
        z1 = max(z1, float(data.geom_xpos[geom_id][2] + world_half[2]))
        found = True
    return (z0, z1) if found else None


def _proxy_aabbs_in_band(
    raw_env, x_lo: float, x_hi: float, y_lo: float, y_hi: float,
    z_lo: float = -np.inf, z_hi: float = np.inf,
) -> list[tuple[float, float, float, float]]:
    """AABBs of scene_aabb_proxy_* geoms intersecting the given band.

    The z-range filter keeps proxies on a different level (e.g. the low
    side table the totes ride above) from constraining the route.
    """

    from robosuite.environments.factory_sorting.factory_sorting_3_3fo3errph7x9 import (
        SCENE_AABB_COLLISION_PREFIX,
    )

    model = raw_env.sim.model
    data = raw_env.sim.data
    hits = []
    for geom_id in range(model.ngeom):
        geom_name = model.geom_id2name(geom_id)
        if not geom_name or not geom_name.startswith(SCENE_AABB_COLLISION_PREFIX):
            continue
        size = model.geom_size[geom_id]
        pos = data.geom_xpos[geom_id]
        if pos[0] + size[0] < x_lo or pos[0] - size[0] > x_hi:
            continue
        if pos[1] + size[1] < y_lo or pos[1] - size[1] > y_hi:
            continue
        if pos[2] + size[2] < z_lo or pos[2] - size[2] > z_hi:
            continue
        hits.append((
            float(pos[0] - size[0]), float(pos[0] + size[0]),
            float(pos[1] - size[1]), float(pos[1] + size[1]),
        ))
    return hits


def _rack_departure_route(backend, neighbor_name, object_name, base_goal):
    """Post-pick carrying route that pulls the held tote clear of its rack.

    Pull the base backward first so the held tote slides off the rack toward
    the robot's own approach side, then travel along the rack row until the
    tote clears every other material object, then hand over to A* for the
    final approach to *base_goal*.  The corridor legs keep the current base
    yaw (holonomic drive), so the tote stays on the same side of the base
    throughout.  Pull/travel distances come from live MuJoCo geometry
    (neighbour and material collision geoms, held-tote offset, scene AABB
    proxies), so the route follows map regeneration.
    """

    from robosuite.environments.factory_sorting.load_factory_sorting_1_3fo3erfhisem_collect import (
        object_collision_geoms,
    )
    from robosuite.environments.factory_sorting.transport_attachment import (
        TRANSPORT_ATTACHMENT_ATTR,
    )
    from robot_agent.skills.pick_up import _base_robosuite_env

    current_xy, current_yaw = backend.get_base_pose()
    base0 = np.asarray(current_xy[:2], dtype=float)
    raw_env = _base_robosuite_env(getattr(backend, "env", None))
    raw_env.sim.forward()

    attachment = getattr(raw_env, TRANSPORT_ATTACHMENT_ATTR, None) or {}
    relative_xy = np.asarray(attachment.get("relative_xy", [0.0, -0.94]), dtype=float)
    cos_yaw, sin_yaw = float(np.cos(current_yaw)), float(np.sin(current_yaw))
    tote_offset = np.array([
        cos_yaw * relative_xy[0] - sin_yaw * relative_xy[1],
        sin_yaw * relative_xy[0] + cos_yaw * relative_xy[1],
    ], dtype=float)

    # Pull axis = the axis the tote sticks out along; pull direction = away
    # from the rack (backward).  Travel axis = the rack row direction,
    # toward the side of the goal.
    pull_axis = 0 if abs(tote_offset[0]) >= abs(tote_offset[1]) else 1
    pull_dir = -1.0 if tote_offset[pull_axis] > 0.0 else 1.0
    travel_axis = 1 - pull_axis
    goal = np.asarray(base_goal[:2], dtype=float)
    travel_dir = 1.0 if goal[travel_axis] >= base0[travel_axis] else -1.0

    tote_aabb = _geom_aabb(raw_env, object_collision_geoms(raw_env, object_name))
    if tote_aabb is None:
        tote_aabb = (
            base0[0] + tote_offset[0] - _L3_FALLBACK_TOTE_HALF,
            base0[0] + tote_offset[0] + _L3_FALLBACK_TOTE_HALF,
            base0[1] + tote_offset[1] - _L3_FALLBACK_TOTE_HALF,
            base0[1] + tote_offset[1] + _L3_FALLBACK_TOTE_HALF,
        )
    half = (
        (tote_aabb[1] - tote_aabb[0]) / 2.0,
        (tote_aabb[3] - tote_aabb[2]) / 2.0,
    )

    pull = _L3_FALLBACK_PULL
    travel = _L3_FALLBACK_TRAVEL
    try:
        neighbor_aabb = _geom_aabb(
            raw_env, object_collision_geoms(raw_env, neighbor_name),
        )
        if neighbor_aabb is not None:
            if pull_dir > 0.0:
                # Tote slides toward +axis: its low edge must pass the
                # neighbour's high edge.
                pull = max(
                    _L3_MIN_PULL,
                    neighbor_aabb[2 * pull_axis + 1] - tote_aabb[2 * pull_axis]
                    + _L3_CLEAR_MARGIN,
                )
            else:
                pull = max(
                    _L3_MIN_PULL,
                    tote_aabb[2 * pull_axis + 1] - neighbor_aabb[2 * pull_axis]
                    + _L3_CLEAR_MARGIN,
                )
        # Travel until the tote clears every other material object on the
        # travel side, so A* can cut back toward the rack row later without
        # sweeping anything off.
        others = [
            str(name) for name in getattr(raw_env, "material_objects", [])
            if str(name) != object_name
        ]
        edge = None
        for other in others:
            other_aabb = _geom_aabb(raw_env, object_collision_geoms(raw_env, other))
            if other_aabb is None:
                continue
            other_edge = other_aabb[2 * travel_axis + (1 if travel_dir > 0.0 else 0)]
            if edge is None:
                edge = other_edge
            elif travel_dir > 0.0:
                edge = max(edge, other_edge)
            else:
                edge = min(edge, other_edge)
        if edge is not None:
            if travel_dir > 0.0:
                # The tote's trailing edge (offset + half behind the base)
                # must clear the far edge of every other object.
                travel = max(
                    0.0,
                    edge + _L3_TRAVEL_MARGIN + half[travel_axis]
                    - tote_offset[travel_axis] - base0[travel_axis],
                )
            else:
                travel = max(
                    0.0,
                    base0[travel_axis] + tote_offset[travel_axis]
                    + half[travel_axis] - edge + _L3_TRAVEL_MARGIN,
                )
    except Exception:
        logger.exception("Rack departure geometry resolution failed; using fallback corridor")

    # Extend the legs past any static-machinery AABB the swept bands cross
    # at the tote's height (proxies on another level, e.g. the low side
    # table the totes ride above, do not constrain the route).
    tote_geoms = object_collision_geoms(raw_env, object_name)
    tote_z = _geom_z_range(raw_env, tote_geoms)
    z_lo, z_hi = tote_z if tote_z is not None else (-np.inf, np.inf)
    pull_vec = np.zeros(2)
    pull_vec[pull_axis] = pull_dir
    travel_vec = np.zeros(2)
    travel_vec[travel_axis] = travel_dir
    for _ in range(3):
        start = base0 + tote_offset
        pulled = start + pull * pull_vec
        end = pulled + travel * travel_vec
        rect_pull = _tote_band(start, pulled, half, pull_axis)
        rect_travel = _tote_band(pulled, end, half, travel_axis)
        extended = False
        for proxy in _proxy_aabbs_in_band(raw_env, *rect_pull, z_lo, z_hi):
            need = (
                (proxy[2 * pull_axis + 1] - pulled[pull_axis] + _L3_CLEAR_MARGIN)
                if pull_dir > 0.0
                else (pulled[pull_axis] - proxy[2 * pull_axis] + _L3_CLEAR_MARGIN)
            )
            if need > 0:
                pull += need
                extended = True
        for proxy in _proxy_aabbs_in_band(raw_env, *rect_travel, z_lo, z_hi):
            need = (
                (proxy[2 * travel_axis + 1] - end[travel_axis] + _L3_CLEAR_MARGIN)
                if travel_dir > 0.0
                else (end[travel_axis] - proxy[2 * travel_axis] + _L3_CLEAR_MARGIN)
            )
            if need > 0:
                travel += need
                extended = True
        # Static machinery at the tote's height that the post-handoff A*
        # route could sweep (A* only plans the base footprint) forces the
        # handoff past it along the travel axis.  The tote's perpendicular
        # position over the remaining route spans the corridor column down to
        # the goal column; any proxy in that span must lie behind the
        # handoff edge in the travel direction, otherwise a later westward
        # (travel-direction-reversal) cut would drag the tote along it —
        # e.g. the L2 line-6 belt runs x≤11.20 down to y=-7.98, so the L2
        # corridor must descend to y≈-8.7 before A* may head west.
        corr_p = end[pull_axis] + tote_offset[pull_axis]
        goal_p = goal[pull_axis] + tote_offset[pull_axis]
        span_lo = min(corr_p, goal_p) - half[pull_axis] - _L3_CLEAR_MARGIN
        span_hi = max(corr_p, goal_p) + half[pull_axis] + _L3_CLEAR_MARGIN
        if pull_axis == 0:
            span_band = (span_lo, span_hi, -np.inf, np.inf)
        else:
            span_band = (-np.inf, np.inf, span_lo, span_hi)
        # Only machinery the remaining route can actually reach along the
        # travel axis constrains the handoff — far-away stations on the same
        # perpendicular span (e.g. the line-6 input module at x≈19 in L3)
        # must not push the corridor toward them.
        t_lo = (
            min(end[travel_axis], goal[travel_axis])
            + tote_offset[travel_axis] - half[travel_axis] - _L3_CLEAR_MARGIN
        )
        t_hi = (
            max(end[travel_axis], goal[travel_axis])
            + tote_offset[travel_axis] + half[travel_axis] + _L3_CLEAR_MARGIN
        )
        for proxy in _proxy_aabbs_in_band(raw_env, *span_band, z_lo, z_hi):
            if proxy[2 * travel_axis + 1] < t_lo or proxy[2 * travel_axis] > t_hi:
                continue
            if travel_dir > 0.0:
                need = (
                    proxy[2 * travel_axis + 1] + _L3_TRAVEL_MARGIN
                    + half[travel_axis] - tote_offset[travel_axis]
                    - base0[travel_axis]
                ) - travel
            else:
                need = (
                    base0[travel_axis] + tote_offset[travel_axis]
                    + half[travel_axis] - proxy[2 * travel_axis]
                    + _L3_TRAVEL_MARGIN
                ) - travel
            if need > 0:
                travel += need
                extended = True
        if not extended:
            break

    point1 = base0 + pull * pull_vec
    point2 = point1 + travel * travel_vec
    route = [(float(point1[0]), float(point1[1]))]
    if np.linalg.norm(point2 - point1) > 1e-3:
        route.append((float(point2[0]), float(point2[1])))
    route.append((float(goal[0]), float(goal[1])))
    logger.info(
        "Rack departure route: %s (neighbor=%s, tote_offset=%s, base_yaw=%.3f)",
        route, neighbor_name, np.round(tote_offset, 3).tolist(), current_yaw,
    )
    return route


def _tote_band(a, b, half, axis):
    """(x_lo, x_hi, y_lo, y_hi) band swept by the tote moving from *a* to *b*."""

    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    return (
        float(lo[0] - half[0]), float(hi[0] + half[0]),
        float(lo[1] - half[1]), float(hi[1] + half[1]),
    )


# ── spawn north-exit waypoint ─────────────────────────────
# The spawn posture has both arms extended west at working height, and the
# direct A* diagonal from the spawn pocket to the grasp pose sweeps them
# through line 6's machine modules.  Drive straight north at the spawn
# column into the safe band between the modules (below) and the rack row
# (above), then let A* finish the approach.  All bounds come from live
# proxy/material geometry; an empty route means the north exit does not
# apply and the caller skips the staging leg entirely.
_SPAWN_ARM_NORTH = 0.47
_SPAWN_ARM_SOUTH = 0.47
_SPAWN_ARM_X_WEST = 0.66
_SPAWN_ARM_X_EAST = 0.40
_SPAWN_ROW_MARGIN_LO = 0.30
_SPAWN_ROW_MARGIN_HI = 0.20
_SPAWN_ROW_MIN_WIDTH = 0.20
_SPAWN_ARM_Z = (1.00, 1.70)


def _spawn_north_route(
    backend, goal_xy, *, scene=None, grid=None,
) -> list[tuple[float, float]]:
    """Spawn→grasp north-exit waypoints derived from live geometry.

    The westbound row must clear every same-height machine module on its
    south side (lower bound) and every material row plus base-height
    machinery on its north side (upper bound); the waypoint sits at the
    band centre with the spawn column's x unchanged.  When the grasp pose
    lies below the band, the route becomes an L: west along the row to the
    goal column, then the descent happens there — verified clear of
    machinery.  The goal-column waypoint sits at the latitude where the
    westbound leg first reaches the goal column (derived from the same A*
    the leg would run), so the leg does not climb back up to the band
    centre when machine blocks force it to a lower corridor.
    Returns an empty list when the north exit does not apply.
    """

    from robosuite.environments.factory_sorting.load_factory_sorting_1_3fo3erfhisem_collect import (
        object_collision_geoms,
    )
    from robot_agent.skills.pick_up import _base_robosuite_env

    raw_env = _base_robosuite_env(getattr(backend, "env", None))
    raw_env.sim.forward()
    current_xy, _ = backend.get_base_pose()
    spawn_x = float(current_xy[0])
    spawn_y = float(current_xy[1])

    if goal_xy is None:
        return []
    goal_x = float(goal_xy[0])
    goal_y = float(goal_xy[1])
    if goal_x >= spawn_x:
        # The goal lies east of the spawn column — the north-exit shape
        # does not apply.
        return []

    # Westbound-leg x-span plus the arm envelope.
    x_lo = goal_x - _SPAWN_ARM_X_WEST
    x_hi = spawn_x + _SPAWN_ARM_X_EAST

    # Lower bound: same-height machinery north edges the arms must clear.
    row_lo = -np.inf
    for proxy in _proxy_aabbs_in_band(
        raw_env, x_lo, x_hi, spawn_y - 0.5, max(goal_y, spawn_y) + 1.5, *_SPAWN_ARM_Z,
    ):
        north = proxy[3]
        if north < spawn_y:
            continue  # machinery behind the spawn row does not constrain
        row_lo = max(row_lo, north + _SPAWN_ARM_SOUTH + _SPAWN_ROW_MARGIN_LO)

    # Upper bound: material rows the arms must stay below, plus base-height
    # machinery the base must stay below.
    row_hi = np.inf
    for name in getattr(raw_env, "material_objects", []):
        aabb = _geom_aabb(raw_env, object_collision_geoms(raw_env, str(name)))
        if aabb is None:
            continue
        if aabb[1] < x_lo or aabb[0] > x_hi:
            continue  # not crossed by the westbound leg
        if aabb[2] < row_lo:
            continue  # below the band — irrelevant
        row_hi = min(row_hi, aabb[2] - _SPAWN_ARM_NORTH - _SPAWN_ROW_MARGIN_HI)
    for proxy in _proxy_aabbs_in_band(
        raw_env, x_lo, x_hi, row_lo, max(goal_y, spawn_y) + 1.5, 0.0, 1.0,
    ):
        south = proxy[2]
        if south > row_lo:
            # Base clearance only 0.2 here — the occupancy grid already
            # keeps the base 0.45 from these obstacles.
            row_hi = min(row_hi, south - 0.40 - 0.20)

    if not np.isfinite(row_lo):
        row_lo = goal_y - _SPAWN_ROW_MIN_WIDTH / 2.0
    if not np.isfinite(row_hi):
        row_hi = row_lo + 0.5
    if row_hi - row_lo < _SPAWN_ROW_MIN_WIDTH:
        return []
    row = (row_lo + row_hi) / 2.0

    # The north column itself must be clear for the arm envelope.
    if _proxy_aabbs_in_band(
        raw_env,
        spawn_x - _SPAWN_ARM_X_WEST, spawn_x + _SPAWN_ARM_X_EAST,
        spawn_y, row,
        *_SPAWN_ARM_Z,
    ):
        return []

    route = [(spawn_x, row)]
    if goal_y < row_lo + 0.1:
        # The goal lies below the band: travel west along the row to the
        # goal column first so the descent happens west of the machinery.
        if _proxy_aabbs_in_band(
            raw_env,
            goal_x - _SPAWN_ARM_X_WEST, goal_x + _SPAWN_ARM_X_EAST,
            goal_y, row,
            *_SPAWN_ARM_Z,
        ):
            return []
        # Machine blocks inside the band can force the westbound A* down
        # to a lower corridor; anchoring the waypoint at the band centre
        # would make the leg climb back up before the descent.  Anchor it
        # at the corridor floor instead — the lowest latitude of the same
        # A* the leg would run — so the leg ends where the corridor is and
        # the final descent starts there without a climb.
        wp_y = row
        if scene is not None and grid is not None:
            from robot_agent.core.map_loader import plan_world_path

            try:
                path = plan_world_path(
                    {"bounds": scene.bounds, "resolution": scene.resolution},
                    grid,
                    np.array([spawn_x, row], dtype=float),
                    np.array([goal_x, row], dtype=float),
                )
            except Exception:
                path = None
            if path:
                floor = min(float(p[1]) for p in path)
                wp_y = min(row, max(floor, goal_y))
        route.append((goal_x, wp_y))
    logger.info(
        "Spawn north-exit route: %s band [%.3f, %.3f] goal_y=%.3f",
        [(round(p[0], 2), round(p[1], 2)) for p in route], row_lo, row_hi, goal_y,
    )
    return route


class AnalyzeSupplySkill(BaseSkill):
    """Execute move -> pick -> move -> place, including L5 multi-object work."""

    def __init__(
        self, *, backend, scene_context: SceneContext,
        grid: np.ndarray, path_spacing: float = 0.35,
    ) -> None:
        super().__init__(
            name="analyze_supply",
            description="Execute the complete competition material transport task",
            keywords=(
                "transport", "move", "carry", "follow", "material",
                "storage", "plastic", "sop", "replenish", "supply",
            ),
        )
        self._backend = backend
        self._move = MoveSkill(
            backend=backend, scene_context=scene_context,
            grid=grid, path_spacing=path_spacing,
        )
        self._pick = PickUpSkill(backend=backend, scene_context=scene_context)
        self._place = PlaceDownSkill(backend=backend, scene_context=scene_context)

    def _step_inputs(
        self, context: ExecutionContext, target: str,
        object_name: str | None = None,
        grasp_pose: dict | None = None,
    ) -> dict:
        """复制本步 metadata 并注入目标/物体名/抓取位姿,传给子技能。"""
        metadata = dict(context.metadata)
        inputs = dict(metadata.get("inputs", {}) or {})
        inputs["target"] = target
        if object_name:
            inputs["object_name"] = object_name
        if grasp_pose:
            inputs["grasp_initial_base_pose"] = grasp_pose
        metadata["inputs"] = inputs
        return metadata

    def _run_waypoint_legs(
        self,
        context: ExecutionContext,
        points: list[tuple[float, float]],
        *,
        source: str,
        object_name: str,
        final_label: str,
        append_exact_final: bool = False,
        final_goal_yaw: float | None = None,
        final_sync_attachment: bool = False,
        final_band_half: float | None = None,
    ) -> SkillResult:
        """逐段 A* 导航通过一串 waypoint,全程物理行驶(无瞬移)。"""
        last_result = None
        leg_count = len(points)
        for index, point in enumerate(points):
            step_meta = self._step_inputs(context, source, object_name)
            step_meta["inputs"]["target"] = f"{point[0]:.6f}, {point[1]:.6f}"
            step_meta["inputs"]["allow_nearest_reachable"] = True
            step_meta["inputs"]["append_exact_goal"] = bool(
                append_exact_final and index == leg_count - 1
            )
            # 末腿边开边转:与 place_metadata 的旋转参数同源
            if index == leg_count - 1 and final_goal_yaw is not None:
                step_meta["inputs"]["goal_yaw"] = float(final_goal_yaw)
                step_meta["inputs"]["rotating_sync_attachment"] = bool(
                    final_sync_attachment,
                )
                if final_band_half is not None:
                    step_meta["inputs"]["rotating_band_half"] = float(final_band_half)
            last_result = self._move.run(ExecutionContext(
                task=f"{final_label} leg {index + 1}/{leg_count}",
                metadata=step_meta,
            ))
            if not last_result.success:
                return last_result
        if last_result is None:
            raise ValueError("A waypoint sequence must not be empty")
        return last_result

    def run(self, context: ExecutionContext) -> SkillResult:
        env_name = str(getattr(self._backend, "_env_name", ""))
        task = _task_spec(env_name)

        if task:
            source, target, object_names = task
        else:
            raw_target = context.metadata.get("inputs", {}).get("target") or context.task
            target = _resolve_station_name(raw_target, self._move._scene)
            available = self._backend.get_available_crates()
            if not available:
                return SkillResult(
                    skill_name=self.name, success=False,
                    message="No material is available", payload={"target": target},
                )
            source = self._nearest_source(available)
            object_names = [available[source]]

        steps_ok = 0
        steps_total = 0
        failures: list[str] = []
        for object_index, object_name in enumerate(object_names):
            steps_total += 1
            staging_ok = True

            planned_grasp_pose, grasp_diagnostics = _live_standoff_pose(
                self._backend, object_name,
            )

            if object_index == 0 and env_name in {
                "FactorySorting5_3FO3ERTPXEUT",
                "FactorySorting7_3FO3ERFKY9RN",
                "FactorySorting9_3FO3ERT2C5FP",
            }:
                # The spawn posture has both arms extended west at working
                # height, and the direct A* diagonal from the spawn pocket
                # to the grasp pose sweeps them through line 6's machine
                # modules.  Drive to the geometry-derived north-exit
                # waypoints first (an empty route skips the staging leg),
                # then A* continues to the live object pose.
                staging_route = _spawn_north_route(
                    self._backend,
                    planned_grasp_pose["xy"] if planned_grasp_pose is not None else None,
                    scene=self._move._scene,
                    grid=self._move._grid,
                )
                if staging_route:
                    staging = self._run_waypoint_legs(
                        context, staging_route, source=source,
                        object_name=object_name,
                        final_label="leave spawn through the safe north-exit corridor",
                    )
                    staging_ok = staging.success
                    if not staging_ok:
                        failures.append(f"{object_name}: {staging.message}")

            if env_name == "FactorySorting9_3FO3ERT2C5FP" and object_index > 0:
                # 空车回程走北侧过道中线。旧路线绕南侧横向过道,
                # 每次回程多跑约 34 米。
                # 回程腿前 12 步同时融合"抓取姿态恢复"(否则恢复在
                # 抓取开始前一次性 qpos 直设, 轨迹里表现为手臂瞬移)。
                snapshot = getattr(self._backend, "_grasp_posture_snapshot", None)
                if snapshot:
                    from robot_agent.skills.fused_retract import (
                        posture_index_from_named,
                    )

                    try:
                        self._backend._pending_retract_posture = (
                            posture_index_from_named(self._backend.env, snapshot)
                        )
                    except Exception:
                        self._backend._pending_retract_posture = None
                corridor_y, source_clear_x = _l5_aisle_waypoint(
                    self._move._scene, source, target,
                )
                return_route = [
                    (source_clear_x, corridor_y),
                    (
                        source_clear_x,
                        float(planned_grasp_pose["xy"][1])
                        if planned_grasp_pose is not None
                        else float(self._move._scene.input_ports[source].approach[1]),
                    ),
                ]
                returned = self._run_waypoint_legs(
                    context, return_route, source=source,
                    object_name=object_name,
                    final_label=f"return safely for {object_name}",
                )
                staging_ok = staging_ok and returned.success
                if not returned.success:
                    failures.append(f"{object_name}: {returned.message}")

            if planned_grasp_pose is not None:
                desired_xy = planned_grasp_pose["xy"]
                source_metadata = self._step_inputs(context, source, object_name)
                source_metadata["inputs"]["target"] = (
                    f"{float(desired_xy[0]):.6f}, {float(desired_xy[1]):.6f}"
                )
                # 末段边开边转:move 层沿网格安全腿转到抓取 yaw
                source_metadata["inputs"]["goal_yaw"] = float(
                    planned_grasp_pose["yaw"],
                )
                source_metadata["inputs"]["allow_nearest_reachable"] = True
                source_metadata["inputs"]["append_exact_goal"] = True
                move_source = self._move.run(ExecutionContext(
                    task=f"move continuously to {object_name}",
                    metadata=source_metadata,
                ))
            else:
                move_source = self._move.run(ExecutionContext(
                    task=f"move to {source}",
                    metadata=self._step_inputs(context, source, object_name),
                ))
                planned_grasp_pose = (
                    (move_source.payload or {}).get("final_base_pose")
                    if move_source.success else None
                )
                grasp_diagnostics = {"mode": "station_approach_fallback"}
            if not staging_ok:
                move_source.success = False
            steps_ok += int(move_source.success)
            if not move_source.success:
                failures.append(f"{object_name}: {move_source.message}")

            steps_total += 1
            pick = self._pick.run(ExecutionContext(
                task=f"pick {object_name} at {source}",
                metadata=self._step_inputs(
                    context, source, object_name,
                    planned_grasp_pose if move_source.success else None,
                ),
            ))
            steps_ok += int(pick.success)
            if not pick.success:
                failures.append(f"{object_name}: {pick.message}")
                continue

            steps_total += 1
            move_metadata = self._step_inputs(context, target, object_name)
            base_goal, place_xy = _place_slots(
                self._place._scene, target, object_index, len(object_names),
                _held_offset_radius(self._backend),
            )
            move_metadata["inputs"]["target"] = (
                f"{base_goal[0]:.6f}, {base_goal[1]:.6f}"
            )
            move_metadata["inputs"]["allow_nearest_reachable"] = True
            move_metadata["inputs"]["append_exact_goal"] = True
            # 放置末段的边开边转:转到面向放置点,携箱需逐步同步附着
            move_metadata["inputs"]["goal_yaw"] = float(np.arctan2(
                float(place_xy[1]) - float(base_goal[1]),
                float(place_xy[0]) - float(base_goal[0]),
            ))
            move_metadata["inputs"]["rotating_sync_attachment"] = True
            move_metadata["inputs"]["rotating_band_half"] = 1.25
            if env_name == "FactorySorting9_3FO3ERT2C5FP":
                # 先向东直拉,让箱子离开 input_1 货架,再进入底盘最短
                # 路径选中的北侧过道中线。箱子悬在底盘西侧约 0.95m,
                # 若在直拉前先改 Y,会把它扫过货架上还没搬的两个箱。
                current_base_xy, _ = self._backend.get_base_pose()
                corridor_y, source_clear_x = _l5_aisle_waypoint(
                    self._move._scene, source, target,
                )
                transport_route = [
                    (source_clear_x, float(current_base_xy[1])),
                    (source_clear_x, corridor_y),
                    (float(base_goal[0]), corridor_y),
                    (float(base_goal[0]), float(base_goal[1])),
                ]
                move_target = self._run_waypoint_legs(
                    context, transport_route, source=target,
                    object_name=object_name,
                    final_label=f"carry {object_name} safely to {target}",
                    append_exact_final=True,
                    final_goal_yaw=float(np.arctan2(
                        float(place_xy[1]) - float(base_goal[1]),
                        float(place_xy[0]) - float(base_goal[0]),
                    )),
                    final_sync_attachment=True,
                    final_band_half=1.25,
                )
            elif env_name in {
                "FactorySorting3_3FO3ERRPH7X9",
                "FactorySorting5_3FO3ERTPXEUT",
            }:
                # The base-only A* path from these racks sweeps the held
                # tote through the neighbour tote waiting on the same rack
                # (knocked green_tote_b01_lower off the shelf in both
                # 2026-08-13 L2 runs, blue_tote_b01_near_right in the L3
                # run).  Pull backward clear of the rack first, then travel
                # along the rack row — see _rack_departure_route.
                neighbor_name = _alternate_object_name(env_name, object_name)
                transport_route = _rack_departure_route(
                    self._backend, neighbor_name, object_name, base_goal,
                )
                move_target = self._run_waypoint_legs(
                    context, transport_route, source=target,
                    object_name=object_name,
                    final_label=f"carry {object_name} safely to {target}",
                    append_exact_final=True,
                    final_goal_yaw=float(np.arctan2(
                        float(place_xy[1]) - float(base_goal[1]),
                        float(place_xy[0]) - float(base_goal[0]),
                    )),
                    final_sync_attachment=True,
                    final_band_half=1.25,
                )
            else:
                move_target = self._move.run(ExecutionContext(
                    task=f"move to {target}",
                    metadata=move_metadata,
                ))
            steps_ok += int(move_target.success)

            steps_total += 1
            place_metadata = self._step_inputs(context, target, object_name)
            place_metadata["inputs"]["place_xy"] = place_xy.tolist()
            place = self._place.run(ExecutionContext(
                task=f"place {object_name} at {target}",
                metadata=place_metadata,
            ))
            steps_ok += int(place.success)
            if (move_target is not None and not move_target.success) or not place.success:
                failures.append(
                    f"{object_name}: move={move_target.success if move_target is not None else True}, "
                    f"place={place.success}"
                )

        success = not failures and steps_ok == steps_total
        return SkillResult(
            skill_name=self.name,
            success=success,
            message=(
                f"Completed {source} -> {target} ({steps_ok}/{steps_total})"
                if success else f"Transport incomplete: {'; '.join(failures)}"
            ),
            payload={
                "action": "analyze_supply", "source": source, "target": target,
                "objects": object_names, "steps_completed": steps_ok,
                "steps_total": steps_total, "failures": failures,
            },
        )

    def _nearest_source(self, available: dict[str, str]) -> str:
        """取离当前底盘位置最近的可用上料口。"""
        base_xy, _ = self._backend.get_base_pose()

        def _port_distance(port_name: str) -> float:
            try:
                centre = np.asarray(
                    self._move._scene.input_ports[port_name].center[:2],
                    dtype=float,
                )
                return float(np.linalg.norm(centre - base_xy))
            except Exception:
                return float("inf")

        return min(available, key=_port_distance)
