"""Rotating final approach: drive and turn in the same leg.

The mobile base is holonomic under direct mode, so the last approach leg
can rotate while translating.  Four guards keep the merged leg safe:

  1. leg-length check — the required rotation must fit the per-step yaw
     limit (max_angular / control_freq) over the leg length;
  2. swept-band pre-check — the start→goal band (inflated by the tote /
     arm envelope plus margin) must not intersect same-height scene
     proxy AABBs;
  3. arrival yaw tolerance — after the leg, the yaw error must be within
     the same tolerance as the in-place turn, with a bounded settle;
  4. per-step judge-collision monitor — the leg aborts immediately and
     degrades to "finish rotation in place + straight drive".

If any guard fails, :func:`try_combined_approach` returns ``None`` and
the caller falls back to its existing in-place turn.
"""

from __future__ import annotations

from typing import Any

import numpy as np

# 全局开关:某关回归异常时可整体关闭。
# 深集成版:旋转合入 A* 路径的最后一段网格安全腿(见 move.py),
# 盲尾直行照旧;到点后原地转向只剩零角度残量。
_ENABLE = True

# 末段腿参数(analyze_supply 会在终点前多停靠这段距离)
_LEAD_IN_M = 2.5
_MAX_STEP_M = 0.10        # max_linear 2.0 / 20Hz
_MAX_DYAW = 0.125         # 与官方原地转向在 turn_steps=25 档下的每步 yaw 增量对齐(25步转π)
_YAW_TOLERANCE = 0.02     # 与原地转向同一容差
_YAW_SETTLE_STEPS = 50
# band_half 传 None 表示跳过 proxy 预检:抓取腿手臂处于折叠 ready
# 姿态(包络 ~0.5m),通行安全已由官方 A* 网格的膨胀保证,proxy 盒
# 反而是粗粒度的区域框,会把合法走廊误判为障碍。
_PLACE_BAND_HALF = 1.25   # 放置前:携箱偏移 ~0.95m + 余量
# 携箱高度带:箱子运输时底部约 1.1m 以上,桌面(z≤0.9)在其下方,
# 不属于扫过障碍;高机器(0-1.9)仍会被拦截。
_BAND_Z_RANGE = (1.05, 1.9)

# 最近一次被拒绝的原因(诊断用;move.py 会写进 payload)
_last_skip_reason: str | None = None


def last_skip_reason() -> str | None:
    return _last_skip_reason


def _reject(reason: str) -> None:
    global _last_skip_reason
    _last_skip_reason = reason
    return None


def _current_pose(backend) -> tuple[Any, np.ndarray, float]:
    from robosuite.environments.factory_sorting.load_factory_sorting_evalization import (
        get_base_world_pose,
    )
    from robot_agent.skills.pick_up import _base_robosuite_env

    raw_env = _base_robosuite_env(getattr(backend, "env", None))
    xy, yaw = get_base_world_pose(raw_env, raw_env.robots[0])
    return raw_env, np.asarray(xy, dtype=float)[:2], float(yaw)


def _band_blocked(raw_env, a_xy: np.ndarray, b_xy: np.ndarray, half: float) -> bool:
    from robot_agent.skills.analyze_supply import _proxy_aabbs_in_band

    xs = [float(a_xy[0]), float(b_xy[0])]
    ys = [float(a_xy[1]), float(b_xy[1])]
    x_lo, x_hi = min(xs) - half, max(xs) + half
    y_lo, y_hi = min(ys) - half, max(ys) + half
    return bool(_proxy_aabbs_in_band(raw_env, x_lo, x_hi, y_lo, y_hi, *_BAND_Z_RANGE))


def split_rotating_tail(path, lead_m: float | None = None):
    """把路径切成三段:(旋转前段, 旋转段, 盲尾段)。

    旋转段 = 从倒数第二个路径点起、沿路径向回累计 ``lead_m`` 的
    一段网格安全腿;旋转前段 = 旋转段起点之前的全部路径(正常
    follow_path 行驶);盲尾段 = 最后一条腿(直行照旧)。路径(除盲尾)
    整体短于 lead 时,整段都是旋转段,前段为空。路径太短或末腿
    不足时返回 None,调用方整体走原流程。
    """
    if path is None or len(path) < 3:
        _reject(f"split: path too short ({0 if path is None else len(path)} pts)")
        return None
    pts = [np.asarray(p, dtype=float)[:2] for p in path]
    lead = lead_m if lead_m is not None else _LEAD_IN_M

    # 从倒数第二个点向回累计弧长,定位旋转段起点所在腿
    acc = 0.0
    i = len(pts) - 2
    while i > 0 and acc + float(np.linalg.norm(pts[i] - pts[i - 1])) < lead:
        acc += float(np.linalg.norm(pts[i] - pts[i - 1]))
        i -= 1

    if i == 0:
        # 网格路径整体都不足 lead:整段都是旋转段,无前段
        approach: list = []
        rot = list(pts[:-1])
    else:
        leg = pts[i] - pts[i - 1]
        leg_len = float(np.linalg.norm(leg))
        if leg_len < 1e-9:
            _reject("split: degenerate leg")
            return None
        remaining = lead - acc
        if remaining > 0:
            cut = pts[i] - leg * (remaining / leg_len)
            approach = list(pts[:i]) + [cut]
            rot = [cut] + list(pts[i:-1])
        else:
            approach = list(pts[:i])
            rot = list(pts[i:-1])
    if len(rot) < 2:
        _reject("split: rotation segment too short")
        return None
    rest = [pts[-2], pts[-1]]
    _last_skip_reason = None
    return approach, rot, rest



def try_combined_approach(
    backend,
    path,
    goal_yaw: float,
    *,
    band_half: float | None,
    label: str,
    sync_attachment: bool = False,
) -> dict | None:
    """沿路径段"边开边转",完整转到 goal_yaw;否则返回 None。

    路径段来自 A* 的网格安全腿(见 split_rotating_tail);每步按弧长
    插值位置,并按同样比例推进 yaw。``sync_attachment`` 为 True 时
    (放置段,携箱)每步同步运输附着。
    """
    if not _ENABLE:
        return _reject("disabled")
    from robosuite.environments.factory_sorting.turn_to_station import (
        lock_base_xy,
        set_base_world_yaw_direct,
        shortest_angle,
    )
    from robot_agent.environments.robosuite_backend import (
        _capture_upper_body_posture,
        _restore_upper_body_posture,
    )

    raw_env, start_xy, start_yaw = _current_pose(backend)
    pts = [np.asarray(p, dtype=float)[:2] for p in path]
    # 弧长表
    cum = [0.0]
    for i in range(1, len(pts)):
        cum.append(cum[-1] + float(np.linalg.norm(pts[i] - pts[i - 1])))
    distance = cum[-1]
    if distance < 0.3:
        return _reject(f"leg too short ({distance:.2f}m)")
    goal_xy = pts[-1].copy()

    turn_angle = shortest_angle(float(goal_yaw) - start_yaw)

    def _point_at(s: float) -> np.ndarray:
        for i in range(1, len(pts)):
            if s <= cum[i] or i == len(pts) - 1:
                leg = pts[i] - pts[i - 1]
                frac = 0.0 if cum[i] - cum[i - 1] <= 1e-9 else (s - cum[i - 1]) / (cum[i] - cum[i - 1])
                return pts[i - 1] + leg * min(1.0, max(0.0, frac))
        return pts[-1].copy()

    # 携箱时读实际附着偏移(箱子在底盘系里的位置),用于方向感知扫过检查
    tote_rel = None
    if sync_attachment:
        from robosuite.environments.factory_sorting.transport_attachment import (
            TRANSPORT_ATTACHMENT_ATTR,
            sync_transport_attachment,
        )
        attachment = getattr(raw_env, TRANSPORT_ATTACHMENT_ATTR, None) or {}
        rel = np.asarray(attachment.get("relative_xy", []), dtype=float)
        if rel.size == 2:
            tote_rel = rel

    def _yaw_profile(alpha: float, direction: float) -> float:
        """固定角速率剖面:每步最多转 _MAX_DYAW,转完即保持直行。"""
        steps_done = max(0.0, alpha * steps)
        merged = float(np.copysign(min(abs(direction), _MAX_DYAW * steps_done), direction))
        return start_yaw + merged

    def _tote_sweep_blocked(direction: float) -> bool:
        """方向感知扫过检查:按固定角速率剖面采样箱子的实际世界
        轨迹,任一采样箱盒碰到同高度代理盒即认为被挡。"""
        from robot_agent.skills.analyze_supply import _proxy_aabbs_in_band

        for k in range(12):
            s = distance * k / 11.0
            base = _point_at(s)
            yaw = _yaw_profile(k / 11.0, direction)
            c, sn = np.cos(yaw), np.sin(yaw)
            tote = base + np.array([
                c * tote_rel[0] - sn * tote_rel[1],
                sn * tote_rel[0] + c * tote_rel[1],
            ])
            box = (tote[0] - 0.5, tote[0] + 0.5, tote[1] - 0.5, tote[1] + 0.5)
            if _proxy_aabbs_in_band(raw_env, *box, *_BAND_Z_RANGE):
                return True
        return False

    # 保护1(部分旋转版):腿长能摊多少就转多少,残量留给调用方的
    # 原地转向兜底;转角过小不值得。
    steps = max(6, int(round(distance / _MAX_STEP_M)))
    capacity = _MAX_DYAW * steps

    # 保护2(方向感知版):旋转方向决定扫过区。携箱时逐一尝试候选
    # 方向(最短方向优先,其次反向长路),取第一个扫过检查通过的;
    # 全部被挡则拒绝。空载(band_half=None 且无附着)不检查。
    chosen = None
    if tote_rel is not None:
        candidates = [turn_angle]
        other = turn_angle - (2.0 * np.pi if turn_angle >= 0 else -2.0 * np.pi)
        if abs(other - turn_angle) > 1e-6:
            candidates.append(other)
        # 每个方向找"扫过检查能通过的最大转角"(从容量往下扫),
        # 取两个方向里合并量最大的;残量照旧留给到点原地转。
        max_steps_avail = min(steps, int(abs(turn_angle) / _MAX_DYAW) + 1)
        best = None
        for direction in candidates:
            for n in range(max_steps_avail, 0, -1):
                rotation = float(np.copysign(_MAX_DYAW * n, direction))
                if abs(rotation) < 0.05:
                    break
                if not _tote_sweep_blocked(rotation):
                    if best is None or abs(rotation) > abs(best):
                        best = rotation
                    break
        if best is None:
            return _reject("no safe rotation direction for carried tote")
        chosen = best
    else:
        rotation = float(turn_angle)
        if abs(rotation) > capacity:
            rotation = float(np.copysign(capacity, rotation))
        if abs(rotation) < 0.05:
            return _reject(f"rotation too small ({abs(rotation):.3f}rad)")
        chosen = rotation
    rotation = chosen

    idle_action = np.zeros_like(raw_env.action_spec[0])
    robot = raw_env.robots[0]
    posture = _capture_upper_body_posture(raw_env, robot)
    aborted = False

    # 合并腿:平移+旋转同帧推进;每步按官方 turn_to_face_xy 的
    # 步进范式(直设位姿 → lock_base_xy → forward → step → 再锁)
    # 执行,避免 qpos 直设激起速度尖峰。保护4:逐帧碰撞监控。
    for index in range(1, steps + 1):
        alpha = index / float(steps)
        step_xy = _point_at(distance * alpha)
        step_yaw = _yaw_profile(alpha, rotation)
        set_base_world_yaw_direct(raw_env, robot, step_yaw, tolerance=1e-5)
        lock_base_xy(raw_env, robot, step_xy)
        _restore_upper_body_posture(raw_env, posture)
        if sync_attachment:
            sync_transport_attachment(raw_env)
        raw_env.sim.forward()
        raw_env.step(idle_action)
        lock_base_xy(raw_env, robot, step_xy)
        _restore_upper_body_posture(raw_env, posture)
        if sync_attachment:
            sync_transport_attachment(raw_env)
        if hasattr(backend, "_record_trajectory_frame"):
            backend._record_trajectory_frame()
        if getattr(raw_env, "has_judge_collision", False):
            aborted = True
            break

    # 保护3:到点 yaw 兜底(容差与原地转向一致)。渐进小步进,
    # 避免一次性 snap 在轨迹里留下单帧大角度跳变。
    _, final_xy, final_yaw = _current_pose(backend)
    yaw_err = shortest_angle(float(goal_yaw) - final_yaw)
    for _ in range(_YAW_SETTLE_STEPS):
        if abs(yaw_err) <= _YAW_TOLERANCE:
            break
        step = float(np.copysign(min(abs(yaw_err), _MAX_DYAW), yaw_err))
        set_base_world_yaw_direct(raw_env, robot, final_yaw + step, tolerance=1e-5)
        lock_base_xy(raw_env, robot, final_xy)
        _restore_upper_body_posture(raw_env, posture)
        if sync_attachment:
            sync_transport_attachment(raw_env)
        raw_env.sim.forward()
        raw_env.step(idle_action)
        lock_base_xy(raw_env, robot, final_xy)
        _restore_upper_body_posture(raw_env, posture)
        if sync_attachment:
            sync_transport_attachment(raw_env)
        if hasattr(backend, "_record_trajectory_frame"):
            backend._record_trajectory_frame()
        _, final_xy, final_yaw = _current_pose(backend)
        yaw_err = shortest_angle(float(goal_yaw) - final_yaw)

    # 位置兜底:若中途中止,直线补完剩余距离(退化为旧流程后半段)
    remaining = float(np.linalg.norm(goal_xy - final_xy))
    if remaining > 1e-3:
        for _ in range(max(6, int(remaining / _MAX_STEP_M))):
            _, cur_xy, _ = _current_pose(backend)
            dist_left = float(np.linalg.norm(goal_xy - cur_xy))
            if dist_left <= 1e-3:
                break
            step_to = cur_xy + (goal_xy - cur_xy) / dist_left * min(dist_left, _MAX_STEP_M)
            lock_base_xy(raw_env, robot, step_to)
            _restore_upper_body_posture(raw_env, posture)
            if sync_attachment:
                sync_transport_attachment(raw_env)
            raw_env.sim.forward()
            raw_env.step(idle_action)
            lock_base_xy(raw_env, robot, step_to)
            _restore_upper_body_posture(raw_env, posture)
            if sync_attachment:
                sync_transport_attachment(raw_env)
            _restore_upper_body_posture(raw_env, posture)
            if hasattr(backend, "_record_trajectory_frame"):
                backend._record_trajectory_frame()

    _, final_xy, final_yaw = _current_pose(backend)
    return {
        "success": True,
        "method": "combined_approach",
        "label": label,
        "steps": steps,
        "aborted_on_collision": aborted,
        "final_error": yaw_err,
        "xy_drift": float(np.linalg.norm(goal_xy - final_xy)),
        "turn_angle": float(turn_angle),
        "merged_angle": float(rotation),
        "residual_angle": float(turn_angle - rotation),
    }
