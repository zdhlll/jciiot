#!/usr/bin/env python3
"""JCIIOT2026 最终提交自检脚本（团队自建，非官方）。

校验四类事项：
1. 五关证据齐全：evidence/L1~L5/ 下 score.json / result.json / trajectory.json 三个文件必须存在。
2. 证据为满分：score.json 的 status == "OK" 且 score 等于各关满分。
3. 轨迹 schema：trajectory.json 顶层键与官方 trajectory_template.json 一致。
4. 受保护边界审计：与官方基线提交相比，禁区（core/、environments/、app.py、
   task_config.json、robosuite/）逐文件一致；全部改动落在允许区域内。

用法：
    python team_submission/audits/verify_final_submission.py [--baseline <commit>]
    默认基线为官方内容基线 0dcdddf（团队 fork 与官方仓库树逐字节一致时的提交）。
    运行后会把边界审计结果写入同目录 official_boundary_audit.json / .md。
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[3]
JCIIOT = REPO_ROOT / "JCIIOT"
EVIDENCE = JCIIOT / "team_submission" / "evidence"
AUDITS = JCIIOT / "team_submission" / "audits"
TEMPLATE = (
    REPO_ROOT / "competition description" / "trajectory_template.json"
)

SCORES = {"L1": 10, "L2": 15, "L3": 20, "L4": 25, "L5": 30}
ENVS = {
    "L1": "FactorySorting1_3FO3ERFHISEM",
    "L2": "FactorySorting3_3FO3ERRPH7X9",
    "L3": "FactorySorting5_3FO3ERTPXEUT",
    "L4": "FactorySorting7_3FO3ERFKY9RN",
    "L5": "FactorySorting9_3FO3ERT2C5FP",
}

# 官方 Contestant Manual「Permission to Modify Code and Configuration」原文：
#   允许修改：src/robot_agent/skills/、src/robot_agent/workflows/、knowledge/robot_params.json
#   禁止修改：src/robot_agent/core/、src/robot_agent/environments/、app.py、knowledge/task_config.json
# 另外按主办方培训要求，robosuite/（含官方 LFS 模型与场景）也保持未修改。
PROTECTED_PATHS = [
    "JCIIOT/app.py",
    "JCIIOT/knowledge/task_config.json",
    "JCIIOT/src/robot_agent/core",
    "JCIIOT/src/robot_agent/environments",
    "JCIIOT/robosuite",
]

# 团队改动允许落入的前缀（除禁区与官方 SOP 生成要求外）。
ALLOWED_PREFIXES = (
    "JCIIOT/src/robot_agent/skills/",
    "JCIIOT/src/robot_agent/workflows/",
    "JCIIOT/knowledge/",           # 官方 SOP 覆盖属竞赛明确要求（见 Contestant Manual）
    "JCIIOT/team_submission/",
    "JCIIOT/sop_autogen_tool/",
    "JCIIOT/.gitignore",
    ".gitignore",
)

DEFAULT_BASELINE = "0dcdddf18a9e694569aa1433cdfc04eb097fed78"

SCHEMA_KEYS = [
    "robot_model", "camera", "units", "joint_names",
    "object_names", "object_joints", "events", "frames",
]


def git(*args: str) -> str:
    out = subprocess.run(
        ["git", "-c", "core.quotepath=false", "-C", str(REPO_ROOT), *args],
        check=True, capture_output=True, text=True,
    )
    return out.stdout


def git_ls_tree(rev: str, paths: list[str]) -> dict[str, str]:
    """返回 {path: object_sha}（blob 与 subtree 统一视图）。"""
    entries: dict[str, str] = {}
    for line in git("ls-tree", "-r", rev, "--", *paths).splitlines():
        _, _, sha = line.split("\t")[0].split(" ")
        entries[line.split("\t")[1]] = sha
    return entries


def check_evidence() -> tuple[bool, list[str]]:
    ok, msgs = True, []
    for level, full in SCORES.items():
        d = EVIDENCE / level
        for fname in ("trajectory.json", "score.json", "result.json"):
            if not (d / fname).is_file():
                ok = False
                msgs.append(f"[FAIL] {level}/{fname} 缺失")
        if not ok:
            continue
        score = json.loads((d / "score.json").read_text(encoding="utf-8"))
        result = json.loads((d / "result.json").read_text(encoding="utf-8"))
        traj = json.loads((d / "trajectory.json").read_text(encoding="utf-8"))
        status_ok = score.get("status") == "OK" and score.get("score") == full
        if not status_ok:
            ok = False
            msgs.append(
                f"[FAIL] {level} 得分 {score.get('score')}/{full} status={score.get('status')}"
            )
        else:
            msgs.append(
                f"[OK] {level} {score.get('score')}/{full}  env={score.get('env_name')} "
                f"elapsed={round(score.get('elapsed_sec', 0), 1)}s"
            )
        if score.get("env_name") != ENVS[level]:
            ok = False
            msgs.append(f"[FAIL] {level} env 名不匹配: {score.get('env_name')}")
        if result.get("success") is not True:
            ok = False
            msgs.append(f"[FAIL] {level} result.success != true")
        missing = [k for k in SCHEMA_KEYS if k not in traj]
        if missing:
            ok = False
            msgs.append(f"[FAIL] {level} trajectory 缺少顶层键: {missing}")
        elif not traj.get("frames"):
            ok = False
            msgs.append(f"[FAIL] {level} trajectory frames 为空")
        else:
            msgs.append(f"[OK] {level} trajectory schema 齐全, {len(traj['frames'])} 帧")
    return ok, msgs


def check_boundary(baseline: str) -> tuple[bool, dict]:
    head = git("rev-parse", "HEAD").strip()
    protected_base = git_ls_tree(baseline, PROTECTED_PATHS)
    protected_head = git_ls_tree(head, PROTECTED_PATHS)

    modified = [p for p in protected_base if protected_base[p] != protected_head.get(p)]
    added = [p for p in protected_head if p not in protected_base]
    missing = [p for p in protected_base if p not in protected_head]

    changed = set(git("diff", "--name-only", baseline, head).splitlines())
    root_allowed = {
        "README.md", "技术报告.md", "新颖性声明.md", "复现指南.md",
        "提交合规说明.md", "ERRATUM.md", ".gitattributes", ".gitignore",
    }
    violations = [
        p for p in sorted(changed)
        if not p.startswith(ALLOWED_PREFIXES) and p not in root_allowed
    ]

    gen_code = [
        "JCIIOT/src/robot_agent/skills/sop_generator.py",
        "JCIIOT/sop_autogen_tool/generate_sops.py",
    ]
    missing_gen = [p for p in gen_code if not (REPO_ROOT / p).is_file()]

    audit = {
        "baseline_commit": baseline,
        "head_commit": head,
        "protected_files_checked": len(protected_base),
        "protected_modified": modified,
        "protected_added": added,
        "protected_missing": missing,
        "changed_paths_total": len(changed),
        "changed_paths_violating_allowed_prefixes": violations,
        "sop_generation_code_present": not missing_gen,
        "missing_sop_generation_code": missing_gen,
        "summary": {
            "compliant": not (modified or added or missing or violations or missing_gen),
        },
    }
    return audit["summary"]["compliant"], audit


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline", default=DEFAULT_BASELINE)
    args = parser.parse_args()

    ev_ok, ev_msgs = check_evidence()
    bd_ok, audit = check_boundary(args.baseline)

    for m in ev_msgs:
        print(m)
    print(f"[{'OK' if bd_ok else 'FAIL'}] 受保护边界: 检查 {audit['protected_files_checked']} 个文件, "
          f"修改 {len(audit['protected_modified'])} / 新增 {len(audit['protected_added'])} / 缺失 {len(audit['protected_missing'])}")
    if audit["changed_paths_violating_allowed_prefixes"]:
        print("[FAIL] 越界改动路径:")
        for p in audit["changed_paths_violating_allowed_prefixes"]:
            print(f"       {p}")
    print(f"[{'OK' if audit['sop_generation_code_present'] else 'FAIL'}] SOP 生成代码保留: "
          f"{'sop_generator.py + generate_sops.py 均在仓库' if audit['sop_generation_code_present'] else audit['missing_sop_generation_code']}")

    passed = ev_ok and bd_ok
    if passed:
        total = sum(SCORES.values())
        print(f"\nFinal submission verification: PASSED")
        print(f"Full-score evidence: 5/5 | Total: {total}/{total} | Boundary: clean")
    else:
        print("\nFinal submission verification: FAILED")
        return 1

    (AUDITS / "official_boundary_audit.json").write_text(
        json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    lines = [
        "# 受保护边界审计报告",
        "",
        f"- 基线提交（官方内容基线）: `{audit['baseline_commit']}`",
        f"- 提交版本: `{audit['head_commit']}`",
        f"- 受保护文件检查数: {audit['protected_files_checked']}",
        f"- 修改: {len(audit['protected_modified'])}，新增: {len(audit['protected_added'])}，缺失: {len(audit['protected_missing'])}",
        f"- 越界改动: {len(audit['changed_paths_violating_allowed_prefixes'])}",
        f"- SOP 生成代码保留: {'是' if audit['sop_generation_code_present'] else '否'}",
        f"- 结论: {'合规' if audit['summary']['compliant'] else '违规'}",
        "",
        "受保护区域：`src/robot_agent/core/`、`src/robot_agent/environments/`、`app.py`、`knowledge/task_config.json`、`robosuite/`。",
        "本报告由 verify_final_submission.py 自动生成。",
    ]
    (AUDITS / "official_boundary_audit.md").write_text("\n".join(lines), encoding="utf-8")
    print(f"\n审计报告已写入 {AUDITS / 'official_boundary_audit.json'} / .md")
    return 0


if __name__ == "__main__":
    sys.exit(main())
