# JCIIOT 2026 工业具身智能挑战赛 — 团队最终提交



## 成绩总览

五关全部满分，**总分 100 / 100**（运行时间 2026-08-15，逐帧轨迹可回放验证）：

| 关卡 | 场景 | 满分 | 得分 | 用时 |
| ---- | ---- | ---- | ---- | ---- |
| L1 | FactorySorting1 | 10 | **10** | 74.6 s |
| L2 | FactorySorting3 | 15 | **15** | 73.0 s |
| L3 | FactorySorting5 | 20 | **20** | 80.4 s |
| L4 | FactorySorting7 | 25 | **25** | 111.3 s |
| L5 | FactorySorting9 | 30 | **30** | 257.1 s |
| **合计** | | **100** | **100** | 596.4 s |

评分由官方 `app.py` 内置 `_score_steps()` 依据执行轨迹 JSON 自动计算，全程无碰撞扣分。

## 评审版本

- 评审锁定标签：**`submission-final-100`**（`git checkout submission-final-100` 即可复现本提交）
- 提交基于官方内容基线 `0dcdddf`（与官方仓库逐字节一致），团队全部改动集中在基线之上的提交里（含自检审计报告），可用 `git diff 0dcdddf..HEAD` 直接审计。

## 提交文档

| 文档 | 内容 |
| ---- | ---- |
| [技术报告.md](./技术报告.md) | Technology Description / Results & Analysis / Compliance / Limitations |
| [新颖性声明.md](./新颖性声明.md) | Novelty Statement（创新点与定量证据） |
| [复现指南.md](./复现指南.md) | 环境、依赖、安装、运行与验证步骤 |
| [提交合规说明.md](./提交合规说明.md) | 官方规则逐条对照与受保护边界审计报告 |

## 仓库结构

```
├── JCIIOT/                          # 官方项目（可复现代码）
│   ├── app.py                       # 官方入口（未修改）
│   ├── src/robot_agent/
│   │   ├── skills/                  # 团队修改区（唯一执行逻辑）
│   │   │   ├── analyze_supply.py    #   确定性端到端搬运工作流
│   │   │   ├── move.py              #   A* 导航与回退策略
│   │   │   ├── pick_up.py           #   阶段表脚本抓取（无 BC 模型）
│   │   │   ├── place_down.py        #   物理放置与扩展重力释放
│   │   │   ├── library.py           #   竞赛环境技能别名路由
│   │   │   └── sop_generator.py     #   运行态 SOP 生成技能
│   │   └── core/ environments/      # 官方锁定区（未修改）
│   ├── knowledge/                   # 任务配置（task_config.json 未修改）+ 团队生成 SOP
│   └── team_submission/
│       ├── evidence/L1~L5/          # 五关满分证据：trajectory / score / result
│       ├── audits/                  # 提交自检脚本与边界审计报告
│       ├── knowledge/               # 团队策略注入
│       └── sop_autogen_tool/        # SOP 生成器（官方 docx → md，供评委审查）
├── competition description/         # 官方赛题文档与轨迹模板（未修改）
└── 提交文档（本文件与上表 4 份）
```




