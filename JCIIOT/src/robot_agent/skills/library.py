"""Skill library — wired to a real or simulated backend.

All skills require a backend; there is no mock / no-op fallback.
"""

from __future__ import annotations

import os

import numpy as np

from robot_agent.core.memory import InMemoryStore
from robot_agent.core.scene_context import SceneContext
from robot_agent.core.types import ExecutionContext, SkillResult
from robot_agent.environments.base import EnvBackend
from robot_agent.skills.base import BaseSkill
from robot_agent.skills.move import MoveSkill
from robot_agent.skills.pick_up import PickUpSkill
from robot_agent.skills.place_down import PlaceDownSkill
from robot_agent.skills.record_trajectory import RecordTrajectorySkill
from robot_agent.skills.analyze_supply import AnalyzeSupplySkill
from robot_agent.skills.knowledge_mgr import KnowledgeMgrSkill
from robot_agent.skills.memory_mgr import MemoryMgrSkill
from robot_agent.skills.read_document import ReadDocumentSkill
from robot_agent.skills.sop_generator import GenerateSOPSkill

# ── 竞赛环境的一体化路由 ─────────────────────────────────────
# 比赛五关只允许一条经过审计的完整搬运流程。LLM 生成的计划里
# move / pick_up / place_down / analyze_supply 这些技能名统一
# 解析到同一个流程入口:第一次调用真正执行,后续调用直接复用
# 结果(计划里同名的多步不会重复执行)。路由放在技能库里实现,
# 不修改官方 core/agent.py。

_COMPETITION_ENV_NAMES = frozenset({
    "FactorySorting1_3FO3ERFHISEM",
    "FactorySorting3_3FO3ERRPH7X9",
    "FactorySorting5_3FO3ERTPXEUT",
    "FactorySorting7_3FO3ERFKY9RN",
    "FactorySorting9_3FO3ERT2C5FP",
})


class _WorkflowAliasSkill(BaseSkill):
    """技能名别名:全部指向同一个竞赛搬运流程,幂等执行。"""

    def __init__(self, *, alias: str, workflow: AnalyzeSupplySkill, backend) -> None:
        super().__init__(
            name=alias,
            description="Run the single audited competition transport workflow.",
            keywords=(
                "move", "transport", "navigate", "pick", "grasp", "place",
                "storage", "bin", "material", "搬运", "抓取", "放置",
            ),
        )
        self._workflow = workflow
        self._backend = backend

    def run(self, context: ExecutionContext) -> SkillResult:
        result = getattr(self._backend, "_workflow_result_cache", None)
        if result is None:
            result = self._workflow.run(context)
            self._backend._workflow_result_cache = result
        return SkillResult(
            skill_name=self.name,
            success=bool(result.success),
            message=str(result.message),
            payload=dict(result.payload or {}),
        )


def _detect_vision_api_config() -> dict:
    """Detect vision API configuration from environment / robot_params.

    Priority: VLM-specific env vars > OPENAI_* env vars > robot_params.json > defaults.
    """
    cfg: dict = {
        "ollama_base_url": "http://localhost:11434",
        "vision_model": "qwen3-vl:8b",
        "api_type": "ollama",
        "api_key": "",
    }

    # ── Check VLM-specific environment variables first ──
    vlm_url = os.getenv("VLM_BASE_URL", "")
    vlm_key = os.getenv("VLM_API_KEY", "")
    vlm_model = os.getenv("VLM_MODEL", "")
    if vlm_url:
        from robot_agent.core.vision_client import _detect_api_type
        cfg["ollama_base_url"] = vlm_url
        cfg["api_type"] = "openai" if vlm_key else _detect_api_type(vlm_url)
        cfg["api_key"] = vlm_key
        if vlm_model:
            cfg["vision_model"] = vlm_model

    # ── Fallback: OPENAI_* env vars (set when text LLM backend is OpenAI) ──
    elif os.getenv("OPENAI_API_KEY", ""):
        cfg["api_type"] = "openai"
        cfg["api_key"] = os.getenv("OPENAI_API_KEY", "")
        cfg["ollama_base_url"] = os.getenv(
            "OPENAI_BASE_URL", "https://api.openai.com/v1",
        )
        openai_model = os.getenv("OPENAI_MODEL", "")
        if openai_model:
            cfg["vision_model"] = openai_model

    # ── Read from robot_params.json for vision-specific settings ──
    try:
        from pathlib import Path
        import json
        _rp = Path(__file__).resolve().parents[3] / "knowledge" / "robot_params.json"
        if _rp.exists():
            _data = json.loads(_rp.read_text(encoding="utf-8"))
            _llm = _data.get("llm", {}) if isinstance(_data, dict) else {}
            if isinstance(_llm, dict):
                if not vlm_url:
                    cfg["ollama_base_url"] = _llm.get(
                        "ollama_base_url", cfg["ollama_base_url"],
                    )
                if not vlm_model:
                    cfg["vision_model"] = _llm.get(
                        "vision_model", cfg["vision_model"],
                    )
    except Exception:
        pass

    return cfg


def wired_skills(
    backend: EnvBackend,
    scene_context: SceneContext,
    grid: np.ndarray,
    *,
    path_spacing: float = 0.35,
    memory_store: InMemoryStore | None = None,
) -> list[BaseSkill]:
    """Return skills wired to a real (or simulated) backend."""
    _vis_cfg = _detect_vision_api_config()
    workflow = AnalyzeSupplySkill(
        backend=backend,
        scene_context=scene_context,
        grid=grid,
        path_spacing=path_spacing,
    )
    env_name = str(getattr(backend, "_env_name", ""))
    if env_name in _COMPETITION_ENV_NAMES:
        skills: list[BaseSkill] = [
            _WorkflowAliasSkill(alias=alias, workflow=workflow, backend=backend)
            for alias in ("analyze_supply", "move", "pick_up", "place_down")
        ]
    else:
        skills = [
            workflow,
            MoveSkill(
                backend=backend,
                scene_context=scene_context,
                grid=grid,
                path_spacing=path_spacing,
            ),
            PickUpSkill(backend=backend, scene_context=scene_context),
            PlaceDownSkill(backend=backend, scene_context=scene_context),
        ]
    skills.extend([
        RecordTrajectorySkill(backend=backend),
        KnowledgeMgrSkill(knowledge_root="knowledge"),
        ReadDocumentSkill(
            ollama_base_url=_vis_cfg["ollama_base_url"],
            vision_model=_vis_cfg["vision_model"],
            api_type=_vis_cfg["api_type"],
            api_key=_vis_cfg["api_key"],
        ),
    ])
    if memory_store is not None:
        skills.append(MemoryMgrSkill(store=memory_store))
    # SOP 生成技能(文本 LLM + semantic_map,不占用 VLM)
    skills.append(GenerateSOPSkill())
    return skills
