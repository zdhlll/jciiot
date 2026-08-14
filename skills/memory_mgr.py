"""Memory-manager skill — the agent's interface to its own experience store.

This skill lets the robot agent search, recall, summarize, and manage its
runtime memory (the history of executed steps).  Useful for self-reflection,
debugging, and learning from past failures.

Agent-facing actions (set via ``inputs.action``):

* ``recall`` — fuzzy search past steps by query
* ``recent`` — get the N most recent steps
* ``summarize`` — natural-language summary of recent activity
* ``failures`` — list only failed steps
* ``by_skill`` — filter steps by skill name
* ``stats`` — memory usage statistics
* ``clear`` — wipe all memory
* ``forget`` — remove steps matching a query
"""

from __future__ import annotations

import logging

from robot_agent.core.memory import InMemoryStore
from robot_agent.core.types import ExecutionContext, SkillResult
from robot_agent.skills.base import BaseSkill

logger = logging.getLogger(__name__)


class MemoryMgrSkill(BaseSkill):
    """Agent skill for introspecting and managing its own memory."""

    def __init__(self, store: InMemoryStore) -> None:
        super().__init__(
            name="memory_mgr",
            description="Memory management: search history, view recent ops, summarize, clear memory",
            keywords=(
                "memory", "history", "recall", "retrieve",
                "history", "ops", "summarize", "summary",
                "clear", "forget", "clean",
            ),
        )
        self._store = store

    def run(self, context: ExecutionContext) -> SkillResult:
        inputs: dict = context.metadata.get("inputs", {})
        action = inputs.get("action", "summarize")

        try:
            if action == "recall":
                return self._do_recall(inputs)
            elif action == "recent":
                return self._do_recent(inputs)
            elif action == "summarize":
                return self._do_summarize()
            elif action == "failures":
                return self._do_failures()
            elif action == "by_skill":
                return self._do_by_skill(inputs)
            elif action == "stats":
                return self._do_stats()
            elif action == "clear":
                return self._do_clear()
            elif action == "forget":
                return self._do_forget(inputs)
            else:
                return SkillResult(
                    skill_name=self.name,
                    success=False,
                    message=f"Unknown memory_mgr action: '{action}'. Supported: recall, recent, summarize, failures, by_skill, stats, clear, forget",
                )
        except Exception as exc:
            logger.exception("memory_mgr action '%s' failed", action)
            return SkillResult(
                skill_name=self.name,
                success=False,
                message=f"Memory operation failed ({action}): {exc}",
            )

    # ── action handlers ────────────────────────────────────

    def _do_recall(self, inputs: dict) -> SkillResult:
        query = str(inputs.get("query") or inputs.get("q") or "")
        if not query:
            return SkillResult(skill_name=self.name, success=False, message="recall requiresquery 参数")
        top_n = int(inputs.get("top_n", 10))
        results = self._store.recall(query, top_n=top_n)
        return SkillResult(
            skill_name=self.name,
            success=True,
            message=f"Recall '{query}' — found {len(results)} related records",
            payload={"query": query, "count": len(results), "results": results},
        )

    def _do_recent(self, inputs: dict) -> SkillResult:
        n = int(inputs.get("n", 5))
        results = self._store.recent(n=n)
        return SkillResult(
            skill_name=self.name,
            success=True,
            message=f"Last {len(results)} operation records",
            payload={"count": len(results), "results": results},
        )

    def _do_summarize(self) -> SkillResult:
        summary = self._store.summarize()
        return SkillResult(
            skill_name=self.name,
            success=True,
            message=summary,
            payload={"summary": summary},
        )

    def _do_failures(self) -> SkillResult:
        results = self._store.failures()
        return SkillResult(
            skill_name=self.name,
            success=True,
            message=f"{len(results)} failure records" if results else "No failure records",
            payload={"count": len(results), "results": results},
        )

    def _do_by_skill(self, inputs: dict) -> SkillResult:
        skill_name = str(inputs.get("skill_name") or inputs.get("skill") or "")
        if not skill_name:
            return SkillResult(skill_name=self.name, success=False, message="by_skill requiresskill_name 参数")
        results = self._store.by_skill(skill_name)
        return SkillResult(
            skill_name=self.name,
            success=True,
            message=f"Skill '{skill_name}' executed {len(results)} times",
            payload={"skill_name": skill_name, "count": len(results), "results": results},
        )

    def _do_stats(self) -> SkillResult:
        stats = self._store.stats()
        s = stats
        return SkillResult(
            skill_name=self.name,
            success=True,
            message=f"Memory stats: {s['total_steps']}/{s['limit']} records (OK {s['success_count']}, FAIL {s['fail_count']})",
            payload=stats,
        )

    def _do_clear(self) -> SkillResult:
        removed = self._store.clear()
        return SkillResult(
            skill_name=self.name,
            success=True,
            message=f"Memory cleared: {removed} records removed",
            payload={"removed": removed},
        )

    def _do_forget(self, inputs: dict) -> SkillResult:
        query = str(inputs.get("query") or inputs.get("q") or "")
        if not query:
            return SkillResult(skill_name=self.name, success=False, message="forget requiresquery 参数")
        removed = self._store.forget(query)
        return SkillResult(
            skill_name=self.name,
            success=True,
            message=f"Forgotten {removed} records matching '{query}'",
            payload={"query": query, "removed": removed},
        )
