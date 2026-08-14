"""Knowledge-base manager skill — the agent's interface to its document store.

This skill lets the robot agent query, search, and manage its own knowledge
base at runtime.  The knowledge base is backed by ``KnowledgeManager`` which
handles deduplication, classification, and persistence.

Agent-facing actions (set via ``inputs.action``):

* ``search`` — fuzzy search documents
* ``list`` — list all or filtered documents
* ``add`` — add a new document
* ``remove`` — remove a document by ID
* ``stats`` — show knowledge base statistics
* ``context`` — get prompt-ready context for the planner
* ``reload`` — rescan the knowledge folder
"""

from __future__ import annotations

import logging
from pathlib import Path

from robot_agent.core.knowledge_manager import KnowledgeManager
from robot_agent.core.types import ExecutionContext, SkillResult
from robot_agent.skills.base import BaseSkill

logger = logging.getLogger(__name__)


class KnowledgeMgrSkill(BaseSkill):
    """Agent skill for interacting with its knowledge base."""

    def __init__(self, *, knowledge_root: str | Path = "knowledge") -> None:
        super().__init__(
            name="knowledge_mgr",
            description="Knowledge base management: search, list, add, remove docs, view stats",
            keywords=(
                "knowledge", "kb", "docs", "document", "search",
                "find", "doc", "document", "kb", "lookup",
            ),
        )
        self._mgr = KnowledgeManager(knowledge_root)
        # Auto-load on first use
        self._loaded = False

    def _ensure_loaded(self) -> None:
        if not self._loaded:
            self._mgr.reload()
            self._loaded = True

    def run(self, context: ExecutionContext) -> SkillResult:
        self._ensure_loaded()

        inputs: dict = context.metadata.get("inputs", {})
        action = inputs.get("action", "context")

        try:
            if action == "search":
                return self._do_search(inputs)
            elif action == "list":
                return self._do_list(inputs)
            elif action == "add":
                return self._do_add(inputs)
            elif action == "remove":
                return self._do_remove(inputs)
            elif action == "stats":
                return self._do_stats()
            elif action == "context":
                return self._do_context(inputs)
            elif action == "reload":
                return self._do_reload()
            else:
                return SkillResult(
                    skill_name=self.name,
                    success=False,
                    message=f"Unknown knowledge_mgr action: '{action}'. Supported: search, list, add, remove, stats, context, reload",
                )
        except Exception as exc:
            logger.exception("knowledge_mgr action '%s' failed", action)
            return SkillResult(
                skill_name=self.name,
                success=False,
                message=f"Knowledge base operation failed ({action}): {exc}",
            )

    # ── action handlers ────────────────────────────────────

    def _do_search(self, inputs: dict) -> SkillResult:
        query = str(inputs.get("query") or inputs.get("q") or "")
        if not query:
            return SkillResult(skill_name=self.name, success=False, message="search requiresquery 参数")
        top_n = int(inputs.get("top_n", 10))
        results = self._mgr.search(query, top_n=top_n)
        return SkillResult(
            skill_name=self.name,
            success=True,
            message=f"Search '{query}' found {len(results)} results",
            payload={"query": query, "count": len(results), "results": results},
        )

    def _do_list(self, inputs: dict) -> SkillResult:
        category = inputs.get("category") or None
        tag = inputs.get("tag") or None
        results = self._mgr.list_docs(category=category, tag=tag)
        return SkillResult(
            skill_name=self.name,
            success=True,
            message=f"Knowledge base has {len(results)} documents" + (f" (category={category})" if category else ""),
            payload={"count": len(results), "filters": {"category": category, "tag": tag}, "results": results},
        )

    def _do_add(self, inputs: dict) -> SkillResult:
        title = str(inputs.get("title") or "")
        content = str(inputs.get("content") or "")
        if not title or not content:
            return SkillResult(skill_name=self.name, success=False, message="add requirestitle 和 content 参数")
        category = str(inputs.get("category") or "")
        tags = inputs.get("tags") or None
        if isinstance(tags, str):
            tags = [t.strip() for t in tags.split(",") if t.strip()]
        doc_id = self._mgr.add_doc(title, content, category=category, tags=tags)
        doc = self._mgr.get_doc(doc_id)
        return SkillResult(
            skill_name=self.name,
            success=True,
            message=f"Document added: '{title}' ({doc.category if doc else ''})",
            payload={"doc_id": doc_id, "title": title, "category": doc.category if doc else ""},
        )

    def _do_remove(self, inputs: dict) -> SkillResult:
        doc_id = str(inputs.get("doc_id") or inputs.get("id") or "")
        if not doc_id:
            return SkillResult(skill_name=self.name, success=False, message="remove requiresdoc_id 参数")
        removed = self._mgr.remove_doc(doc_id)
        if removed:
            return SkillResult(skill_name=self.name, success=True, message=f"Document {doc_id} removed")
        return SkillResult(skill_name=self.name, success=False, message=f"Document not found: {doc_id}")

    def _do_stats(self) -> SkillResult:
        stats = self._mgr.stats()
        return SkillResult(
            skill_name=self.name,
            success=True,
            message=f"Knowledge base: {stats['total_docs']} docs, {stats['total_size_kb']} KB",
            payload=stats,
        )

    def _do_context(self, inputs: dict) -> SkillResult:
        category = inputs.get("category") or None
        max_chars = int(inputs.get("max_chars", 4000))
        ctx = self._mgr.as_prompt_context(category=category, max_chars=max_chars)
        return SkillResult(
            skill_name=self.name,
            success=True,
            message=f"Knowledge base context generated ({len(ctx)} chars)",
            payload={"context": ctx, "category": category},
        )

    def _do_reload(self) -> SkillResult:
        added = self._mgr.reload()
        return SkillResult(
            skill_name=self.name,
            success=True,
            message=f"Knowledge base refreshed: +{added} docs",
            payload={"added": added, "total": self._mgr.doc_count},
        )

    @property
    def manager(self) -> KnowledgeManager:
        """Expose the underlying KnowledgeManager for programmatic use."""
        self._ensure_loaded()
        return self._mgr
