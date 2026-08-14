"""Read Document Skill — extract text + images from .docx, analyze with VLM."""

from __future__ import annotations

import logging
from pathlib import Path

from robot_agent.core.types import ExecutionContext, SkillResult
from robot_agent.skills.base import BaseSkill

logger = logging.getLogger(__name__)


class ReadDocumentSkill(BaseSkill):
    """Read a .docx file: extract text, optionally describe images via VLM.

    LLM can invoke this with::

        {"skill_name": "read_document",
         "inputs": {"file": "knowledge/JCIIOT_2026_case_1_SOP.docx",
                    "use_vision": true}}
    """

    def __init__(
        self,
        *,
        ollama_base_url: str = "http://localhost:11434",
        vision_model: str = "qwen3-vl:8b",
        api_type: str = "ollama",
        api_key: str = "",
    ) -> None:
        super().__init__(
            name="read_document",
            description="Read .docx files, extract text and analyze images with vision model",
            keywords=("read", "document", "docx", "analyze", "vision", "parse", "extract"),
        )
        self._ollama_url = ollama_base_url
        self._vision_model = vision_model
        self._api_type = api_type
        self._api_key = api_key

    def run(self, context: ExecutionContext) -> SkillResult:
        file_path = context.metadata.get("inputs", {}).get("file", "")
        use_vision = context.metadata.get("inputs", {}).get("use_vision", True)

        path = Path(file_path)
        if not path.exists():
            return SkillResult(
                skill_name=self.name, success=False,
                message=f"File not found: {file_path}",
                payload={"file": file_path},
            )

        try:
            from docx import Document
            doc = Document(str(path))

            # Extract all paragraph text
            paragraphs = [p.text.strip() for p in doc.paragraphs if p.text.strip()]
            full_text = "\n".join(paragraphs)

            # Extract images
            images = {}
            for rel in doc.part.rels.values():
                if "image" in rel.reltype:
                    name = rel.target_ref.split("/")[-1] if rel.target_ref else "image.png"
                    images[name] = rel.target_part.blob

            # Optional vision analysis
            img_descriptions = {}
            if use_vision and images:
                try:
                    from robot_agent.core.vision_client import ask_vision
                    for name, img_data in images.items():
                        try:
                            desc = ask_vision(
                                "Describe this factory layout image. What stations, tables, "
                                "production lines, objects, and their positions do you see?",
                                img_data,
                                base_url=self._ollama_url,
                                model=self._vision_model,
                                api_type=self._api_type,
                                api_key=self._api_key,
                            )
                            img_descriptions[name] = desc
                        except Exception as exc:
                            img_descriptions[name] = f"VLM error: {exc}"
                except Exception as exc:
                    logger.warning("Vision analysis skipped: %s", exc)

            return SkillResult(
                skill_name=self.name,
                success=True,
                message=f"Read {len(paragraphs)} paragraphs, {len(images)} images"
                        + (f", {len(img_descriptions)} analyzed by VLM" if img_descriptions else ""),
                payload={
                    "file": str(path),
                    "paragraph_count": len(paragraphs),
                    "image_count": len(images),
                    "images_analyzed": len(img_descriptions),
                    "text": full_text,
                    "image_descriptions": img_descriptions,
                },
            )
        except Exception as exc:
            logger.exception("ReadDocumentSkill failed")
            return SkillResult(
                skill_name=self.name, success=False,
                message=f"Failed: {exc}",
                payload={"file": file_path, "error": str(exc)},
            )
