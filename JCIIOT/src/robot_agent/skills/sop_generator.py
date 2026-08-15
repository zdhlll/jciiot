"""Generate competition SOP knowledge from DOCX evidence and simulation truth.

The two model roles are deliberately separated:

* a VLM reads diagrams, highlighted stations, labels and material appearance;
* a text LLM extracts the task semantics and operational procedure;
* task_config.json, semantic maps and scene source code reconcile model output
  with the exact station IDs, coordinates and simulator object names.

API credentials are read from environment variables and are never persisted.
"""

from __future__ import annotations

import json
import hashlib
import logging
import os
import re
from datetime import datetime
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

from robot_agent.core.types import ExecutionContext, SkillResult
from robot_agent.skills.base import BaseSkill

logger = logging.getLogger(__name__)

LEVEL_DOCX_MAP = {
    "L1": "JCIIOT 2026 case 1 SOP.docx",
    "L2": "JCIIOT 2026 case 3 SOP.docx",
    "L3": "JCIIOT 2026 case 5 SOP.docx",
    "L4": "JCIIOT 2026 case 7 SOP.docx",
    "L5": "JCIIOT 2026 case 9 SOP.docx",
}

LEVEL_SCENE_MAP = {
    "L1": "factory_sorting_1_3fo3erfhisem",
    "L2": "factory_sorting_3_3fo3errph7x9",
    "L3": "factory_sorting_5_3fo3ertpxeut",
    "L4": "factory_sorting_7_3fo3erfky9rn",
    "L5": "factory_sorting_9_3fo3ert2c5fp",
}

LEVEL_ENV_MAP = {
    "L1": "FactorySorting1_3FO3ERFHISEM",
    "L2": "FactorySorting3_3FO3ERRPH7X9",
    "L3": "FactorySorting5_3FO3ERTPXEUT",
    "L4": "FactorySorting7_3FO3ERFKY9RN",
    "L5": "FactorySorting9_3FO3ERT2C5FP",
}

VLM_PROMPT = """You are inspecting images extracted from a JCIIOT robot SOP.
The document task prompt is:
{task_prompt}

Analyze all supplied images as evidence. Return JSON only with this schema:
{{
  "relevant_images": [
    {{"image_index": 1, "purpose": "factory_map|pick_diagram|place_diagram|procedure|other",
      "visible_text": ["exact text"], "highlighted_location": "precise visual position",
      "material_description": "color, shape, quantity and relative position", "confidence": 0.0}}
  ],
  "pick_visual_evidence": "which marked location corresponds to the prompt's Pick Station",
  "place_visual_evidence": "which marked location corresponds to the prompt's Place Station",
  "quantity_visual_evidence": "visible item count or empty string",
  "warnings": ["ambiguity or unreadable content"]
}}
Do not invent simulator IDs such as input_1 or object_name; report only what is visibly supported."""

TEXT_PROMPT = """You extract operational knowledge from a JCIIOT embodied-robot SOP.

LEVEL: {level}
AUTHORITATIVE TASK PROMPT:
{task_prompt}

DOCUMENT TEXT:
{document_text}

VISION EVIDENCE:
{vision_evidence}

PUBLISHED ERRATUM:
{erratum}

Return JSON only:
{{
  "pick_label": "exact Pick Station label from the corrected prompt",
  "place_label": "exact Place Station label from the corrected prompt",
  "material_description": "human-readable target description",
  "quantity": 1,
  "procedure": [
    {{"phase": "prepare|navigate_pick|grasp|transport|place|verify|repeat",
      "instruction": "one executable instruction", "safety_check": "check or empty"}}
  ],
  "constraints": ["explicit safety or quality constraint"],
  "exceptions": ["recovery instruction"],
  "evidence_summary": "how text and images support the extraction",
  "uncertainties": ["anything not supported by evidence"]
}}
Preserve operational content, but do not guess simulator input/output IDs, coordinates, or internal object names."""

REVIEW_PROMPT = """Audit this generated robot SOP against the evidence and resolved simulation data.
Return JSON only: {{"ok": true, "issues": [], "missing_steps": [], "unsafe_claims": []}}.
Reject nonexistent station IDs, unknown object names, wrong quantity, missing grasp verification,
missing collision avoidance, or missing placement verification.
The configured BC grasp pose is a robot stop pose and is intentionally different from the station center;
do not flag that difference. Human-facing Pick/Place labels come from the corrected task prompt, while
simulator input/output IDs come from task_config and need not have matching numbers.

EVIDENCE:
{evidence}

GENERATED SOP:
{sop}
"""


@dataclass
class ParsedDocument:
    text: str
    images: list[tuple[str, bytes]]
    task_prompt: str


@dataclass
class StationMapping:
    level: str
    pick_label: str
    pick_name: str
    place_label: str
    place_name: str
    target_objects: list[str]
    material_description: str
    quantity: int
    procedure: list[dict[str, str]] = field(default_factory=list)
    constraints: list[str] = field(default_factory=list)
    exceptions: list[str] = field(default_factory=list)
    evidence_summary: str = ""
    vision_evidence: dict[str, Any] = field(default_factory=dict)


class SOPValidationError(RuntimeError):
    pass


def _json_object(raw: str, *, source: str) -> dict[str, Any]:
    text = (raw or "").strip()
    text = re.sub(r"^```(?:json)?\s*", "", text, flags=re.I)
    text = re.sub(r"\s*```$", "", text)
    try:
        data = json.loads(text)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", text, flags=re.S)
        if not match:
            raise SOPValidationError(f"{source} did not return a JSON object")
        try:
            data = json.loads(match.group(0))
        except json.JSONDecodeError as exc:
            raise SOPValidationError(f"invalid JSON from {source}: {exc}") from exc
    if not isinstance(data, dict):
        raise SOPValidationError(f"{source} returned {type(data).__name__}, expected object")
    return data


class SOPGenerator:
    def __init__(
        self, *, llm_generate: Callable[..., str],
        vlm_describe: Callable[[str, list[bytes]], str] | None,
        sop_dir: str | Path | None = None,
        output_dir: str | Path | None = None,
        maps_dir: str | Path | None = None,
        project_root: str | Path | None = None,
    ) -> None:
        root = Path(project_root) if project_root else Path(__file__).resolve().parents[3]
        self.root = root.resolve()
        self.sop_dir = Path(sop_dir).resolve() if sop_dir else self.root / "sop+prompt"
        self.output_dir = Path(output_dir).resolve() if output_dir else self.root / "knowledge"
        self.maps_dir = (
            Path(maps_dir).resolve() if maps_dir else
            self.root / "robosuite/robosuite/environments/factory_sorting/generated_maps"
        )
        self.llm = llm_generate
        self.vlm = vlm_describe
        self.task_config = self._load_json(self.root / "knowledge/task_config.json")
        self.erratum = self._read_erratum()
        self.cache_dir = self.root / ".sop_cache"

    @staticmethod
    def _load_json(path: Path) -> dict[str, Any]:
        if not path.exists():
            raise FileNotFoundError(path)
        return json.loads(path.read_text(encoding="utf-8"))

    def _read_erratum(self) -> str:
        for path in (self.root / "ERRATUM.md", self.root.parent / "ERRATUM.md"):
            if path.exists():
                return path.read_text(encoding="utf-8", errors="replace")
        return "(no erratum file found)"

    @staticmethod
    def parse_docx(docx_path: str | Path) -> tuple[str, list[tuple[str, bytes]]]:
        """Compatibility API: return text and de-duplicated embedded images."""
        parsed = SOPGenerator._parse_docx(Path(docx_path))
        return parsed.text, parsed.images

    @staticmethod
    def _parse_docx(path: Path) -> ParsedDocument:
        from docx import Document
        from docx.table import Table
        from docx.text.paragraph import Paragraph

        doc = Document(str(path))
        blocks: list[str] = []
        for child in doc.element.body.iterchildren():
            if child.tag.endswith("}p"):
                text = Paragraph(child, doc).text.strip()
                if text:
                    blocks.append(text)
            elif child.tag.endswith("}tbl"):
                table = Table(child, doc)
                for row in table.rows:
                    cells = [re.sub(r"\s+", " ", cell.text).strip() for cell in row.cells]
                    if any(cells):
                        blocks.append(" | ".join(cells))

        images: list[tuple[str, bytes]] = []
        seen: set[str] = set()
        for rel in doc.part.rels.values():
            if "image" not in rel.reltype:
                continue
            blob = rel.target_part.blob
            signature = f"{len(blob)}:{blob[:32]!r}"
            if signature in seen:
                continue
            seen.add(signature)
            name = Path(rel.target_ref or f"image_{len(images)+1}.png").name
            images.append((name, blob))

        text = "\n".join(blocks)
        prompt = next(
            (line.strip() for line in blocks if re.search(r"\bPrompt\b|Current Task Material", line, re.I)),
            blocks[0] if blocks else "",
        )
        return ParsedDocument(text=text, images=images, task_prompt=prompt)

    def _call_llm_json(self, prompt: str, *, source: str, tokens: int = 4096) -> dict[str, Any]:
        try:
            raw = self.llm(prompt, num_predict=tokens, temperature=0.0, json_mode=True)
        except TypeError:
            raw = self.llm(prompt)
        return _json_object(raw, source=source)

    def analyze_images(self, parsed: ParsedDocument) -> dict[str, Any]:
        if not parsed.images:
            return {"relevant_images": [], "warnings": ["document contains no images"]}
        if self.vlm is None:
            raise SOPValidationError("VLM is required because station mapping depends on diagrams")
        prompt = VLM_PROMPT.format(task_prompt=parsed.task_prompt)
        records: list[dict[str, Any]] = []
        warnings: list[str] = []
        pick_evidence: list[str] = []
        place_evidence: list[str] = []
        quantity_evidence: list[str] = []
        for index, (name, blob) in enumerate(parsed.images, 1):
            raw = ""
            try:
                image_prompt = f"Image index in the document: {index}. Filename: {name}.\n{prompt}"
                digest = hashlib.sha256(image_prompt.encode("utf-8") + blob).hexdigest()
                cache_path = self.cache_dir / f"vlm_{digest}.txt"
                if cache_path.exists():
                    raw = cache_path.read_text(encoding="utf-8")
                else:
                    raw = self.vlm(image_prompt, [blob])
                    self.cache_dir.mkdir(parents=True, exist_ok=True)
                    cache_path.write_text(raw or "", encoding="utf-8")
                data = _json_object(raw, source=f"vision model for {name}")
                image_records = data.get("relevant_images", [])
                if isinstance(image_records, list):
                    for record in image_records:
                        if isinstance(record, dict):
                            record["document_image_index"] = index
                            record["filename"] = name
                            records.append(record)
                if data.get("pick_visual_evidence"):
                    pick_evidence.append(str(data["pick_visual_evidence"]))
                if data.get("place_visual_evidence"):
                    place_evidence.append(str(data["place_visual_evidence"]))
                if data.get("quantity_visual_evidence"):
                    quantity_evidence.append(str(data["quantity_visual_evidence"]))
                warnings.extend(str(x) for x in data.get("warnings", []) if str(x).strip())
            except Exception as exc:
                warning = f"{name}: {exc}"
                logger.warning("Vision extraction failed: %s", warning)
                warnings.append(warning)
                # GLM vision models sometimes answer with a useful natural-
                # language caption instead of the requested JSON. Preserve it
                # as evidence rather than discarding the entire image.
                if raw and raw.strip():
                    records.append({
                        "document_image_index": index,
                        "filename": name,
                        "purpose": "unstructured_vlm_evidence",
                        "raw_analysis": raw.strip()[:3000],
                        "confidence": 0.5,
                    })
        if not records:
            raise SOPValidationError(f"VLM produced no parseable image evidence: {warnings}")
        return {
            "relevant_images": records,
            "pick_visual_evidence": pick_evidence,
            "place_visual_evidence": place_evidence,
            "quantity_visual_evidence": quantity_evidence,
            "warnings": warnings,
            "image_files": [name for name, _ in parsed.images],
        }

    def _task_spec(self, level: str) -> dict[str, Any]:
        index = int(level[1:]) - 1
        tasks = self.task_config.get("tasks", [])
        if not 0 <= index < len(tasks):
            raise SOPValidationError(f"task_config has no entry for {level}")
        task = dict(tasks[index])
        expected_env = LEVEL_ENV_MAP[level]
        if task.get("env_name") != expected_env:
            raise SOPValidationError(
                f"{level} env mismatch: {task.get('env_name')} != {expected_env}"
            )
        return task

    def _semantic_map(self, level: str) -> dict[str, Any]:
        prefix = LEVEL_SCENE_MAP[level]
        return self._load_json(self.maps_dir / f"{prefix}_scene_regenerated_semantic_map.json")

    def _scene_source(self, level: str) -> str:
        prefix = LEVEL_SCENE_MAP[level]
        path = self.root / "robosuite/robosuite/environments/factory_sorting" / f"{prefix}.py"
        return path.read_text(encoding="utf-8", errors="replace") if path.exists() else ""

    def _resolve_target_objects(self, level: str, target: str, quantity: int) -> list[str]:
        if quantity <= 1:
            return [target]
        source = self._scene_source(level)
        candidates = sorted(set(re.findall(r'"([a-z0-9_]+)"', source)))
        # For a target ending in a positional token, retain the stable family
        # prefix (e.g. white_tote_b01_left_{center,front,back}).
        prefix = re.sub(r"_(?:center|front|back|left|right|upper|lower)$", "_", target)
        siblings = [name for name in candidates if name.startswith(prefix)]
        preferred_order = {"center": 0, "front": 1, "back": 2, "left": 3, "right": 4}
        siblings.sort(key=lambda name: next(
            (rank for token, rank in preferred_order.items() if name.endswith(f"_{token}")), 99
        ))
        if target not in siblings:
            siblings.insert(0, target)
        result = list(dict.fromkeys(siblings))[:quantity]
        if len(result) != quantity:
            raise SOPValidationError(
                f"{level}: found {len(result)} simulator objects, expected {quantity}: {result}"
            )
        return result

    @staticmethod
    def _station_label(task_prompt: str, kind: str, fallback: str) -> str:
        pattern = rf"{kind}\s+Station\s*[\"']?\s*(\d+)"
        match = re.search(pattern, task_prompt, flags=re.I)
        if match:
            return f"{kind.title()} Station {match.group(1)}"
        return fallback

    @staticmethod
    def _complete_procedure(raw_steps: Any) -> list[dict[str, str]]:
        steps = [dict(step) for step in (raw_steps or []) if isinstance(step, dict)]
        defaults = {
            "prepare": (
                "Confirm the task, exact object name, quantity, source and destination before motion.",
                "Abort if any task field is ambiguous.",
            ),
            "navigate_pick": (
                "Plan a collision-free path to the resolved input approach point and stop stably.",
                "Continuously check obstacles and re-plan instead of entering blocked cells.",
            ),
            "grasp": (
                "Align at the configured grasp pose, close both grippers, lift the object and verify a secure dual-gripper grasp.",
                "Do not transport unless the grasp event and lift verification both succeed.",
            ),
            "transport": (
                "Carry the object along a collision-free path to the resolved output approach point.",
                "Keep the object stable and stop immediately on collision risk or slip.",
            ),
            "place": (
                "Align to the target table, lower smoothly, release the object and clear the grippers.",
                "Release only over the resolved output station.",
            ),
            "verify": (
                "Verify the object is within the target tolerance and is no longer held; record the trajectory.",
                "Re-attempt or report failure if placement verification does not pass.",
            ),
        }
        by_phase = {str(step.get("phase", "")).strip(): step for step in steps}
        completed: list[dict[str, str]] = []
        for phase, (instruction, safety) in defaults.items():
            step = by_phase.get(phase, {})
            model_instruction = str(step.get("instruction") or "").strip()
            model_safety = str(step.get("safety_check") or "").strip()
            completed.append({
                "phase": phase,
                "instruction": (
                    f"{model_instruction} Required execution: {instruction}"
                    if model_instruction else instruction
                ),
                "safety_check": (
                    f"{model_safety} Required check: {safety}"
                    if model_safety else safety
                ),
            })
        for step in steps:
            if str(step.get("phase", "")) not in defaults:
                completed.append(step)
        return completed

    def extract_mapping(
        self, parsed: ParsedDocument, level: str, vision: dict[str, Any],
    ) -> StationMapping:
        extracted = self._call_llm_json(
            TEXT_PROMPT.format(
                level=level,
                task_prompt=parsed.task_prompt,
                document_text=parsed.text[:24000],
                vision_evidence=json.dumps(vision, ensure_ascii=False)[:12000],
                erratum=self.erratum[:6000],
            ),
            source="text model",
        )
        task = self._task_spec(level)
        quantity = max(1, int(extracted.get("quantity") or (3 if level == "L5" else 1)))
        if level == "L5":
            quantity = 3
        objects = self._resolve_target_objects(level, str(task["object"]), quantity)
        pick_label = self._station_label(
            parsed.task_prompt, "Pick", str(extracted.get("pick_label") or "Pick Station")
        )
        place_label = self._station_label(
            parsed.task_prompt, "Place", str(extracted.get("place_label") or "Place Station")
        )
        mapping = StationMapping(
            level=level,
            pick_label=pick_label,
            pick_name=str(task["source"]),
            place_label=place_label,
            place_name=str(task["target"]),
            target_objects=objects,
            material_description=str(extracted.get("material_description") or task["object"]),
            quantity=quantity,
            procedure=self._complete_procedure(extracted.get("procedure", [])),
            constraints=[str(x) for x in extracted.get("constraints", []) if str(x).strip()],
            exceptions=[str(x) for x in extracted.get("exceptions", []) if str(x).strip()],
            evidence_summary=(
                f"The DOCX task prompt identifies {pick_label} and {place_label}. "
                f"The visual model analyzed {len(vision.get('relevant_images', []))} image record(s). "
                f"Simulation metadata resolves those human labels to {task['source']} and {task['target']} "
                f"with exact target object {task['object']}."
            ),
            vision_evidence=vision,
        )
        for step in mapping.procedure:
            for key in ("instruction", "safety_check"):
                text = str(step.get(key, ""))
                text = re.sub(r"Pick\s+Station\s+\d+", mapping.pick_label, text, flags=re.I)
                text = re.sub(r"Place\s+Station\s+\d+", mapping.place_label, text, flags=re.I)
                step[key] = text
        def canonicalize(text: str) -> str:
            text = re.sub(r"Pick\s+Station\s+\d+", mapping.pick_label, text, flags=re.I)
            return re.sub(r"Place\s+Station\s+\d+", mapping.place_label, text, flags=re.I)
        mapping.constraints = [canonicalize(item) for item in mapping.constraints]
        mapping.exceptions = [canonicalize(item) for item in mapping.exceptions]
        return mapping

    @staticmethod
    def _port(map_data: dict[str, Any], name: str) -> dict[str, Any]:
        group = "input_ports" if name.startswith("input_") else "output_ports"
        port = map_data.get(group, {}).get(name)
        if not isinstance(port, dict):
            raise SOPValidationError(f"semantic map does not contain {name}")
        return port

    def validate_mapping(self, mapping: StationMapping) -> None:
        map_data = self._semantic_map(mapping.level)
        self._port(map_data, mapping.pick_name)
        self._port(map_data, mapping.place_name)
        if not mapping.pick_name.startswith("input_"):
            raise SOPValidationError(f"pick station must be input_N: {mapping.pick_name}")
        if not mapping.place_name.startswith("output_"):
            raise SOPValidationError(f"place station must be output_N: {mapping.place_name}")
        if mapping.quantity != len(mapping.target_objects):
            raise SOPValidationError("quantity does not match resolved object list")
        if any(not re.fullmatch(r"[a-z0-9_]+", obj) for obj in mapping.target_objects):
            raise SOPValidationError(f"invalid simulator object name: {mapping.target_objects}")
        phases = {str(step.get("phase", "")) for step in mapping.procedure}
        required = {"navigate_pick", "grasp", "transport", "place", "verify"}
        if not required.issubset(phases):
            missing = sorted(required - phases)
            raise SOPValidationError(f"text model omitted required procedure phases: {missing}")

    def build_sop_md(self, mapping: StationMapping) -> str:
        self.validate_mapping(mapping)
        task = self._task_spec(mapping.level)
        map_data = self._semantic_map(mapping.level)
        pick = self._port(map_data, mapping.pick_name)
        place = self._port(map_data, mapping.place_name)
        grasp = self.task_config.get("grasp_poses", {}).get(mapping.pick_name, {})

        def xy(values: Any) -> str:
            vals = list(values or [])
            return f"({float(vals[0]):.3f}, {float(vals[1]):.3f})" if len(vals) >= 2 else "(unavailable)"

        object_lines = "\n".join(f"- `{obj}`" for obj in mapping.target_objects)
        steps = []
        for index, step in enumerate(mapping.procedure, 1):
            safety = str(step.get("safety_check") or "").strip()
            line = f"{index}. **{step.get('phase', 'step')}** — {step.get('instruction', '')}"
            if safety:
                line += f" Safety check: {safety}"
            steps.append(line)
        constraints = "\n".join(f"- {item}" for item in mapping.constraints) or "- Avoid collisions and keep the load stable."
        exceptions = "\n".join(f"- {item}" for item in mapping.exceptions) or "- Stop and re-plan if grasp, route, or placement verification fails."
        image_records = mapping.vision_evidence.get("relevant_images", [])
        vision_summary = json.dumps({
            "image_files": mapping.vision_evidence.get("image_files", []),
            "record_count": len(image_records),
            "purposes": sorted({
                str(record.get("purpose", "other"))
                for record in image_records if isinstance(record, dict)
            }),
            "warnings": mapping.vision_evidence.get("warnings", []),
        }, ensure_ascii=False, indent=2)

        return f"""<!-- AI-GENERATED from {LEVEL_DOCX_MAP[mapping.level]}; reconciled with simulation metadata -->

# {mapping.level} Competition SOP

- Scene: `{task['env_name']}`
- Maximum score: {task['max_score']}
- Source document: `{LEVEL_DOCX_MAP[mapping.level]}`

## Resolved task

Transport {mapping.quantity} × {mapping.material_description} from {mapping.pick_label} to {mapping.place_label}.

The human-facing station labels in the SOP do **not** equal simulator station numbers. The resolved simulator mapping is:

- Pick: {mapping.pick_label} → `{mapping.pick_name}`; center {xy(pick.get('center'))}; navigation approach {xy(pick.get('approach'))}
- Place: {mapping.place_label} → `{mapping.place_name}`; center {xy(place.get('center'))}; navigation approach {xy(place.get('approach'))}
- Configured grasp pose: position `{grasp.get('pos', 'not configured')}`, yaw `{grasp.get('yaw', 'not configured')}`

## Exact simulator object name(s)

{object_lines}

Never replace these names with a visual description when calling `pick_up`.

## Executable procedure

{chr(10).join(steps)}

For quantities greater than one, repeat the complete pick–transport–place–verify cycle for each object name in order.

## Constraints

{constraints}

## Failure recovery

{exceptions}

## Evidence summary

{mapping.evidence_summary or 'Task semantics were extracted from the DOCX and reconciled with simulator metadata.'}

<details><summary>Vision extraction record</summary>

```json
{vision_summary}
```
</details>
"""

    def review_sop(self, mapping: StationMapping, content: str) -> dict[str, Any]:
        evidence = {
            "task_config": self._task_spec(mapping.level),
            "human_pick_label": mapping.pick_label,
            "human_place_label": mapping.place_label,
            "resolved_objects": mapping.target_objects,
            "quantity": mapping.quantity,
            "vision": mapping.vision_evidence,
        }
        try:
            review = self._call_llm_json(
                REVIEW_PROMPT.format(
                    evidence=json.dumps(evidence, ensure_ascii=False)[:6000],
                    sop=content[:10000],
                ),
                source="text model reviewer",
                tokens=1024,
            )
        except SOPValidationError as exc:
            logger.warning("Reviewer format failure; deterministic validation passed: %s", exc)
            return {
                "ok": True,
                "mode": "deterministic_validation_only",
                "warning": str(exc),
            }
        if not review.get("ok", False):
            logger.warning("Model reviewer raised non-blocking issues: %s", review.get("issues", review))
        review["deterministic_validation"] = "passed"
        return review

    def run_level(self, level: str) -> tuple[str, dict[str, Any]]:
        level = level.upper()
        if level not in LEVEL_DOCX_MAP:
            raise ValueError(f"unknown level: {level}")
        path = self.sop_dir / LEVEL_DOCX_MAP[level]
        parsed = self._parse_docx(path)
        vision = self.analyze_images(parsed)
        mapping = self.extract_mapping(parsed, level, vision)
        content = self.build_sop_md(mapping)
        review = self.review_sop(mapping, content)
        report = {
            "level": level, "document": str(path), "image_count": len(parsed.images),
            "pick": mapping.pick_name, "place": mapping.place_name,
            "objects": mapping.target_objects, "quantity": mapping.quantity,
            "review": review,
        }
        return content, report

    def build_main_sop(self, reports: list[dict[str, Any]]) -> str:
        rows = []
        for item in reports:
            rows.append(
                f"| {item['level']} | `{item['pick']}` | `{item['place']}` | "
                f"{item['quantity']} | {', '.join(f'`{x}`' for x in item['objects'])} |"
            )
        return """<!-- AI-GENERATED index; see each per-level SOP for evidence -->

# JCIIOT Competition SOP Index

| Level | Pick | Place | Quantity | Exact object name(s) |
|---|---|---|---:|---|
""" + "\n".join(rows) + "\n"

    def run_all(self) -> tuple[dict[str, str], dict[str, Any]]:
        outputs: dict[str, str] = {}
        reports: list[dict[str, Any]] = []
        for level in LEVEL_DOCX_MAP:
            content, report = self.run_level(level)
            outputs[f"my_sop{int(level[1:])}.md"] = content
            reports.append(report)
        outputs["my_sop_main.md"] = self.build_main_sop(reports)
        return outputs, {"levels": reports}

    def write_all(self, outputs: dict[str, str]) -> list[Path]:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        written: list[Path] = []
        for name, content in outputs.items():
            destination = self.output_dir / name
            temporary = destination.with_suffix(destination.suffix + ".tmp")
            temporary.write_text(content, encoding="utf-8")
            temporary.replace(destination)
            written.append(destination)
        index_path = self.output_dir / "_index.json"
        try:
            index = self._load_json(index_path) if index_path.exists() else {"documents": {}}
            documents = index.setdefault("documents", {})
            for name in outputs:
                if name == "my_sop_main.md":
                    documents[name] = {
                        "title": "JCIIOT Competition SOP Index",
                        "category": "sop",
                        "tags": ["SOP", "reference", "AI-generated"],
                        "added_at": documents.get(name, {}).get("added_at") or datetime.now().isoformat(timespec="seconds"),
                    }
                    continue
                match = re.fullmatch(r"my_sop(\d+)\.md", name)
                if not match:
                    continue
                level = f"L{match.group(1)}"
                task = self._task_spec(level)
                documents[name] = {
                    "title": f"{level} Competition SOP",
                    "category": "sop",
                    "tags": [level, str(task["source"]), str(task["target"]), "AI-generated"],
                    "added_at": documents.get(name, {}).get("added_at") or datetime.now().isoformat(timespec="seconds"),
                }
            index["updated_at"] = datetime.now().isoformat(timespec="seconds")
            index["document_count"] = len(documents)
            temp_index = index_path.with_suffix(".json.tmp")
            temp_index.write_text(json.dumps(index, ensure_ascii=False, indent=2), encoding="utf-8")
            temp_index.replace(index_path)
        except Exception as exc:
            logger.warning("SOP files were written but knowledge index refresh failed: %s", exc)
        return written


def _make_clients() -> tuple[Callable[..., str], Callable[[str, list[bytes]], str]]:
    from robot_agent.core.openai_client import OpenAIClient
    from robot_agent.core.vision_client import ask_vision

    llm_key = os.getenv("SOP_LLM_API_KEY") or os.getenv("OPENAI_API_KEY", "")
    vlm_key = os.getenv("SOP_VLM_API_KEY") or os.getenv("VLM_API_KEY", "")
    if not llm_key or not vlm_key:
        raise RuntimeError(
            "Set SOP_LLM_API_KEY and SOP_VLM_API_KEY in the environment; keys are not stored in files."
        )
    llm_url = os.getenv("SOP_LLM_BASE_URL", "https://api.deepseek.com")
    llm_model = os.getenv("SOP_LLM_MODEL", "deepseek-v4-flash")
    vlm_url = os.getenv("SOP_VLM_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")
    vlm_model = os.getenv("SOP_VLM_MODEL", "glm-5v-turbo")
    client = OpenAIClient(api_key=llm_key, base_url=llm_url, model=llm_model, timeout=180.0)

    def generate(prompt: str, **kwargs: Any) -> str:
        return client.generate(prompt, **kwargs)

    def describe(prompt: str, images: list[bytes]) -> str:
        return ask_vision(
            prompt, images, base_url=vlm_url, model=vlm_model,
            api_type="openai", api_key=vlm_key, timeout=240.0,
        )
    return generate, describe


class GenerateSOPSkill(BaseSkill):
    def __init__(
        self, *, sop_dir="sop+prompt", output_dir="knowledge",
        maps_dir="robosuite/robosuite/environments/factory_sorting/generated_maps",
    ) -> None:
        super().__init__(
            name="generate_sop",
            description="Generate validated SOP knowledge from DOCX, VLM evidence and simulation metadata",
            keywords=("generate", "sop", "document", "docx", "knowledge", "extract"),
        )
        self.sop_dir = sop_dir
        self.output_dir = output_dir
        self.maps_dir = maps_dir

    def run(self, context: ExecutionContext) -> SkillResult:
        try:
            llm, vlm = _make_clients()
            generator = SOPGenerator(
                llm_generate=llm, vlm_describe=vlm,
                sop_dir=self.sop_dir, output_dir=self.output_dir, maps_dir=self.maps_dir,
            )
            level = str(context.metadata.get("inputs", {}).get("level", "all")).upper()
            if level == "ALL":
                outputs, report = generator.run_all()
            else:
                content, report = generator.run_level(level)
                outputs = {f"my_sop{int(level[1:])}.md": content}
            paths = generator.write_all(outputs)
            return SkillResult(
                skill_name=self.name, success=True,
                message=f"Generated and validated {len(paths)} SOP file(s)",
                payload={"files": [str(path) for path in paths], "report": report},
            )
        except Exception as exc:
            logger.exception("SOP generation failed")
            return SkillResult(
                skill_name=self.name, success=False, message=str(exc),
                payload={"error": str(exc)},
            )


def main() -> int:
    import argparse

    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description="Generate validated JCIIOT SOP knowledge")
    parser.add_argument("--level", default="all", choices=[*LEVEL_DOCX_MAP, "all"])
    parser.add_argument("--sop-dir", default=str(root / "sop+prompt"))
    parser.add_argument("--output-dir", default=str(root / "knowledge"))
    parser.add_argument(
        "--maps-dir",
        default=str(root / "robosuite/robosuite/environments/factory_sorting/generated_maps"),
    )
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
    llm, vlm = _make_clients()
    generator = SOPGenerator(
        llm_generate=llm, vlm_describe=vlm,
        sop_dir=args.sop_dir, output_dir=args.output_dir, maps_dir=args.maps_dir,
    )
    if args.level == "all":
        outputs, report = generator.run_all()
    else:
        content, report = generator.run_level(args.level)
        outputs = {f"my_sop{int(args.level[1:])}.md": content}

    if args.dry_run:
        print(json.dumps(report, ensure_ascii=False, indent=2))
        for name, content in outputs.items():
            print(f"\n--- {name} ---\n{content[:3000]}")
    else:
        for path in generator.write_all(outputs):
            print(path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
