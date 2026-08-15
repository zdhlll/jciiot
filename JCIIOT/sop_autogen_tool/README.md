# SOP Autogen Tool

This folder is intentionally separate from the existing project code. It
generates SOP knowledge Markdown files from the original `.docx` files without
reading `knowledge/sop*.md` or `knowledge/sop_main.md` at runtime.

## What It Reads

- `sop+prompt/*.docx`
- `../ERRATUM.md`
- `robosuite/.../generated_maps/*_semantic_map.json`
- `knowledge/task_config.json` only for validation and missing exact object-name
  fallback

## What It Does Not Read

- `knowledge/sop1.md`
- `knowledge/sop2.md`
- `knowledge/sop3.md`
- `knowledge/sop4.md`
- `knowledge/sop5.md`
- `knowledge/sop_main.md`

The output format is encoded as local templates in `generate_sops.py` so the
generated files follow the current knowledge-base style without using the old
SOP files during inference.

## Run

From the `JCIIOT` project root:

```powershell
python sop_autogen_tool/generate_sops.py --project-root . --out-dir sop_autogen_tool/generated_knowledge
```

Use a text LLM by setting one of these:

```powershell
$env:OPENAI_API_KEY="..."
$env:OPENAI_BASE_URL="https://api.deepseek.com"
$env:OPENAI_MODEL="deepseek-v4-flash"
```

or:

```powershell
$env:OLLAMA_BASE_URL="http://localhost:11434"
$env:OLLAMA_MODEL="qwen3.6:27b-mtp-q4_K_M"
```

Use a vision model for embedded SOP images by setting:

```powershell
$env:VLM_BASE_URL="http://localhost:11434"
$env:VLM_MODEL="qwen3-vl:8b"
```

If the LLM is unavailable, the tool falls back to deterministic extraction and
records that in `generation_report.json`. For competition audit, prefer running
with `--require-llm`.

```powershell
python sop_autogen_tool/generate_sops.py --project-root . --out-dir sop_autogen_tool/generated_knowledge --require-llm
```

## Replace The Old Knowledge Files

After checking the generated files:

```powershell
Copy-Item sop_autogen_tool/generated_knowledge/sop*.md knowledge/ -Force
Copy-Item sop_autogen_tool/generated_knowledge/sop_main.md knowledge/ -Force
```

Then run the app normally. `KnowledgeManager.reload()` will rescan the Markdown
files.

## Optional Code Integration

For judge review inside the allowed code area, copy the generator file to:

```powershell
Copy-Item sop_autogen_tool/generate_sops.py src/robot_agent/workflows/sop_generation_flow.py -Force
```

Then it can be invoked as:

```powershell
python src/robot_agent/workflows/sop_generation_flow.py --project-root . --out-dir knowledge
```
