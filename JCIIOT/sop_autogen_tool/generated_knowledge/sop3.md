<!-- AI-GENERATED FROM DOCX - DO NOT MODIFY MANUALLY -->

# L3 Task - Cross-line Transport + Obstacle + Interference

Level: L3 (max 20 points)
Scene: factory_sorting_5

## Task

Transport a blue material transfer bin from Pick Station 1 to Place Station 2

## Station Mapping

- Pick Station 6 = aux_input_1, center (0.14, 8.47)
- Place Station 2 = output_5, center (4.87, -7.26)
- Robot start: (13.5, 0.0)
- Target object: ['blue_tote_b01_far_right', 'blue_tote_b01_near_right']

## Object Inventory (L3 Scene)

Every input port and its assigned graspable object:

- input_1: blue material transfer bin

CRITICAL: When calling pick_up, you MUST provide the exact object_name from the inventory above.

## SOP Constraints

- Ensure path safety and feasibility.
- Confirm material stability during transport.
- Verify placement area is unoccupied before placing.

## Generation Evidence

- Source DOCX: `sop+prompt/JCIIOT 2026 case 5 SOP.docx`
- Runtime policy: existing `knowledge/sop*.md` files were not read.

## Generation Warnings

- Source mismatch: docx/model derived input_6, task_config has aux_input_1.
- Corrected source from task_config validation data.
- Filled missing exact object_name from task_config validation data.
