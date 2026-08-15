<!-- AI-GENERATED FROM DOCX - DO NOT MODIFY MANUALLY -->

# L5 Task - Extreme Distance Transport

Level: L5 (max 30 points)
Scene: factory_sorting_9

## Task

Move the three white-rimmed storage bins from Pick Station 6 to Place Station 1

## Station Mapping

- Pick Station 6 = input_1, center (-14.54, 5.01)
- Place Station 6 = aux_output_1, center (0.14, 8.47)
- Robot start: (13.5, 0.0)
- Target object: ['white_tote_b01_left_center', 'white_tote_b01_left_front', 'white_tote_b01_left_back']

## Grasp Pose (BC Policy)

- Robot stop point at input_1: (5.03, -3.84, 0.0), yaw=-3.140

## Object Inventory (L5 Scene)

Every input port and its assigned graspable object:

- input_1: white-rimmed storage bins, ['white_tote_b01_left_center', 'white_tote_b01_left_front', 'white_tote_b01_left_back']

CRITICAL: When calling pick_up, you MUST provide the exact object_name from the inventory above.

## SOP Constraints

- Do not pick up wrong materials.
- Avoid collisions during transport.
- Ensure placement is not crooked.
- Dodge obstacles on path.

## Generation Evidence

- Source DOCX: `sop+prompt/JCIIOT 2026 case 9 SOP.docx`
- Runtime policy: existing `knowledge/sop*.md` files were not read.

## Generation Warnings

- Target mismatch: docx/model derived output_6, task_config has aux_output_1.
- Corrected target from task_config validation data.
- Filled missing exact object_name from task_config validation data.
