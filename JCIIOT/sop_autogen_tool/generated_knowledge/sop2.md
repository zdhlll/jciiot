<!-- AI-GENERATED FROM DOCX - DO NOT MODIFY MANUALLY -->

# L2 Task - Cross-line Transport + Obstacle Avoidance

Level: L2 (max 15 points)
Scene: factory_sorting_3

## Task

Transport 1 Green-rimmed storage bin from Placement Point 1 to Place Station 3

## Station Mapping

- Pick Station 1 = input_6, center (11.94, 3.93)
- Place Station 3 = output_4, center (-0.17, -7.29)
- Robot start: (13.5, 0.0)
- Target object: ['green_tote_b01_upper', 'green_tote_b01_lower']

## Grasp Pose (BC Policy)

- Robot stop point at input_6: (6.00, 4.80, 0.0), yaw=-3.139

## Object Inventory (L2 Scene)

Every input port and its assigned graspable object:

- input_1: Green-rimmed storage bin
- input_6: ['green_tote_b01_upper', 'green_tote_b01_lower']

CRITICAL: When calling pick_up, you MUST provide the exact object_name from the inventory above.

## SOP Constraints

- Quantity to transport is exactly 1.
- Zero tolerance for collisions.
- Ensure material stability during transport.

## Generation Evidence

- Source DOCX: `sop+prompt/JCIIOT 2026 case 3 SOP.docx`
- Runtime policy: existing `knowledge/sop*.md` files were not read.

## Generation Warnings

- Filled missing exact object_name from task_config validation data.
