<!-- AI-GENERATED FROM DOCX - DO NOT MODIFY MANUALLY -->

# L4 Task - Large Object Cross-line Transport

Level: L4 (max 25 points)
Scene: factory_sorting_7

## Task

Transfer a blue, hollow plastic box from Pick Station 5 to Place Station 2

## Station Mapping

- Pick Station 5 = input_2, center (-9.76, 5.01)
- Place Station 2 = output_5, center (4.87, -7.26)
- Robot start: (13.5, 0.0)
- Target object: ['blue_container_h01_back_upper', 'blue_container_h01_back_lower']

## Grasp Pose (BC Policy)

- Robot stop point at input_2: (8.56, -3.92, 0.0), yaw=-3.140

## Object Inventory (L4 Scene)

Every input port and its assigned graspable object:

- input_1: blue, hollow plastic box
- input_2: ['blue_container_h01_back_upper', 'blue_container_h01_back_lower']

CRITICAL: When calling pick_up, you MUST provide the exact object_name from the inventory above.

## SOP Constraints

- Ensure stable transport and accurate placement at destination.

## Generation Evidence

- Source DOCX: `sop+prompt/JCIIOT 2026 case 7 SOP.docx`
- Runtime policy: existing `knowledge/sop*.md` files were not read.

## Generation Warnings

- Filled missing exact object_name from task_config validation data.
