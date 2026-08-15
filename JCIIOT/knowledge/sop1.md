<!-- AI-GENERATED FROM DOCX - DO NOT MODIFY MANUALLY -->

# L1 Task - Single-line Transport

Level: L1 (max 10 points)
Scene: factory_sorting_1

## Task

Transport a blue, hollow plastic box from Pick Station 2 to Place Station 3

## Station Mapping

- Pick Station 2 = input_5, center (7.19, 3.94)
- Place Station 3 = output_4, center (-0.17, -7.29)
- Robot start: (13.5, 0.0)
- Target object: ['line_5_container_h01_near', 'line_5_container_h01_far']

## Grasp Pose (BC Policy)

- Robot stop point at input_5: (8.00, 4.60, 0.0), yaw=-3.139

## Object Inventory (L1 Scene)

Every input port and its assigned graspable object:

- input_1: blue, hollow plastic box
- input_5: ['line_5_container_h01_near', 'line_5_container_h01_far']

CRITICAL: When calling pick_up, you MUST provide the exact object_name from the inventory above.

## SOP Constraints

- Inspect pick station area for clearance before grasping.
- Plan grasping path to avoid touching other materials.
- Maintain secure grip to prevent slipping during transit.
- Navigate optimal path while dynamically avoiding obstacles.
- Ensure material stability; avoid shaking or collision.
- Verify placement area is clear before releasing.
- Place material precisely within designated boundaries.

## Generation Evidence

- Source DOCX: `sop+prompt/JCIIOT 2026 case 1 SOP.docx`
- Runtime policy: existing `knowledge/sop*.md` files were not read.

## Generation Warnings

- Filled missing exact object_name from task_config validation data.
