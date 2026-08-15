<!-- AI-GENERATED FROM DOCX - DO NOT MODIFY MANUALLY -->

# Standard Operating Procedure (SOP)

Task ID: MT-MOBILE-001
Version: v2.0

## Standard Transport Workflow

1. Navigate to Pick Station
2. Pick material: move arm above object -> close gripper -> lift 150mm -> confirm grasp
3. Navigate to Place Station with object held
4. Place material: lower to table height -> open gripper -> confirm deviation < 10mm
5. Return or repeat

## Task Coordinate Reference

All five levels, robot start at (13.5, 0.0):

| Level | Scene | Pick Station | Pick Coords | Object Name | Place Station | Place Coords |
| --- | --- | --- | --- | --- | --- | --- |
| L1 | factory_sorting_1 | Pick Station 2 | input_5 (7.19, 3.94) | ['line_5_container_h01_near', 'line_5_container_h01_far'] | Place Station 3 | output_4 (-0.17, -7.29) |
| L2 | factory_sorting_3 | Pick Station 1 | input_6 (11.94, 3.93) | ['green_tote_b01_upper', 'green_tote_b01_lower'] | Place Station 3 | output_4 (-0.17, -7.29) |
| L3 | factory_sorting_5 | Pick Station 6 | aux_input_1 (0.14, 8.47) | ['blue_tote_b01_far_right', 'blue_tote_b01_near_right'] | Place Station 2 | output_5 (4.87, -7.26) |
| L4 | factory_sorting_7 | Pick Station 5 | input_2 (-9.76, 5.01) | ['blue_container_h01_back_upper', 'blue_container_h01_back_lower'] | Place Station 2 | output_5 (4.87, -7.26) |
| L5 | factory_sorting_9 | Pick Station 6 | input_1 (-14.54, 5.01) | ['white_tote_b01_left_center', 'white_tote_b01_left_front', 'white_tote_b01_left_back'] | Place Station 6 | aux_output_1 (0.14, 8.47) |

## CRITICAL pick_up Rules

- pick_up requires BOTH `target` (station name like input_6) AND `object_name` (exact object name from the table above)
- Never guess object names - always use the exact name from the per-level SOP Object Inventory

## BC Policy Grasp Poses

| Input Station | Grasp Pose (x, y, yaw) |
| --- | --- |
| input_1 | (5.03, -3.84, -3.140) |
| input_2 | (8.56, -3.92, -3.140) |
| input_3 | (12.38, -3.76, -3.140) |
| input_4 | (15.80, -3.77, -3.140) |
| input_5 | (8.00, 4.60, -3.139) |
| input_6 | (6.00, 4.80, -3.139) |
