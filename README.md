# Vehicle Auto Level custom node for ComfyUI

This custom node estimates roll/tilt from the cropped vehicle cutout and rotates the vehicle automatically before the rest of the composite chain.

## What it does
- Builds a vehicle mask from alpha when available, otherwise falls back to border-difference masking
- Estimates tilt from the bottom profile of the vehicle cutout
- Rotates the vehicle by the estimated correction angle
- Optionally crops back to the content after rotation

## Install
Copy `vehicle_auto_level.py` into:
`ComfyUI/custom_nodes/vehicle_auto_level_pack/`

Then restart ComfyUI.

## Recommended defaults
- `max_correction_degrees`: 7.0
- `trim_side_fraction`: 0.18
- `min_confidence`: 0.35

## Notes
This is a heuristic. It is best for mild roll/tilt correction on dealer photos. It will not fully solve camera-perspective mismatch.
