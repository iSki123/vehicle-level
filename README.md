
# Vehicle Auto Level v2

This revision fixes the most common failure mode from v1: applying the estimated tilt in the wrong direction.

## What changed
- Added `invert_estimated_angle` boolean and set it to default `True`.
- Output angle is now the applied correction angle.

## Recommended settings
- max_correction_degrees: 4.0 to 6.0
- trim_side_fraction: 0.22 to 0.28 for cars
- min_confidence: 0.45
- invert_estimated_angle: True
- manual_angle_offset: use ±0.5 to ±2.0 only if needed
