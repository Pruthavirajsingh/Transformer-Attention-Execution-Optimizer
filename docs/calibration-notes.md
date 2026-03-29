# Calibration Notes

This project went through several modeling corrections.

## What was corrected

### 1. Sequential compute assumption

The simulator originally treated compute too sequentially.

It now includes:
- parallel units
- efficiency scaling with sequence length

### 2. Memory scaling

The simulator originally overused `n^2` memory scaling.

It now models a more practical working set:
- `n x d` activations
- a small block buffer

### 3. Bottleneck logic

The simulator originally summed compute and memory too directly.

It now uses a roofline-style bottleneck:
- the larger of compute or memory dominates

### 4. Global calibration

A final global scale factor aligns simulated latency with measured latency.

## Why this is a good approach

This keeps the structure correct while letting the model stay close to reality.

It is a practical systems approach:
- model the right shape
- validate against measurements
- calibrate the result

