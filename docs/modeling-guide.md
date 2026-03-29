# Modeling Guide

This project uses a lightweight performance model that is intentionally simple, explainable, and stable under calibration.

## The modeling goal

The goal is not perfect prediction.
The goal is to:
- rank strategies correctly
- respect memory limits
- behave reasonably as sequence length changes

That is the right level of fidelity for a strategy-selection tool.

## Key model parts

### Compute cost

The simulator estimates attention compute from the main operations:
- QK^T
- softmax
- V multiplication

This gives a rough FLOP estimate that captures the dominant arithmetic work without pretending to be a kernel-level profiler.

### Memory cost

The simulator estimates memory from:
- Q, K, V
- output
- a working buffer

The memory model avoids relying on the full naive `n x n` attention matrix, because practical implementations stream and reuse data instead of materializing everything at once.

### Roofline idea

The simulator uses a roofline-style bottleneck model:
- compute time and memory time are both estimated
- the dominant bottleneck drives latency

That reflects real systems better than adding the two costs directly, because attention execution is usually limited by one bottleneck at a time.

### Calibration

The simulator also includes calibration factors:
- parallelism
- efficiency scaling with sequence length
- a memory-vs-compute dominance correction
- a global scaling factor

These are not meant to be perfect physics.
They are meant to keep the model useful, stable, and aligned with measured behavior.

## Why calibration matters

Analytical models rarely match real measurements exactly.

Calibration helps the model:
- stay in the right range
- match real trends
- remain useful for decisions

In systems work, that is often more valuable than exact numeric prediction.

## What the optimizer trusts

The optimizer trusts relative comparisons:
- which strategy is faster
- which strategy fits memory
- which tile size is better

That is the right level of fidelity for this project because the output is a decision, not a detailed hardware report.
