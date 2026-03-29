# Field Guide: What You Need to Know

This project sits at the intersection of machine learning systems and performance modeling.

## Core field concepts

### 1. Sequence length

Sequence length is the number of tokens processed by attention.

As sequence length grows:
- compute increases
- memory traffic increases
- latency usually increases

This is why long-context inference is so expensive.

### 2. Hidden dimension

Hidden dimension is the vector width used for each token.

It affects:
- FLOP count
- activation size
- memory bandwidth demand

Larger hidden dimensions increase both arithmetic work and data movement.

### 3. Memory bandwidth

Bandwidth is how fast data can move between memory and compute.

If bandwidth is limited, the system becomes memory-bound.

In attention workloads, this is often the real bottleneck.

### 4. Parallelism

Parallelism means multiple units can work at once.

In modeling terms:
- more parallel units reduce effective compute time
- memory time may still dominate

Parallelism helps, but it does not remove memory pressure.

### 5. Roofline thinking

Roofline modeling asks:
- is compute the bottleneck?
- or is memory the bottleneck?

The slower one is usually the one that matters most.

This is a practical mental model for understanding performance.

### 6. Calibration vs overfitting

Calibration adjusts the model so it is useful on real workloads.

Overfitting means tuning too much to one benchmark or one machine.

This project aims for calibration, not overfitting.

That distinction matters if you want the model to generalize beyond one test run.

### 7. Decision quality

For this project, the important thing is not exact milliseconds.

The important thing is:
- the best strategy is chosen correctly
- memory limits are respected
- trends match reality

That is what makes the model useful in practice.

## Practical takeaway

If you understand:
- attention mechanics
- memory bandwidth
- bottlenecks
- calibration

then you understand the main ideas behind this project.

That combination is the foundation of a strong ML systems project.
