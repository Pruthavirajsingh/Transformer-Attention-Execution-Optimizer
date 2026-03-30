# Limitations and Future Work

This project is intentionally lightweight. It is designed to be useful for strategy selection and systems reasoning, not to act as a full compiler or kernel-level performance model.

## Current limitations

### 1. The simulator is approximate

The latency model is calibrated to be directionally correct and useful for ranking strategies.

It is not a cycle-accurate hardware simulator.

### 2. Hardware behavior is simplified

The model does not explicitly simulate:
- cache hierarchies
- warp scheduling
- tensor core behavior
- kernel launch overhead in detail

These effects matter in real systems, but they are outside the scope of this project.

### 3. Strategy models are analytical

The full, tiled, and chunked strategies are represented using simple formulas.

They are not real CUDA, Triton, or PyTorch kernel implementations.

### 4. Memory modeling is intentionally coarse

The project captures the main memory trends:
- activation size
- working buffers
- block-based reuse

It does not model every temporary tensor or every allocator detail.

### 5. Benchmarking is CPU-only

The benchmark script compares against PyTorch on CPU.

That is useful for validation, but it does not represent GPU behavior directly.

### 6. The dashboard is an interface layer

The Streamlit app presents the existing optimizer in a friendlier format.

It does not change the underlying model or search logic.

## Future work

### 1. Add GPU benchmarking

Benchmark the same attention paths on CUDA hardware to compare against realistic inference conditions.

### 2. Support more execution strategies

Extend the search space with:
- sliding-window attention
- paged attention
- sparse attention variants
- fused attention kernels

### 3. Improve calibration data

Use multiple benchmark shapes and hardware profiles to make the calibration more robust.

### 4. Model complete transformer blocks

Extend beyond attention to include:
- MLP layers
- layer normalization
- residual connections
- KV-cache reuse across decoding steps

### 5. Add smarter search

Replace the simple grid search with a more adaptive method if needed:
- heuristic search
- Bayesian optimization
- learned cost model

### 6. Add richer reporting

Future versions could output:
- strategy comparisons
- memory breakdowns
- latency breakdowns
- recommended configurations across multiple sequence lengths

## Design principle

The main design rule for future work is to keep the system explainable.

Any added feature should improve decision quality without turning the project into an overengineered framework.
