# Project Overview

## Goal

This project answers a focused systems question:

> Given an attention workload and hardware limits, which execution strategy is best?

The goal is not to build a compiler or a kernel library. It is a compact decision system that models the tradeoffs clearly enough to be useful.

## Main parts

### 1. Simulator

The simulator estimates:
- compute cost
- memory cost
- latency

It is designed to preserve ranking between strategies rather than predict exact timings. That makes it suitable for strategy selection and comparison.

### 2. Strategies

The strategies module models three approaches:
- `full_attention_strategy`
- `tiled_attention_strategy`
- `chunked_attention_strategy`

Each one changes the compute and memory formulas in a lightweight way so the model reflects the main execution tradeoffs.

### 3. Optimizer

The optimizer does a simple grid search:
- try all strategies
- try a few tile sizes
- skip options that exceed memory limit
- choose the lowest latency feasible option

This keeps the search transparent and easy to reason about.

### 4. Scripts

The repository includes:
- `main.py` for one optimizer run
- `app.py` for the Streamlit dashboard
- `validate_strategies.py` for sanity checks
- `benchmark_attention.py` for comparison with real PyTorch execution
- `plot_attention_latency.py` for a simple latency plot

These scripts let you validate the model from three angles:
- behavior
- ranking
- empirical calibration
