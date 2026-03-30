# Dashboard Guide

The Streamlit dashboard is the interactive face of the project.

## Purpose

It lets you explore the optimizer without touching the command line.

The dashboard is useful for:
- quick demos
- sanity checks
- comparing strategies
- showing scaling behavior visually

## Inputs

The sidebar includes:
- sequence length
- hidden dimension
- memory limit in MB

These feed directly into the existing optimizer.

## Main sections

### Best Strategy

Shows the selected strategy, tile size, latency, and memory usage.

### Strategy Comparison

Shows all three strategies side by side for the same input.

### Scaling Behavior

Shows how the chosen strategy latency changes as sequence length increases.

## Why it helps

The dashboard makes the model easier to explain.

It shows:
- how memory limits affect the answer
- why one strategy is chosen over another
- how performance changes with scale

## Run it

```powershell
streamlit run app.py
```
