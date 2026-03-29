# Usage Guide

## Run the optimizer demo

```powershell
python main.py
```

This prints the best strategy for the default example configuration.

## Run the strategy validation script

```powershell
python validate_strategies.py
```

This checks the expected memory ordering:
- full memory > tiled memory
- tiled memory > chunked memory

## Run the benchmark

```powershell
python benchmark_attention.py
```

This compares simulator latency with real PyTorch execution for:
- 128
- 512
- 1024

## Where the code lives

- `src/attention_optimizer/simulator.py`
- `src/attention_optimizer/strategies.py`
- `src/attention_optimizer/optimizer.py`

