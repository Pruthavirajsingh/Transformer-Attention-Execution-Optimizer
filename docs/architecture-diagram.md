# Architecture Diagram

```mermaid
flowchart LR
    A[Input config\nsequence length, hidden dim,\nhardware limits] --> B[Simulator]
    A --> C[Strategies]
    B --> D[Optimizer]
    C --> D
    D --> E[Best strategy\nfull / tiled / chunked]

    F[PyTorch benchmark] --> B
    F --> G[Calibration]
    G --> B
```

## How the pieces fit together

- The simulator estimates compute, memory, and latency.
- The strategies module defines the candidate execution plans.
- The optimizer evaluates those candidates and chooses the best feasible one.
- The benchmark script compares the simulator against real PyTorch execution.
- Calibration keeps the model aligned with real measurements without overfitting individual components.

