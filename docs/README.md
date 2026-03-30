# Documentation Index

This folder explains the project, the modeling choices, and the field concepts behind attention optimization.

## What the system does

The project models transformer attention under memory constraints and selects the best execution strategy with a simple search loop.

It compares:
- full attention
- tiled attention
- chunked attention

It estimates:
- latency
- memory usage
- feasibility under the available memory budget

## Recommended reading order

1. Read [project overview](project-overview.md) for the system-level summary.
2. Read [architecture diagram](architecture-diagram.md) to see how the pieces connect.
3. Read [attention concepts](attention-concepts.md) for the domain fundamentals.
4. Read [modeling guide](modeling-guide.md) for the performance model.
5. Read [field guide](field-guide.md) for the systems concepts that matter most.
6. Read [dashboard guide](dashboard-guide.md) for the Streamlit interface.
7. Read [dashboard screenshots](images/) for preview images.
8. Read [usage guide](usage-guide.md) for run instructions.
