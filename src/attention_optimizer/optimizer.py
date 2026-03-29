"""Grid-search optimizer for attention execution strategies."""

from __future__ import annotations

from typing import Any

from .strategies import (
    StrategyConfig,
    chunked_attention_strategy,
    full_attention_strategy,
    tiled_attention_strategy,
)


def optimize_attention_execution(
    config: StrategyConfig,
    tile_sizes: list[int] | None = None,
) -> dict[str, Any]:
    """Find the lowest-latency strategy that fits in memory.

    The search is intentionally simple:
    - evaluate full attention once
    - evaluate tiled attention for a small set of tile sizes
    - evaluate chunked attention for the same tile sizes
    - skip any candidate whose memory exceeds the hardware limit
    - return the feasible candidate with the lowest estimated latency
    """

    if tile_sizes is None:
        tile_sizes = [16, 32, 64, 128]

    best_result = None

    def consider_candidate(result: Any) -> None:
        nonlocal best_result

        # Skip candidates that do not fit in memory.
        if result.memory_bytes > config.hardware.memory_limit_bytes:
            return

        if best_result is None or result.estimated_latency_sec < best_result.estimated_latency_sec:
            best_result = result

    # Try full attention first.
    consider_candidate(full_attention_strategy(config))

    # Try all tiled and chunked configurations using the provided tile sizes.
    for tile_size in tile_sizes:
        consider_candidate(tiled_attention_strategy(config, tile_size))
        consider_candidate(chunked_attention_strategy(config, tile_size))

    if best_result is None:
        raise ValueError("No attention strategy fits within the memory limit")

    return {
        "strategy": best_result.strategy_type,
        "tile_size": best_result.tile_size,
        "latency": best_result.estimated_latency_sec,
        "memory": best_result.memory_bytes,
    }
