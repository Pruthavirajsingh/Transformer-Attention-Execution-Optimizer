"""Basic sanity checks for attention strategy ranking."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this script directly from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from attention_optimizer.simulator import HardwareConfig
from attention_optimizer.strategies import (
    StrategyConfig,
    chunked_attention_strategy,
    full_attention_strategy,
    tiled_attention_strategy,
)


def validate_strategy_ranking() -> None:
    """Run all strategies and verify the expected memory ordering."""

    config = StrategyConfig(
        sequence_length=1024,
        hidden_dim=768,
        hardware=HardwareConfig(
            compute_flops_per_sec=200e12,
            memory_bandwidth_bytes_per_sec=900e9,
            memory_limit_bytes=2 * 1024 * 1024 * 1024,
        ),
    )

    full_result = full_attention_strategy(config)
    tiled_result = tiled_attention_strategy(config, tile_size=64)
    chunked_result = chunked_attention_strategy(config, tile_size=64)

    print(f"Full: latency {full_result.estimated_latency_sec}, memory {full_result.memory_bytes}")
    print(f"Tiled: latency {tiled_result.estimated_latency_sec}, memory {tiled_result.memory_bytes}")
    print(
        f"Chunked: latency {chunked_result.estimated_latency_sec}, "
        f"memory {chunked_result.memory_bytes}"
    )

    assert full_result.memory_bytes > tiled_result.memory_bytes, "Expected full memory > tiled memory"
    assert tiled_result.memory_bytes > chunked_result.memory_bytes, "Expected tiled memory > chunked memory"


if __name__ == "__main__":
    validate_strategy_ranking()
    print("Strategy ranking checks passed.")
