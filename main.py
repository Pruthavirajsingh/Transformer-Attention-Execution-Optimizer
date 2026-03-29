"""Run the attention execution optimizer."""

from __future__ import annotations

import sys
from pathlib import Path

# Allow running this script directly from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from attention_optimizer.optimizer import optimize_attention_execution
from attention_optimizer.strategies import StrategyConfig
from attention_optimizer.simulator import HardwareConfig


def main() -> None:
    # Example inputs. These can be adjusted as needed for experiments.
    sequence_length = 1024
    hidden_dim = 768
    memory_limit_bytes = 2 * 1024 * 1024 * 1024  # 2 GB

    hardware = HardwareConfig(
        compute_flops_per_sec=200e12,  # 200 TFLOPs/s
        memory_bandwidth_bytes_per_sec=900e9,  # 900 GB/s
        memory_limit_bytes=memory_limit_bytes,
    )

    config = StrategyConfig(
        sequence_length=sequence_length,
        hidden_dim=hidden_dim,
        hardware=hardware,
    )

    result = optimize_attention_execution(config)

    print("Best Strategy:")
    print(f"- Type: {result['strategy']}")
    print(f"- Tile Size: {result['tile_size']}")
    print(f"- Latency: {result['latency']}")
    print(f"- Memory: {result['memory']}")


if __name__ == "__main__":
    main()
