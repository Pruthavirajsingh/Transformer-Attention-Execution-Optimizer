"""Benchmark simulated attention latency against real PyTorch execution."""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import torch

# Allow running this script directly from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from attention_optimizer.simulator import AttentionConfig, HardwareConfig, simulate_attention


def run_attention(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Compute standard scaled dot-product attention on CPU."""

    d = q.shape[-1]
    scores = torch.matmul(q, k.transpose(-2, -1)) / (d ** 0.5)
    weights = torch.softmax(scores, dim=-1)
    output = torch.matmul(weights, v)
    return output


def benchmark_sequence_length(sequence_length: int, hidden_dim: int) -> None:
    """Print simulator vs real execution time for one input size."""

    hardware = HardwareConfig(
        compute_flops_per_sec=200e12,
        memory_bandwidth_bytes_per_sec=900e9,
        memory_limit_bytes=2 * 1024 * 1024 * 1024,
    )

    sim_config = AttentionConfig(
        sequence_length=sequence_length,
        hidden_dim=hidden_dim,
        tile_size=None,
    )
    sim_result = simulate_attention(sim_config, hardware)

    torch.manual_seed(0)
    q = torch.randn(sequence_length, hidden_dim, dtype=torch.float32)
    k = torch.randn(sequence_length, hidden_dim, dtype=torch.float32)
    v = torch.randn(sequence_length, hidden_dim, dtype=torch.float32)

    start = perf_counter()
    _ = run_attention(q, k, v)
    end = perf_counter()

    actual_latency_ms = (end - start) * 1000.0
    simulator_latency_ms = sim_result.estimated_latency_sec * 1000.0

    print(f"Sequence Length: {sequence_length}")
    print(f"Simulator Latency: {simulator_latency_ms:.4f} ms")
    print(f"Actual Latency: {actual_latency_ms:.4f} ms")
    print()


def main() -> None:
    hidden_dim = 768
    for sequence_length in [128, 512, 1024]:
        benchmark_sequence_length(sequence_length, hidden_dim)


if __name__ == "__main__":
    main()
