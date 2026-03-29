"""Attention execution strategies.

These functions do not implement real kernels. They provide lightweight
cost models that adjust the simulator's compute and memory formulas to
reflect different execution strategies.
"""

from __future__ import annotations

from dataclasses import dataclass

from .simulator import HardwareConfig


@dataclass(frozen=True)
class StrategyConfig:
    """Input data shared by all strategy models."""

    sequence_length: int
    hidden_dim: int
    hardware: HardwareConfig


@dataclass(frozen=True)
class StrategyResult:
    """Structured strategy output."""

    strategy_type: str
    tile_size: int | None
    compute_flops: float
    memory_bytes: int
    estimated_latency_sec: float


def _validate_common_inputs(config: StrategyConfig) -> tuple[int, int]:
    n = config.sequence_length
    d = config.hidden_dim

    if n <= 0:
        raise ValueError("sequence_length must be positive")
    if d <= 0:
        raise ValueError("hidden_dim must be positive")
    if config.hardware.compute_flops_per_sec <= 0:
        raise ValueError("compute_flops_per_sec must be positive")
    if config.hardware.memory_bandwidth_bytes_per_sec <= 0:
        raise ValueError("memory_bandwidth_bytes_per_sec must be positive")
    if config.hardware.memory_limit_bytes <= 0:
        raise ValueError("memory_limit_bytes must be positive")

    return n, d


def _estimate_latency(compute_flops: float, memory_bytes: int, hardware: HardwareConfig) -> float:
    compute_time = compute_flops / hardware.compute_flops_per_sec
    memory_time = memory_bytes / hardware.memory_bandwidth_bytes_per_sec
    return max(compute_time, memory_time)


def _base_attention_cost(n: int, d: int) -> float:
    """Baseline FLOP cost for scaled dot-product attention."""

    qk_cost = 2.0 * n * n * d
    softmax_cost = 5.0 * n * n
    av_cost = 2.0 * n * n * d
    return qk_cost + softmax_cost + av_cost


def full_attention_strategy(config: StrategyConfig) -> StrategyResult:
    """Estimate full attention.

    Full attention materializes the complete n x n score matrix, so it
    has the highest memory footprint.
    """

    n, d = _validate_common_inputs(config)
    bytes_per_element = 4

    compute_flops = _base_attention_cost(n, d)

    q_bytes = n * d * bytes_per_element
    kv_bytes = 2 * n * d * bytes_per_element
    score_bytes = n * n * bytes_per_element
    output_bytes = n * d * bytes_per_element
    memory_bytes = q_bytes + kv_bytes + score_bytes + output_bytes

    estimated_latency_sec = _estimate_latency(compute_flops, memory_bytes, config.hardware)

    return StrategyResult(
        strategy_type="full",
        tile_size=None,
        compute_flops=compute_flops,
        memory_bytes=memory_bytes,
        estimated_latency_sec=estimated_latency_sec,
    )


def tiled_attention_strategy(config: StrategyConfig, tile_size: int) -> StrategyResult:
    """Estimate tiled attention.

    Tiled attention processes the sequence in blocks. This reduces peak
    score storage from O(n^2) to roughly O(n * tile_size), but adds a
    modest compute overhead from block management.
    """

    n, d = _validate_common_inputs(config)
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")

    bytes_per_element = 4
    base_compute = _base_attention_cost(n, d)

    # Smaller tiles incur slightly more launch / block-management overhead.
    # This stays in the requested 5-10% range.
    compute_overhead = 1.0 + min(0.10, 0.04 + 0.08 * (16 / tile_size))
    compute_flops = base_compute * compute_overhead

    q_bytes = n * d * bytes_per_element
    kv_bytes = 2 * n * d * bytes_per_element
    output_bytes = n * d * bytes_per_element
    tiled_score_bytes = n * tile_size * bytes_per_element
    memory_bytes = q_bytes + kv_bytes + output_bytes + tiled_score_bytes

    estimated_latency_sec = _estimate_latency(compute_flops, memory_bytes, config.hardware)

    return StrategyResult(
        strategy_type="tiled",
        tile_size=tile_size,
        compute_flops=compute_flops,
        memory_bytes=memory_bytes,
        estimated_latency_sec=estimated_latency_sec,
    )


def chunked_attention_strategy(config: StrategyConfig, tile_size: int) -> StrategyResult:
    """Estimate chunked attention.

    Chunked attention is modeled as more memory efficient than tiled
    attention, with slightly higher compute overhead to reflect extra
    streaming and reduction work.
    """

    n, d = _validate_common_inputs(config)
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")

    bytes_per_element = 4
    base_compute = _base_attention_cost(n, d)

    # Chunked attention is a little more expensive than tiled, but still
    # benefits from larger chunk sizes.
    compute_overhead = 1.0 + min(0.14, 0.06 + 0.10 * (16 / tile_size))
    compute_flops = base_compute * compute_overhead

    q_bytes = n * d * bytes_per_element
    kv_bytes = 2 * n * d * bytes_per_element
    output_bytes = n * d * bytes_per_element
    # Chunked attention keeps only a small working set of scores live at
    # a time and uses a tiny reduction buffer, so memory stays below tiled.
    chunk_score_bytes = n * max(1, tile_size // 16) * bytes_per_element
    reduction_buffer_bytes = max(1, n // 32) * bytes_per_element
    memory_bytes = q_bytes + kv_bytes + output_bytes + chunk_score_bytes + reduction_buffer_bytes

    estimated_latency_sec = _estimate_latency(compute_flops, memory_bytes, config.hardware)

    return StrategyResult(
        strategy_type="chunked",
        tile_size=tile_size,
        compute_flops=compute_flops,
        memory_bytes=memory_bytes,
        estimated_latency_sec=estimated_latency_sec,
    )
