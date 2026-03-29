"""Lightweight attention execution simulator.

The goal of this module is not to model hardware perfectly.
It provides a simple, consistent estimate of:
* compute cost
* memory cost
* estimated latency

The estimates are designed to preserve relative ranking between
attention strategies under the same hardware assumptions.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import log2


@dataclass(frozen=True)
class HardwareConfig:
    """Hardware characteristics used by the simulator."""

    compute_flops_per_sec: float
    memory_bandwidth_bytes_per_sec: float
    memory_limit_bytes: int
    compute_scale_factor: float = 1000.0
    memory_scale_factor: float = 1000.0
    parallel_units: int = 256
    base_latency_ms: float = 1.0
    compute_dominance_factor: float = 0.2
    global_latency_scale: float = 0.25


@dataclass(frozen=True)
class AttentionConfig:
    """Attention problem dimensions."""

    sequence_length: int
    hidden_dim: int
    tile_size: int | None = None


@dataclass(frozen=True)
class AttentionEstimate:
    """Estimated execution cost for an attention configuration."""

    compute_flops: float
    memory_bytes: int
    estimated_latency_ms: float

    @property
    def estimated_latency_sec(self) -> float:
        """Compatibility accessor for older code paths."""

        return self.estimated_latency_ms / 1000.0


def simulate_attention(
    config: AttentionConfig,
    hardware: HardwareConfig,
) -> AttentionEstimate:
    """Estimate attention compute, memory, and latency.

    The model is intentionally simple:
    - Compute cost includes QK^T, softmax, and attention-weighted V.
    - Memory cost approximates storage of Q, K, V, scores, and outputs.
    - Latency is a calibrated sum of compute time and memory time.
    - Units are milliseconds.
    """

    n = config.sequence_length
    d = config.hidden_dim
    tile_size = config.tile_size or n

    if n <= 0:
        raise ValueError("sequence_length must be positive")
    if d <= 0:
        raise ValueError("hidden_dim must be positive")
    if tile_size <= 0:
        raise ValueError("tile_size must be positive")
    if hardware.compute_flops_per_sec <= 0:
        raise ValueError("compute_flops_per_sec must be positive")
    if hardware.memory_bandwidth_bytes_per_sec <= 0:
        raise ValueError("memory_bandwidth_bytes_per_sec must be positive")
    if hardware.memory_limit_bytes <= 0:
        raise ValueError("memory_limit_bytes must be positive")
    if hardware.compute_scale_factor <= 0:
        raise ValueError("compute_scale_factor must be positive")
    if hardware.memory_scale_factor <= 0:
        raise ValueError("memory_scale_factor must be positive")
    if hardware.parallel_units <= 0:
        raise ValueError("parallel_units must be positive")
    if hardware.base_latency_ms < 0:
        raise ValueError("base_latency_ms must be non-negative")
    if hardware.compute_dominance_factor <= 0:
        raise ValueError("compute_dominance_factor must be positive")
    if hardware.global_latency_scale <= 0:
        raise ValueError("global_latency_scale must be positive")

    # Simple FLOP model for scaled dot-product attention.
    # QK^T: n*n*d multiply-adds
    # softmax: proportional to n*n
    # AV: n*n*d multiply-adds
    qk_cost = 2.0 * n * n * d
    softmax_cost = 5.0 * n * n
    av_cost = 2.0 * n * n * d
    compute_flops = qk_cost + softmax_cost + av_cost

    # Memory model in bytes. Assume float32 activations.
    # Modern attention implementations avoid materializing the full n^2
    # attention matrix and instead keep mostly n*d activations plus a
    # small working buffer for the current block/chunk.
    bytes_per_element = 4
    q_bytes = n * d * bytes_per_element
    kv_bytes = 2 * n * d * bytes_per_element
    output_bytes = n * d * bytes_per_element
    if config.tile_size is None:
        # Full attention still needs a somewhat larger temporary buffer,
        # but we do not model the full n^2 matrix in memory.
        working_buffer_bytes = n * d * bytes_per_element
    else:
        # Tiled or chunked execution keeps only a block-sized buffer live.
        working_buffer_bytes = n * tile_size * bytes_per_element

    memory_bytes = q_bytes + kv_bytes + output_bytes + working_buffer_bytes

    compute_time_ms = (compute_flops / hardware.compute_flops_per_sec) * 1000.0
    compute_time_ms = compute_time_ms / hardware.parallel_units

    # Larger inputs usually keep hardware busier and amortize overheads
    # better, so we model a simple deterministic efficiency boost.
    efficiency_factor = 1.0 + log2(n)
    compute_time_ms = compute_time_ms / efficiency_factor

    # Attention is typically memory-bound, not compute-bound, so we
    # reduce the influence of raw FLOP time during calibration.
    compute_time_ms = compute_time_ms * hardware.compute_dominance_factor

    memory_time_ms = (memory_bytes / hardware.memory_bandwidth_bytes_per_sec) * 1000.0
    # Real systems are usually limited by either compute or memory at a
    # given point in time, so we use the dominant bottleneck instead of
    # summing both contributions.
    raw_latency_ms = (
        hardware.base_latency_ms
        + max(
            compute_time_ms * hardware.compute_scale_factor,
            memory_time_ms * hardware.memory_scale_factor,
        )
    )

    # Final global calibration keeps the structure intact while aligning
    # the model with empirical measurements, without overfitting internals.
    estimated_latency_ms = hardware.global_latency_scale * raw_latency_ms

    return AttentionEstimate(
        compute_flops=compute_flops,
        memory_bytes=memory_bytes,
        estimated_latency_ms=estimated_latency_ms,
    )
