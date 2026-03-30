"""Streamlit dashboard for the attention optimizer."""

from __future__ import annotations

import sys
from pathlib import Path

import streamlit as st

# Allow running this app from the project root.
sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))

from attention_optimizer.optimizer import optimize_attention_execution
from attention_optimizer.simulator import HardwareConfig
from attention_optimizer.strategies import (
    StrategyConfig,
    chunked_attention_strategy,
    full_attention_strategy,
    tiled_attention_strategy,
)


def format_latency_ms(latency_seconds: float) -> float:
    """Convert seconds to milliseconds for display."""

    return latency_seconds * 1000.0


def format_memory_mb(memory_bytes: int) -> float:
    """Convert bytes to megabytes for display."""

    return memory_bytes / (1024 * 1024)


def build_strategy_rows(config: StrategyConfig, tile_size: int) -> list[dict[str, str]]:
    """Evaluate all strategies for the same input and mark the best one."""

    results = [
        full_attention_strategy(config),
        tiled_attention_strategy(config, tile_size),
        chunked_attention_strategy(config, tile_size),
    ]

    best = min(results, key=lambda item: item.estimated_latency_sec)
    rows: list[dict[str, str]] = []

    for result in results:
        is_best = result == best
        rows.append(
            {
                "Strategy": f"**{result.strategy_type.title()}**" if is_best else result.strategy_type.title(),
                "Latency (ms)": f"{format_latency_ms(result.estimated_latency_sec):.4f}",
                "Memory (MB)": f"{format_memory_mb(result.memory_bytes):.2f}",
                "Best": "Yes" if is_best else "",
            }
        )

    return rows


def collect_latency_curve(hidden_dim: int, memory_limit_bytes: int) -> tuple[list[int], list[float]]:
    """Run the optimizer across a few sequence lengths."""

    sequence_lengths = [128, 256, 512, 1024, 2048]
    latencies_ms: list[float] = []

    hardware = HardwareConfig(
        compute_flops_per_sec=200e12,
        memory_bandwidth_bytes_per_sec=900e9,
        memory_limit_bytes=memory_limit_bytes,
    )

    for sequence_length in sequence_lengths:
        config = StrategyConfig(
            sequence_length=sequence_length,
            hidden_dim=hidden_dim,
            hardware=hardware,
        )
        result = optimize_attention_execution(config)
        latencies_ms.append(format_latency_ms(result["latency"]))

    return sequence_lengths, latencies_ms


st.set_page_config(page_title="Transformer Attention Optimizer", page_icon="⚙️", layout="centered")

st.title("Transformer Attention Optimizer")
st.caption("Estimate the best attention strategy under memory constraints.")

with st.sidebar:
    st.header("Inputs")
    sequence_length = st.slider("Sequence length", min_value=128, max_value=4096, value=1024, step=128)
    hidden_dim = st.slider("Hidden dimension", min_value=64, max_value=1024, value=768, step=64)
    memory_limit_mb = st.slider("Memory limit (MB)", min_value=10, max_value=1000, value=256, step=10)

memory_limit_bytes = memory_limit_mb * 1024 * 1024

hardware = HardwareConfig(
    compute_flops_per_sec=200e12,
    memory_bandwidth_bytes_per_sec=900e9,
    memory_limit_bytes=memory_limit_bytes,
)

config = StrategyConfig(
    sequence_length=sequence_length,
    hidden_dim=hidden_dim,
    hardware=hardware,
)

try:
    result = optimize_attention_execution(config)
    comparison_tile_size = result["tile_size"] or 64
    strategy_rows = build_strategy_rows(config, comparison_tile_size)

    latency_ms = format_latency_ms(result["latency"])
    memory_mb = format_memory_mb(result["memory"])

    st.header("Best Strategy")
    col1, col2 = st.columns(2)
    col1.metric("Best Strategy", result["strategy"].title())
    col2.metric("Tile Size", "N/A" if result["tile_size"] is None else str(result["tile_size"]))

    metric_col1, metric_col2 = st.columns(2)
    metric_col1.metric("Latency", f"{latency_ms:.2f} ms")
    metric_col2.metric("Memory Usage", f"{memory_mb:.2f} MB")

    st.info(
        {
            "full": "Chosen for fastest compute under sufficient memory.",
            "tiled": "Chosen due to memory constraints.",
            "chunked": "Chosen for optimal memory efficiency.",
        }[result["strategy"]]
    )

    st.divider()
    st.header("Strategy Comparison")
    st.markdown("Best strategy is highlighted in the table below.")
    st.markdown(
        "| Strategy | Latency (ms) | Memory (MB) | Best |\n"
        "| --- | ---: | ---: | --- |\n"
        + "\n".join(
            f"| {row['Strategy']} | {row['Latency (ms)']} | {row['Memory (MB)']} | {row['Best']} |"
            for row in strategy_rows
        )
    )

    st.markdown(
        """
        - **Full**: baseline attention with the largest memory footprint.
        - **Tiled**: block-based attention with lower working-set pressure.
        - **Chunked**: the most memory-efficient option in this model.
        """
    )

    st.divider()
    st.header("Scaling Behavior")

    st.caption("Best-strategy latency as sequence length increases.")

    try:
        import matplotlib.pyplot as plt

        curve_lengths, curve_latencies = collect_latency_curve(hidden_dim, memory_limit_bytes)
        fig, ax = plt.subplots(figsize=(8, 4.5))
        ax.plot(curve_lengths, curve_latencies, marker="o", linewidth=2, color="#2563eb")
        ax.set_xlabel("Sequence Length")
        ax.set_ylabel("Latency (ms)")
        ax.set_title("Latency vs Sequence Length")
        ax.grid(True, alpha=0.3)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    except ImportError:
        st.info("Install matplotlib to see the latency scaling chart: `pip install matplotlib`.")

    st.write("### Current configuration")
    st.write(f"- Sequence length: {sequence_length}")
    st.write(f"- Hidden dimension: {hidden_dim}")
    st.write(f"- Memory limit: {memory_limit_mb} MB")

    st.divider()
    st.header("Why this matters")
    st.write("Lower latency improves user experience for interactive models and real-time applications.")
    st.write("Lower memory usage reduces deployment cost and makes larger workloads easier to serve.")
    st.write("Better optimization improves scalability as sequence length and model demand grow.")

except ValueError as exc:
    st.error(str(exc))
