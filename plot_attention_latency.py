"""Create a simple SVG plot for simulated vs actual attention latency."""

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
    return torch.matmul(weights, v)


def collect_data(sequence_lengths: list[int], hidden_dim: int) -> tuple[list[int], list[float], list[float]]:
    """Collect simulator and actual timings for a few sequence lengths."""

    hardware = HardwareConfig(
        compute_flops_per_sec=200e12,
        memory_bandwidth_bytes_per_sec=900e9,
        memory_limit_bytes=2 * 1024 * 1024 * 1024,
    )

    simulator_ms: list[float] = []
    actual_ms: list[float] = []

    for sequence_length in sequence_lengths:
        sim_config = AttentionConfig(sequence_length=sequence_length, hidden_dim=hidden_dim)
        sim_result = simulate_attention(sim_config, hardware)

        torch.manual_seed(0)
        q = torch.randn(sequence_length, hidden_dim, dtype=torch.float32)
        k = torch.randn(sequence_length, hidden_dim, dtype=torch.float32)
        v = torch.randn(sequence_length, hidden_dim, dtype=torch.float32)

        start = perf_counter()
        _ = run_attention(q, k, v)
        end = perf_counter()

        simulator_ms.append(sim_result.estimated_latency_sec * 1000.0)
        actual_ms.append((end - start) * 1000.0)

    return sequence_lengths, simulator_ms, actual_ms


def scale_point(value: float, min_value: float, max_value: float, start: float, end: float) -> float:
    """Map a numeric value to a screen coordinate."""

    if max_value == min_value:
        return (start + end) / 2.0
    ratio = (value - min_value) / (max_value - min_value)
    return start + ratio * (end - start)


def make_svg(sequence_lengths: list[int], simulator_ms: list[float], actual_ms: list[float]) -> str:
    """Render a minimal SVG line plot."""

    width = 900
    height = 540
    margin_left = 90
    margin_right = 40
    margin_top = 40
    margin_bottom = 80

    plot_width = width - margin_left - margin_right
    plot_height = height - margin_top - margin_bottom

    x_min = min(sequence_lengths)
    x_max = max(sequence_lengths)
    y_values = simulator_ms + actual_ms
    y_min = 0.0
    y_max = max(y_values) * 1.15

    def points(values: list[float]) -> str:
        coords = []
        for x, y in zip(sequence_lengths, values):
            sx = scale_point(float(x), float(x_min), float(x_max), margin_left, margin_left + plot_width)
            sy = scale_point(y, y_min, y_max, margin_top + plot_height, margin_top)
            coords.append(f"{sx:.1f},{sy:.1f}")
        return " ".join(coords)

    x_ticks = sequence_lengths
    y_ticks = [0.0, y_max * 0.25, y_max * 0.5, y_max * 0.75, y_max]

    axis_color = "#334155"
    sim_color = "#2563eb"
    actual_color = "#ea580c"
    grid_color = "#cbd5e1"

    lines = []
    lines.append('<?xml version="1.0" encoding="UTF-8"?>')
    lines.append(
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">'
    )
    lines.append('<rect width="100%" height="100%" fill="white"/>')
    lines.append(f'<text x="{margin_left}" y="26" font-size="22" font-family="Arial" fill="{axis_color}">Attention Latency: Simulator vs Actual</text>')

    # Grid and Y labels.
    for y_val in y_ticks:
        sy = scale_point(y_val, y_min, y_max, margin_top + plot_height, margin_top)
        lines.append(f'<line x1="{margin_left}" y1="{sy:.1f}" x2="{margin_left + plot_width}" y2="{sy:.1f}" stroke="{grid_color}" stroke-width="1"/>')
        lines.append(f'<text x="{margin_left - 12}" y="{sy + 4:.1f}" font-size="12" font-family="Arial" fill="{axis_color}" text-anchor="end">{y_val:.1f}</text>')

    # Axes.
    lines.append(f'<line x1="{margin_left}" y1="{margin_top}" x2="{margin_left}" y2="{margin_top + plot_height}" stroke="{axis_color}" stroke-width="2"/>')
    lines.append(f'<line x1="{margin_left}" y1="{margin_top + plot_height}" x2="{margin_left + plot_width}" y2="{margin_top + plot_height}" stroke="{axis_color}" stroke-width="2"/>')

    # X ticks and labels.
    for x_val in x_ticks:
        sx = scale_point(float(x_val), float(x_min), float(x_max), margin_left, margin_left + plot_width)
        lines.append(f'<line x1="{sx:.1f}" y1="{margin_top + plot_height}" x2="{sx:.1f}" y2="{margin_top + plot_height + 6}" stroke="{axis_color}" stroke-width="2"/>')
        lines.append(f'<text x="{sx:.1f}" y="{margin_top + plot_height + 24}" font-size="12" font-family="Arial" fill="{axis_color}" text-anchor="middle">{x_val}</text>')

    lines.append(f'<text x="{width / 2:.1f}" y="{height - 24}" font-size="14" font-family="Arial" fill="{axis_color}" text-anchor="middle">Sequence Length</text>')
    lines.append(
        f'<text x="24" y="{height / 2:.1f}" font-size="14" font-family="Arial" fill="{axis_color}" text-anchor="middle" transform="rotate(-90 24 {height / 2:.1f})">Latency (ms)</text>'
    )

    # Lines and points.
    lines.append(f'<polyline fill="none" stroke="{sim_color}" stroke-width="3" points="{points(simulator_ms)}"/>')
    lines.append(f'<polyline fill="none" stroke="{actual_color}" stroke-width="3" points="{points(actual_ms)}"/>')

    for value, color in ((simulator_ms, sim_color), (actual_ms, actual_color)):
        for x_val, y_val in zip(sequence_lengths, value):
            sx = scale_point(float(x_val), float(x_min), float(x_max), margin_left, margin_left + plot_width)
            sy = scale_point(y_val, y_min, y_max, margin_top + plot_height, margin_top)
            lines.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="5" fill="{color}"/>')

    # Legend.
    legend_x = margin_left + plot_width - 180
    legend_y = margin_top + 10
    lines.append(f'<rect x="{legend_x}" y="{legend_y}" width="160" height="64" rx="8" fill="#f8fafc" stroke="{grid_color}"/>')
    lines.append(f'<line x1="{legend_x + 14}" y1="{legend_y + 22}" x2="{legend_x + 48}" y2="{legend_y + 22}" stroke="{sim_color}" stroke-width="3"/>')
    lines.append(f'<text x="{legend_x + 56}" y="{legend_y + 26}" font-size="12" font-family="Arial" fill="{axis_color}">Simulator</text>')
    lines.append(f'<line x1="{legend_x + 14}" y1="{legend_y + 44}" x2="{legend_x + 48}" y2="{legend_y + 44}" stroke="{actual_color}" stroke-width="3"/>')
    lines.append(f'<text x="{legend_x + 56}" y="{legend_y + 48}" font-size="12" font-family="Arial" fill="{axis_color}">Actual PyTorch</text>')

    lines.append("</svg>")
    return "\n".join(lines)


def main() -> None:
    sequence_lengths, simulator_ms, actual_ms = collect_data([128, 512, 1024], hidden_dim=768)
    svg = make_svg(sequence_lengths, simulator_ms, actual_ms)

    output_path = Path(__file__).resolve().parent / "docs" / "attention_latency.svg"
    output_path.write_text(svg, encoding="utf-8")
    print(f"Saved plot to {output_path}")


if __name__ == "__main__":
    main()
