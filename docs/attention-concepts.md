# Attention Concepts

This file explains the core ideas you need to understand this project.

## 1. What attention does

Attention lets a model compare tokens with each other and decide which tokens matter most.

In simplified form:
- Q = query
- K = key
- V = value

The attention operation compares Q and K, turns the scores into weights, and uses those weights to combine V.

## 2. Why attention is expensive

Attention becomes expensive because:
- sequence length increases
- pairwise token interactions grow quickly
- memory traffic becomes large

The naive attention matrix has size `n x n`, which is costly for long sequences.

## 3. Why memory matters

In practice, attention is often memory-bound.

That means:
- the hardware may spend more time moving data than computing math
- memory bandwidth can dominate latency

This is why the simulator focuses on memory usage and bandwidth, not just FLOPs.

## 4. Full attention

Full attention is the simplest form:
- compute all token pairs
- highest memory use
- usually the least efficient choice under tight constraints

## 5. Tiled attention

Tiled attention splits the sequence into blocks.

Benefits:
- lower peak memory
- better cache behavior

Tradeoff:
- some compute overhead from managing blocks

## 6. Chunked attention

Chunked attention processes the sequence in smaller pieces and reduces working-set size.

Benefits:
- lower memory than tiled in this project model
- often a better fit when memory is tight

Tradeoff:
- a little more overhead than tiled attention

## 7. Why this project uses approximations

The project is intentionally simple.

It does not simulate:
- exact GPU kernels
- every cache level
- scheduler behavior
- tensor core details

Instead it uses a compact model that keeps the relative ranking useful.

