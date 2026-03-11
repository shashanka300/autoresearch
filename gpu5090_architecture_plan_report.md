# RTX 5090 Architecture Plan and Experiment Report

Date: 2026-03-10
Branch: `autoresearch/mar10b`

## 1) Operational Constraints (Single GPU)

- Hardware: 1x RTX 5090 class GPU (observed usable VRAM in logs: ~31.8 GB total).
- Runtime budget: fixed 5 minutes per run (from project design).
- Repo constraints: edit only `train.py`, no new dependencies, metric is `val_bpb` (lower is better).
- Practical throughput finding so far: `DEVICE_BATCH_SIZE=32` is stable (~22.6 GB peak VRAM), and improves total tokens over `16`.

Implication: prioritize low/medium-complexity architecture changes that preserve runtime stability and avoid large engineering overhead (e.g., full MoE/MLA rewrites) until quick wins are exhausted.

## 2) External Architecture Signals (from Raschka comparison)

The source highlights recurrent patterns across strong modern LLMs:

- SwiGLU/GeGLU-style gated MLPs are common.
- GQA is common (often with fewer KV heads than Q heads).
- RoPE variants (including partial/no-RoPE in some layers) are used.
- Sliding-window attention appears in several models, but exact pattern is model-specific.
- MoE/MLA and advanced attention variants can be powerful but are high complexity.

Source: https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison

## 3) Completed Experiment Report (Observed)

| Commit | Change | val_bpb | VRAM (GB) | Status | Observation |
|---|---|---:|---:|---|---|
| `7ccb112` | compile/triton baseline patch | 0.000000 | 0.0 | crash | Environment/tooling issue fixed by installing `uv`; not a model result. |
| `3949d45` | `DEVICE_BATCH_SIZE: 128 -> 16` | 0.001861 | 11.6 | keep | Solved OOM and established first valid baseline. |
| `45e43b4` | `DEVICE_BATCH_SIZE: 16 -> 32` | 0.001579 | 22.6 | keep | Better throughput (more tokens), better val_bpb. |
| `e28371e` | `WINDOW_PATTERN: SSSL -> SSSS` | 0.001677 | 22.6 | discard | Worse than best; full sliding harmed quality at same compute. |
| `d8444c6` | `WARMDOWN_RATIO: 0.5 -> 0.3` | 0.001440 | 22.6 | keep | Best result so far; longer high-LR phase helped. |
| `5093c95` | `WARMDOWN_RATIO: 0.3 -> 0.2` | 0.001440 | 22.6 | discard | Tie with best; no net gain, reverted. |

Current best config fingerprint:
- `DEVICE_BATCH_SIZE = 32`
- `WARMDOWN_RATIO = 0.3`
- `WINDOW_PATTERN = "SSSL"`

## 4) Recommended Architecture Plan for 5090

### Phase A: Fast Screening (single-factor, low complexity)

Goal: identify high-signal directions quickly under 5-minute runs.

1. Activation family
- A1: ReGLU -> SwiGLU
- A2: ReGLU -> GeGLU
- A3: ReGLU square ablation (`relu(x)` only)

2. Attention structure
- B1: GQA ratio (`n_kv_head = n_head // 2`)
- B2: GQA ratio (`n_kv_head = n_head // 4`)
- B3: attention projection bias on/off

3. Positioning
- C1: Partial RoPE (50% rotary channels)
- C2: RoPE base 50k
- C3: RoPE base 100k

4. Residual/value ablations
- D1: remove value embeddings
- D2: value embeddings on all layers
- D3: remove learned residual scalars

Decision rule for promotion:
- Promote variants with >= 0.0008 val_bpb improvement OR same val_bpb with simpler code/lower VRAM.

### Phase B: Pairwise Combinations (only from winners)

Goal: exploit interactions with minimal combinatorial blow-up.

Run only combinations of top 3 Phase-A winners:
- P1: best activation + best GQA ratio
- P2: best activation + best RoPE variant
- P3: best GQA ratio + best residual/value ablation
- P4: best activation + best GQA + best schedule tweak (`WARMDOWN_RATIO=0.3` retained)

Decision rule:
- Keep only strict improvements over current best (`0.001440`) with stable VRAM <= ~26 GB.

### Phase C: Medium Complexity (after clear gain plateau)

- M1: Pre+Post norm (Gemma-style)
- M2: per-head QK norm
- M3: attention sinks
- M4: 2-token prediction head (if implementation remains clean)

### Deferred (High Complexity; not first pass on one 5090)

- Full MoE (with load balancing)
- MLA/KV latent compression
- Gated DeltaNet or full linear-attention replacement

These are likely worthwhile later, but engineering and regression risk are high relative to current setup.

## 5) Combination Matrix to Execute Next

Use this fixed order to maximize information per GPU-hour:

1. SwiGLU
2. GeGLU
3. GQA 1/2
4. GQA 1/4
5. Partial RoPE 50%
6. RoPE base 50k
7. Remove value embeddings
8. Residual scalars fixed (1.0/0.0)
9. Best two-way combo from top-2 winners
10. Best three-way combo (if #9 improves)

## 6) Reporting Format for Each New Run

Record this for each experiment row:
- `commit`, `val_bpb`, `memory_gb`, `status`, `description`
- Plus a short note:
  - Throughput (`total_tokens_M`)
  - Stability (OOM/NaN/compile issue)
  - Simplicity delta (code +/- lines and complexity)

Template note:
- `obs: tokens=XX.XM, mfu=Y.Y%, stable=<yes/no>, complexity=<low/med/high>, verdict=<keep/discard>`

## 7) Immediate Next Best Bet

Given current evidence on this machine:
- First architectural bet: **SwiGLU** (high prior from modern models, low implementation risk).
- Second bet: **GQA reduction to n_kv_head = n_head // 2** (memory/throughput tradeoff likely favorable on 5090).
- Then combine both if either is positive.
