# Current Chosen Configuration

Date: 2026-03-10
Branch: autoresearch/mar10b

Best run commit: 9c5066b
Best val_bpb: 0.001356
Peak memory: 19.2 GB
Status: keep

## Exact config fingerprint
- DEVICE_BATCH_SIZE = 32
- WARMDOWN_RATIO = 0.3
- WINDOW_PATTERN = "SSSL"
- MLP = SwiGLU
- GQA = 4:1
- Partial RoPE = yes
- RoPE base = 50000
- Value embeddings = on
- Residual scalars = learned
- QKV bias = off
- Optimizer vector-param fix = off

## Active Branch Note
- Current HEAD includes an optimizer safety fix for vector params (`optimizer_vector_fix=on`).
- This safety fix did not improve quality by itself (`a18e376`: 0.001383), but is safe to keep.

## Logs
- results.tsv (compact)
- detailed_results.tsv (full config per run)
