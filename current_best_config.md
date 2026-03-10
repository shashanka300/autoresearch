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
- MLP = parameter-matched SwiGLU
- GQA = n_kv_head = n_head // 4 (4:1)
- RoPE = Partial RoPE (50% rotated channels)
- RoPE base = 50000
- Value embeddings = ON (alternating layers)
- Residual scalars = learned
- torch.compile = OFF on Windows auto mode

## Key discarded alternatives vs chosen config
- GeGLU gate: worse val_bpb (0.001456)
- GQA 2:1: worse val_bpb (0.001434)
- Value embeddings OFF: much worse (0.001536)
- Fixed residual scalars (1,0): worse (0.001527)

## Where to see all runs
- Simple log: results.tsv
- Detailed log with config columns: detailed_results.tsv
