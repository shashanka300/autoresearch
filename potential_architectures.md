# Potential Architecture Experiments

Reference: [The Big LLM Architecture Comparison](https://magazine.sebastianraschka.com/p/the-big-llm-architecture-comparison)

## Current Architecture (Baseline)

| Feature | Current Implementation |
|---------|----------------------|
| Attention | GQA + Sliding Window ("SSSL" pattern) |
| Position | RoPE |
| Normalization | Pre-Norm (RMSNorm) + QK-Norm |
| Activation | ReGLU (`relu(x).square()`) |
| Optimizer | MuonAdamW (Muon for matrices, AdamW for rest) |
| Residuals | Per-layer scalars (`resid_lambdas`, `x0_lambdas`) |
| Output | Softcap logits (softcap=15) |
| Value Embeddings | ResFormer-style gated value embeddings |

### Current Hyperparameters
```python
ASPECT_RATIO = 64       # model_dim = depth * ASPECT_RATIO
HEAD_DIM = 128          # target head dimension for attention
WINDOW_PATTERN = "SSSL" # sliding window pattern: L=full, S=half context
DEPTH = 8               # number of transformer layers
DEVICE_BATCH_SIZE = 32  # per-device batch size
```

---

## 1. Attention Mechanisms

### 1a. Remove Sliding Window
- **Change**: `WINDOW_PATTERN = "LLLL"` (full attention every layer)
- **Complexity**: Low (one-line change)
- **Reference**: Most LLMs use full attention; sliding window may hurt long-range dependencies
- **Try**: `"LLLL"`, `"SSLL"`, `"SLSL"`, `"SSSS"`

### 1b. Vary Window Sizes
- **Change**: Modify short window ratio in `_compute_window_sizes()`
- **Complexity**: Low
- **Options**: Short window = 1/4, 1/3, 1/2 of context
- **Reference**: Gemma 2/3 use sliding window on every other layer

### 1c. Multi-Head Latent Attention (MLA)
- **Description**: Compress KV tensors into lower-dimensional latent space before KV cache storage; project back at inference time
- **Complexity**: High
- **Reference**: DeepSeek V3/R1, Kimi K2, Kimi Linear
- **Implementation**: Add compression/projection matrices for KV

### 1d. Gated Attention
- **Description**: Add sigmoid-controlled output gate to attention
- **Complexity**: Medium
- **Reference**: Qwen3-Next, Kimi Linear
- **Implementation**: `output = gate * attention_output` where gate = sigmoid(proj(x))

### 1e. Gated DeltaNet
- **Description**: Linear attention variant with gating mechanism; updates small fast-weight memory; avoids quadratic cost
- **Complexity**: High
- **Reference**: Qwen3-Next, Kimi Linear
- **Note**: Significant architecture change, would need new attention implementation

### 1f. Attention Sinks
- **Description**: Learned per-head bias logits appended to attention scores for long-context stability
- **Complexity**: Medium
- **Reference**: gpt-oss
- **Implementation**: Add learnable bias parameters to attention scores

### 1g. Attention Bias
- **Description**: Learnable bias units in Q/K/V weight projections
- **Complexity**: Medium
- **Reference**: gpt-oss, Grok 2.5, GLM-4.5
- **Implementation**: Add bias=True to attention projections

### 1h. Vary GQA Ratio
- **Description**: Different ratios of query heads to KV heads
- **Complexity**: Low
- **Current**: n_kv_head = n_head (full MHA)
- **Try**: n_kv_head = n_head // 2, n_head // 4
- **Reference**: Llama 3/4, Gemma 3, Mistral, Qwen3 all use GQA

---

## 2. Positional Embeddings

### 2a. Partial RoPE
- **Description**: Apply rotation to only first `rotary_dim` channels; remaining dimensions unchanged
- **Complexity**: Low
- **Reference**: MiniMax-M1/M2
- **Implementation**: Modify `apply_rotary_emb()` to only rotate first N dims

### 2b. NoPE for Some Layers
- **Description**: Remove RoPE from every Nth layer, rely on causal mask
- **Complexity**: Low
- **Reference**: SmolLM3 (every 4th layer)
- **Implementation**: Skip RoPE in selected layers

### 2c. Vary RoPE Base
- **Description**: Different rotary base frequency values
- **Complexity**: Low
- **Current**: base=10000
- **Try**: 50000, 100000, 500000 (longer context extrapolation)
- **Reference**: Many modern LLMs use larger bases

### 2d. No RoPE At All
- **Description**: Remove RoPE entirely, rely only on causal mask
- **Complexity**: Medium
- **Risk**: May hurt position awareness significantly

---

## 3. Normalization Patterns

### 3a. Post-Norm
- **Description**: Move RMSNorm after attention/FFN (inside residual)
- **Complexity**: Low
- **Reference**: OLMo 2, original transformer
- **Implementation**: `x = x + norm(block(x))` instead of `x = x + block(norm(x))`

### 3b. Pre+Post-Norm
- **Description**: RMSNorm both before AND after each block
- **Complexity**: Medium
- **Reference**: Gemma 2/3
- **Implementation**: `x = x + post_norm(block(pre_norm(x)))`

### 3c. Per-Head QK-Norm
- **Description**: Unique QK-Norm parameters for each attention head
- **Complexity**: Medium
- **Reference**: MiniMax-M2
- **Implementation**: Separate RMSNorm per head instead of shared

### 3d. Remove QK-Norm
- **Description**: Test if QK-Norm is actually helping
- **Complexity**: Low
- **Implementation**: Comment out `q, k = norm(q), norm(k)` in attention

### 3e. LayerNorm Instead of RMSNorm
- **Description**: Classic LayerNorm with bias
- **Complexity**: Low
- **Reference**: Original transformer, some older models
- **Implementation**: Replace `F.rms_norm` with `F.layer_norm`

---

## 4. Activation Functions

### 4a. SwiGLU
- **Description**: Gated linear unit with Swish/SiLU
- **Complexity**: Low
- **Reference**: Most modern LLMs (Llama, Gemma, Qwen, etc.)
- **Implementation**:
  ```python
  gate = F.silu(self.c_gate(x))
  up = self.c_up(x)
  return self.c_proj(gate * up)
  ```

### 4b. GeGLU
- **Description**: GELU-gated linear unit
- **Complexity**: Low
- **Reference**: Some PaLM variants
- **Implementation**: Same as SwiGLU but with GELU

### 4c. Plain Swish/SiLU
- **Description**: No gating, just `silu(x)`
- **Complexity**: Low
- **Implementation**: Simple activation swap

### 4d. Plain GELU
- **Description**: Standard GELU activation (no gating)
- **Complexity**: Low
- **Reference**: GPT-2, original transformer variants

### 4e. Remove ReGLU Square
- **Description**: Try `relu(x)` instead of `relu(x).square()`
- **Complexity**: Low
- **Rationale**: The square may be too aggressive

---

## 5. Mixture-of-Experts (MoE)

### 5a. Sparse MoE
- **Description**: Replace MLP with routed experts (e.g., top-8 of 64 experts)
- **Complexity**: High
- **Reference**: DeepSeek V3/R1, Llama 4, Qwen3 MoE, GLM-4.5, MiniMax-M2
- **Implementation**: Add router network + multiple expert MLPs

### 5b. MoE + Shared Expert
- **Description**: One always-active expert + routed experts
- **Complexity**: High
- **Reference**: DeepSeek V3/R1, Llama 4, GLM-4.5, Qwen3-Next, Grok 2.5
- **Rationale**: Shared expert handles common patterns

### 5c. Dense-to-MoE Transition
- **Description**: First N layers dense, then MoE
- **Complexity**: High
- **Reference**: GLM-4.5 (3 dense layers first)
- **Rationale**: Early layers learn general features, later layers specialize

### 5d. Expert Choice Routing
- **Description**: Experts select tokens (not tokens select experts)
- **Complexity**: High
- **Reference**: Some research papers
- **Rationale**: Better load balancing

---

## 6. Residual & Skip Connections

### 6a. Remove Per-Layer Scalars
- **Description**: Test if `resid_lambdas` and `x0_lambdas` help
- **Complexity**: Low
- **Implementation**: Set `resid_lambdas=1`, `x0_lambdas=0` fixed

### 6b. Fixed Residual Scaling
- **Description**: Replace learned scalars with fixed values
- **Complexity**: Low
- **Try**: `resid_lambdas=0.9`, `x0_lambdas=0.1` fixed

### 6c. Skip Connections Every N Layers
- **Description**: Deep networks sometimes skip intermediate layers
- **Complexity**: Medium
- **Reference**: DenseNet-style skip connections

### 6d. Pre-Norm Only on Skip
- **Description**: Only apply norm to skip connection, not residual branch
- **Complexity**: Low

---

## 7. Model Scaling / Dimensions

### 7a. Vary DEPTH
- **Description**: Different number of transformer layers
- **Complexity**: Low
- **Try**: 4, 6, 10, 12, 16 layers
- **Trade-off**: Deeper = more expressive but slower

### 7b. Vary ASPECT_RATIO
- **Description**: Different width/depth ratios
- **Complexity**: Low
- **Try**: 32, 48, 64, 80, 96
- **Trade-off**: Wider = faster inference, deeper = more flexibility

### 7c. Vary HEAD_DIM
- **Description**: Different attention head dimensions
- **Complexity**: Low
- **Try**: 64, 96, 128, 192, 256
- **Reference**: Most models use 64-128

### 7d. Deeper & Narrower
- **Description**: More layers, smaller hidden dim
- **Complexity**: Low
- **Reference**: Some models prefer depth over width

### 7e. Shallower & Wider
- **Description**: Fewer layers, larger hidden dim
- **Complexity**: Low
- **Reference**: gpt-oss uses wider architecture

---

## 8. Optimizer & Training

### 8a. Vary Muon Momentum Schedule
- **Description**: Different momentum curves
- **Complexity**: Low
- **Current**: Linear ramp from 0.85 to 0.95 over 300 steps
- **Try**: Different start/end values, different ramp schedules

### 8b. Vary Weight Decay Schedule
- **Description**: Different decay curves
- **Complexity**: Low
- **Current**: Linear decay from 0.2 to 0
- **Try**: Constant weight decay, different max values

### 8c. Different Adam Betas
- **Description**: Different momentum parameters for Adam
- **Complexity**: Low
- **Current**: (0.8, 0.95)
- **Try**: (0.9, 0.95), (0.9, 0.999), (0.7, 0.95)

### 8d. Add Warmup
- **Description**: LR warmup at start of training
- **Complexity**: Low
- **Current**: WARMUP_RATIO = 0.0
- **Try**: 0.02, 0.05, 0.1

### 8e. Non-Zero Final LR
- **Description**: Don't decay LR all the way to zero
- **Complexity**: Low
- **Current**: FINAL_LR_FRAC = 0.0
- **Try**: 0.1, 0.05

### 8f. Gradient Clipping
- **Description**: Add norm clipping
- **Complexity**: Low
- **Try**: max_grad_norm = 1.0, 0.5, 2.0

---

## 9. Output / Logits

### 9a. Remove Softcap
- **Description**: Test if softcap is necessary
- **Complexity**: Low
- **Implementation**: Comment out softcap logic

### 9b. Vary Softcap Value
- **Description**: Different softcap thresholds
- **Complexity**: Low
- **Current**: softcap = 15
- **Try**: 10, 20, 30, 50

### 9c. Tie Embeddings
- **Description**: Share wte and lm_head weights
- **Complexity**: Low
- **Benefit**: Reduces parameters, may regularize
- **Implementation**: `self.lm_head.weight = self.transformer.wte.weight`

---

## 10. Multi-Token Prediction

### 10a. 2-Token Prediction
- **Description**: Predict next 2 tokens per forward pass
- **Complexity**: Medium
- **Reference**: Qwen3-Next, DeepSeek V3.2, GLM-4.5
- **Implementation**: Add second prediction head, train with both losses

### 10b. 3-Token Prediction
- **Description**: Predict next 3 tokens per forward pass
- **Complexity**: Medium
- **Benefit**: Enables speculative decoding at inference

---

## 11. Value Embeddings (Current Feature)

### 11a. Remove Value Embeddings
- **Description**: Test if value embeddings help
- **Complexity**: Low
- **Implementation**: Comment out value embedding logic

### 11b. All Layers Have Value Embeddings
- **Description**: Currently alternating layers have VE
- **Complexity**: Low
- **Current**: `has_ve()` returns True for alternating layers
- **Try**: All layers have VE

### 11c. No Layers Have Value Embeddings
- **Description**: Ablate value embeddings entirely
- **Complexity**: Low

---

## Experiment Priority Order

### Tier 1: Quick Wins (Low Complexity, High Impact Potential)
1. `WINDOW_PATTERN` variations: `"LLLL"`, `"SSSS"`, `"SLSL"`
2. Remove softcap or vary softcap value
3. SwiGLU instead of ReGLU
4. Vary `HEAD_DIM`: 64, 96, 256
5. Vary `ASPECT_RATIO`: 48, 80
6. Vary `DEPTH`: 6, 10, 12
7. Remove QK-Norm (ablation)
8. Remove value embeddings (ablation)
9. Tie embeddings

### Tier 2: Medium Complexity
10. Pre+Post-Norm
11. Partial RoPE
12. NoPE for some layers
13. Vary RoPE base
14. Attention bias
15. Attention sinks
16. Different GQA ratios
17. Per-head QK-Norm

### Tier 3: High Complexity (Major Changes)
18. MLA (Multi-Head Latent Attention)
19. MoE with shared expert
20. Gated DeltaNet
21. Multi-token prediction
22. Gated attention

---

## Reference Model Architectures

| Model | Attention | Norm | Activation | Position | Notable Features |
|-------|-----------|------|------------|----------|------------------|
| Llama 3 | GQA | Pre-Norm RMSNorm | SwiGLU | RoPE | Standard modern LLM |
| Gemma 2/3 | GQA + Sliding Window | Pre+Post-Norm RMSNorm + QK-Norm | GeGLU | RoPE | Sliding window, dual norm |
| DeepSeek V3 | MLA | Pre-Norm RMSNorm | SwiGLU | RoPE | MoE + Shared Expert, MLA |
| Qwen3 | GQA | Pre-Norm RMSNorm + QK-Norm | SwiGLU | RoPE | GQA, QK-Norm |
| Mistral | GQA + Sliding Window | Pre-Norm RMSNorm | SwiGLU | RoPE | Sliding window |
| OLMo 2 | MHA | Post-Norm RMSNorm + QK-Norm | SwiGLU | RoPE | Post-norm, QK-Norm |
| MiniMax-M2 | GQA | Pre-Norm RMSNorm + Per-Head QK-Norm | SwiGLU | Partial RoPE | Per-head QK-Norm, MoE |
| gpt-oss | GQA + Sliding Window | Pre-Norm RMSNorm | SwiGLU | RoPE | Attention sinks, attention bias |