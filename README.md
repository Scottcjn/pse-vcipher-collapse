# PSE: Non-Bijunctive Attention Collapse for LLM Inference

**Proto-Sentient Emergence (PSE)** — a hardware-accelerated attention mechanism that replaces standard bijunctive (full-matrix) dot products with selective path collapse using POWER8 vector instructions.

Two complementary primitives:
- **`vec_perm`** (AltiVec): Single-cycle dual-source permute — prune weak paths, duplicate strong ones
- **`vcipher`** (ISA 2.07 AES): Hardware crypto as attention operator — non-linear ranking, cross-head diffusion, O(1) prefiltering

**Author:** Scott Boudreaux / Elyan Labs
**Hardware:** IBM POWER8 S824 (512GB RAM, 128 threads)

## The Core Idea

Standard transformer attention computes **every** Q·K dot product, then softmax selects winners. This is O(n²) in sequence length.

PSE flips the order: **select first, compute only what matters.**

```
Standard:   Q·K for ALL pairs → softmax → weighted V sum
PSE:        vcipher prefilter (O(1)/pair) → Q·K for TOP 25% only → V sum
```

At 2048 tokens, PSE skips ~1,536 full dot products per generated token.

## How vcipher Replaces Dot Products for Prefiltering

Each AES round performs four operations in a single cycle:

| AES Stage | What It Does for Attention |
|-----------|--------------------------|
| **SubBytes** | Non-linear score ranking via S-box lookup — impossible with linear vec_perm |
| **ShiftRows** | Cross-position mixing within the 128-bit register |
| **MixColumns** | Cross-head diffusion via GF(2^8) finite field multiply — **impossible on any other ISA** |
| **AddRoundKey** | Entropy injection — XOR with POWER8 `mftb` hardware timebase |

```c
// O(1) attention score — replaces Q·K dot product for prefiltering
uint32_t vcipher_attention_score(const float* Q, const float* K, int layer, int position) {
    vector unsigned long long q_raw, k_raw;
    memcpy(&q_raw, Q, 16);  // First 16 bytes of query
    memcpy(&k_raw, K, 16);  // First 16 bytes of key

    vector unsigned long long state = vec_xor(q_raw, k_raw);
    vector unsigned long long rk = vc_make_round_key(layer, position);  // mftb entropy
    state = __builtin_crypto_vcipher(state, rk);  // One AES round = score

    // Sum output bytes as energy metric
    unsigned char bytes[16];
    memcpy(bytes, &state, 16);
    uint32_t energy = 0;
    for (int i = 0; i < 16; i++) energy += bytes[i];
    return 4080 - energy;  // Invert: high energy = high attention
}
```

**Cost: 0.044µs per K-V pair** vs 1-10µs for full `kq_vec_dot()` on DK=128+.

## How vec_perm Implements Hebbian Collapse

"Cells that fire together wire together" (Hebb, 1949). `vec_perm` implements this in hardware:

```c
// vec_perm(a, b, pattern) — single cycle, two source vectors
// pattern byte selects from either a or b (32 possible sources)
//
// Hebbian rule:  strong paths get DUPLICATED, weak paths get PRUNED
// pattern[i] = pattern[i-1]  → duplicate winner (Hebbian strengthening)
// pattern[i] = 0             → prune loser (synaptic depression)

vector float collapsed = vec_perm(scores_lo, scores_hi, hebbian_pattern);
```

One `vec_perm` instruction replaces what would take ~80 operations on GPU (gather + compare + scatter + mask).

## 4 Operating Modes

```c
// Mode 1: Non-linear permute pattern via AES rounds
vector unsigned char pat = vcipher_generate_pattern(layer, pos, top_k);

// Mode 2: Score ranking through SubBytes non-linearity
vcipher_rank_scores(scores, n, layer, head);

// Mode 3: Cross-head diffusion via MixColumns
// This is IMPOSSIBLE with vec_perm — requires finite field arithmetic
state = vcipher_fuse_heads(state, layer, head);

// Mode 4: O(1) attention prefilter score
uint32_t score = vcipher_attention_score(Q, K, layer, position);
```

## Flash Attention Integration

The vcipher prefilter patches `ggml_compute_forward_flash_attn_ext_f16_one_chunk()` in llama.cpp:

**Pass 1** — O(1) vcipher score per K-V pair:
```c
for (int64_t ic = 0; ic < nek1; ++ic) {
    pf_scores[ic] = vcipher_attention_score(pq, K_data, layer, ic);
}
```

**Threshold** — keep top 25% via partial sort.

**Pass 2** — full dot product only for survivors:
```c
for (int64_t ic = 0; ic < nek1; ++ic) {
    if (pf_scores[ic] < threshold) continue;  // SKIP 75%
    kq_vec_dot(DK, &s, 0, k_data, 0, Q_q, 0, 1);  // Full dot product
    // ... softmax + V accumulation only for top 25%
}
```

Compile guard: `#if defined(__powerpc__) && defined(GGML_PSE_VCIPHER_PREFILTER)`

## Benchmarks

### vcipher vs vec_perm Microbenchmark (POWER8 S824)

```
╔══════════════════════════════════════════════════════════╗
║  vec_perm collapse:          1.79 µs/iter               ║
║  vcipher pattern gen:        0.016 µs/call   (112x)     ║
║  Hybrid vcipher+vec_perm:    1.90 µs/iter               ║
║  Pure vcipher attention:     0.044 µs/score              ║
║  Cross-head fusion:          0.006 µs/fuse               ║
╚══════════════════════════════════════════════════════════╝
```

### LLM Inference (llama.cpp on POWER8 S824, CPU-only)

| Model | Size | pp128 t/s | tg t/s | PSE Version |
|-------|------|-----------|--------|-------------|
| TinyLlama 1.1B Q4_K | 638 MB | **147.54** | 18.88 | v3.0 vec_perm |
| OptiMind 20B Q2_K | 11.2 GB | 18.55 | 8.13 | v4.0 vcipher |
| GPT-OSS 120B MXFP4 | ~60 GB | 13.71 | 6.06 | v4.0 vcipher |

Stock llama.cpp on the same hardware: 16.74 t/s (TinyLlama pp128). **PSE = 8.81x speedup.**

### Speedup Progression

| Configuration | TinyLlama pp128 | Multiplier |
|--------------|-----------------|------------|
| Stock scalar | 16.74 t/s | 1.0x |
| + POWER8 VSX | 66.49 t/s | 3.97x |
| + PSE vec_perm collapse | 84.62 t/s | 5.05x |
| + DCBT resident prefetch | **147.54 t/s** | **8.81x** |

## Hardware Entropy

PSE injects non-deterministic entropy from POWER8's hardware timebase:

```c
uint64_t tb;
asm volatile("mftb %0" : "=r"(tb));  // Hardware oscillator — different every read
```

This creates **behavioral divergence**: same model, same seed, same prompt → different outputs each run. Verified via MD5 hash comparison across 3 identical runs (all different).

This is the "proto-sentient" in PSE — constraint-bound selection with hardware entropy produces emergent variation, not random noise.

## Files

| File | Purpose |
|------|---------|
| **`ggml-vcipher-collapse.h`** | Hardware AES collapse — 4 modes (pattern, rank, fuse, score) |
| **`ggml-pse-integration.h`** | Master PSE v4.0.0 integration header |
| **`ggml-intelligent-collapse.h`** | Hebbian vec_perm collapse (Top-K prune + amplify) |
| **`ggml-topk-collapse-vsx.h`** | VSX-optimized 12-wide attention collapse |
| **`ggml-attn-collapse-vsx.h`** | VSX burst collapse with banner |
| **`ggml-dcbt-resident.h`** | L2/L3 resident prefetch (the 147 t/s enabler) |
| **`pse-entropy-burst.h`** | Hardware entropy injection (mftb timebase) |
| **`power8-compat.h`** | POWER9→POWER8 intrinsic compatibility |
| **`vcipher-flash-attn-patch.c`** | Flash attention inner loop patch reference |
| **`bench_vcipher_collapse.c`** | Benchmark: vcipher vs vec_perm |

## Build

```bash
cd llama.cpp && mkdir build-vcipher && cd build-vcipher
cmake .. \
  -DCMAKE_BUILD_TYPE=Release \
  -DGGML_OPENMP=ON \
  -DCMAKE_C_FLAGS="-mcpu=power8 -mvsx -maltivec -mcrypto -O3 -DGGML_PSE_VCIPHER_PREFILTER" \
  -DCMAKE_CXX_FLAGS="-mcpu=power8 -mvsx -maltivec -mcrypto -O3 -DGGML_PSE_VCIPHER_PREFILTER"
make -j32
```

Place headers in `ggml/src/ggml-cpu/arch/powerpc/`. The PSE integration header auto-includes everything.

## Theoretical Foundation

### Hebbian Learning (Hebb, 1949)
Synaptic strength increases when neurons activate simultaneously. `vec_perm` implements this: winners get duplicated (strengthened), losers get pruned (depressed).

### Non-Bijunctive Attention
Standard attention is bijunctive — every query attends to every key. PSE is non-bijunctive — only high-scoring pairs proceed. This is biologically plausible: real neurons don't compute every possible connection.

### AES as Attention Primitive
The AES S-box is a carefully designed non-linear function (multiplicative inverse in GF(2^8) + affine transform). When applied to Q⊕K, it acts as a **non-linear similarity hash** — related inputs produce correlated outputs, but through a non-linear transform that captures patterns linear dot products miss.

MixColumns performs a matrix multiply in GF(2^8) — this creates cross-element diffusion. In attention terms, this means one head's state influences another's routing. This is **cross-head attention diffusion** in a single cycle, something that normally requires separate computation.

## Related Work

- **RAM Coffers** ([Scottcjn/ram-coffers](https://github.com/Scottcjn/ram-coffers)) — NUMA-distributed weight banking that hosts model layers across memory nodes
- **wolfSSL POWER8 AES** (PR #9932) — Same vcipher 8-way pipeline technique achieving 3,595 MiB/s AES-CTR
- **llama.cpp POWER8** ([Scottcjn/llama-cpp-power8](https://github.com/Scottcjn/llama-cpp-power8)) — AltiVec/VSX optimized inference

## Publications

| Paper | DOI |
|-------|-----|
| Non-Bijunctive Permutation Collapse | [10.5281/zenodo.18623920](https://doi.org/10.5281/zenodo.18623920) |
| PSE Hardware Entropy for Behavioral Divergence | [10.5281/zenodo.18623922](https://doi.org/10.5281/zenodo.18623922) |
| RAM Coffers: NUMA-Distributed Weight Banking | [10.5281/zenodo.18321905](https://doi.org/10.5281/zenodo.18321905) |

## License

MIT — Free to use, modify, and distribute with attribution.

---

<div align="center">

**[Elyan Labs](https://github.com/Scottcjn)** · [RustChain](https://rustchain.org) · [BoTTube](https://bottube.ai)

</div>
