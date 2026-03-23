import { useState } from 'react';

const M = "'JetBrains Mono','Fira Code',monospace";
const S = "'Inter','Segoe UI',sans-serif";
const CYAN   = '#22d3ee';
const LIME   = '#84cc16';
const AMBER  = '#f59e0b';
const PINK   = '#f472b6';
const PURPLE = '#a855f7';
const GRAY   = '#6b7280';

const CARD = {
  padding: '12px 14px',
  borderRadius: 7,
  background: 'rgba(255,255,255,.03)',
  border: '1px solid rgba(255,255,255,.06)',
};

// ─── Syntax highlighter ───────────────────────────────────────────────────────
function HL({ code }) {
  const kw = '#f472b6', cm = '#4a5568', nm = '#22d3ee', st = '#f59e0b';
  const s = {
    padding: '14px 16px', borderRadius: 8,
    background: 'rgba(0,0,0,.5)', border: '1px solid rgba(255,255,255,.06)',
    fontFamily: M, fontSize: 11, lineHeight: 1.85, whiteSpace: 'pre', overflowX: 'auto',
  };
  return (
    <div style={s}>
      {code.split('\n').map((l, i) => {
        const t = l.trimStart();
        if (t.startsWith('#') || t.startsWith('//'))
          return <div key={i} style={{ color: cm, fontStyle: 'italic' }}>{l}</div>;
        const h = l
          .replace(/\b(def|for|if|else|elif|return|while|import|from|class|in|range|pass|True|False|None|torch|tl|triton|float|int|void|const|auto|__global__|__shared__)\b/g, '⟨k⟩$1⟨/k⟩')
          .replace(/\b(\d+\.?\d*)\b/g, '⟨n⟩$1⟨/n⟩')
          .replace(/"([^"]*)"/g, '⟨s⟩"$1"⟨/s⟩')
          .replace(/'([^']*)'/g, "⟨s⟩'$1'⟨/s⟩");
        const ps = h.split(/(⟨k⟩.*?⟨\/k⟩|⟨n⟩.*?⟨\/n⟩|⟨s⟩.*?⟨\/s⟩)/g);
        return (
          <div key={i}>
            {ps.map((p, j) => {
              if (p.startsWith('⟨k⟩')) return <span key={j} style={{ color: kw }}>{p.slice(3, -4)}</span>;
              if (p.startsWith('⟨n⟩')) return <span key={j} style={{ color: nm }}>{p.slice(3, -4)}</span>;
              if (p.startsWith('⟨s⟩')) return <span key={j} style={{ color: st }}>{p.slice(3, -4)}</span>;
              return <span key={j} style={{ color: '#c9d1e0' }}>{p}</span>;
            })}
          </div>
        );
      })}
    </div>
  );
}

// ─── Math components ──────────────────────────────────────────────────────────
function MathBlock({ children, label }) {
  return (
    <div style={{ margin: '14px 0', padding: '14px 18px', borderRadius: 8, background: 'rgba(255,255,255,.02)', border: '1px solid rgba(255,255,255,.08)' }}>
      {label && <div style={{ fontSize: 9, color: '#555', fontFamily: M, textTransform: 'uppercase', letterSpacing: 1.5, marginBottom: 8 }}>{label}</div>}
      <div style={{ fontFamily: "'Cambria Math','Georgia','Times New Roman',serif", fontSize: 17, color: '#e0e0e0', textAlign: 'center', letterSpacing: 1, lineHeight: 2 }}>
        {children}
      </div>
    </div>
  );
}
function Var({ children, color }) {
  return <span style={{ fontStyle: 'italic', color: color || '#e8c240', fontWeight: 500 }}>{children}</span>;
}
function Sub({ children }) {
  return <sub style={{ fontSize: '0.65em', fontStyle: 'normal', color: '#888' }}>{children}</sub>;
}

// ─── Tier badge colors ────────────────────────────────────────────────────────
// tier: 'pytorch' | 'triton' | 'flashmla' | 'deepgemm' | 'kept'
function tierStyle(tier) {
  switch (tier) {
    case 'pytorch':  return { bg: `${GRAY}18`,   border: `${GRAY}55`,   text: GRAY,   label: 'PyTorch' };
    case 'triton':   return { bg: `${AMBER}14`,  border: `${AMBER}55`,  text: AMBER,  label: 'Triton'  };
    case 'flashmla': return { bg: `${CYAN}14`,   border: `${CYAN}55`,   text: CYAN,   label: 'FlashMLA' };
    case 'deepgemm': return { bg: `${LIME}14`,   border: `${LIME}55`,   text: LIME,   label: 'DeepGEMM' };
    case 'kept':     return { bg: `${PURPLE}14`, border: `${PURPLE}55`, text: PURPLE, label: 'Kept' };
    default:         return { bg: 'transparent', border: '#333',        text: '#666',  label: '—' };
  }
}

function TierBadge({ tier }) {
  const { bg, border, text, label } = tierStyle(tier);
  return (
    <span style={{
      display: 'inline-block', padding: '2px 7px', borderRadius: 4,
      background: bg, border: `1px solid ${border}`,
      fontFamily: M, fontSize: 9, color: text, letterSpacing: 0.5,
      textTransform: 'uppercase', whiteSpace: 'nowrap',
    }}>
      {label}
    </span>
  );
}

function CellContent({ tier, text, sub }) {
  const { text: col } = tierStyle(tier);
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
      <TierBadge tier={tier} />
      <span style={{ fontFamily: M, fontSize: 10, color: col, lineHeight: 1.4 }}>{text}</span>
      {sub && <span style={{ fontSize: 10, color: '#555', lineHeight: 1.4 }}>{sub}</span>}
    </div>
  );
}

// ─── Component detail data ────────────────────────────────────────────────────
const DETAIL = {
  rmsnorm: {
    file: 'unsloth/kernels/rms_layernorm.py',
    sig: 'fast_rms_layernorm(W, x, eps=1e-6)',
    constraint: 'Kept Triton at CUDA tier — already memory-bound optimal; no FP8 benefit here.',
    code: `# Unsloth Triton RMSNorm — single-pass, no separate variance kernel
@triton.jit
def _fast_rms_layernorm_kernel(X, W, Out, stride, N, eps, BLOCK: tl.constexpr):
    row = tl.program_id(0)
    X  += row * stride
    Out += row * stride
    cols = tl.arange(0, BLOCK)
    x   = tl.load(X + cols, mask=cols < N, other=0.0).to(tl.float32)
    var = tl.sum(x * x, axis=0) / N
    rstd = 1 / tl.sqrt(var + eps)
    w = tl.load(W + cols, mask=cols < N)
    tl.store(Out + cols, (x * rstd * w).to(Out.dtype.element_ty), mask=cols < N)`,
  },
  swiglu: {
    file: 'unsloth/kernels/swiglu.py',
    sig: 'swiglu_fg_kernel(e, g)',
    constraint: 'Fused gate+up into single HBM read. 3N traffic vs 5N unfused.',
    code: `# Fused SwiGLU: silu(gate) * up — avoids materialising intermediate activation
@triton.jit
def _swiglu_forward_kernel(e, g, h, n_elements, BLOCK: tl.constexpr):
    idx = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    mask = idx < n_elements
    gate_val = tl.load(e + idx, mask=mask).to(tl.float32)
    up_val   = tl.load(g + idx, mask=mask).to(tl.float32)
    out = gate_val * tl.sigmoid(gate_val) * up_val   # silu(gate) * up
    tl.store(h + idx, out.to(h.dtype.element_ty), mask=mask)`,
  },
  crossentropy: {
    file: 'unsloth/kernels/cross_entropy_loss.py',
    sig: 'fast_cross_entropy_loss(logits, labels, chunk_size=4096)',
    constraint: '154K vocab: naive CE materialises [B, 154K] fp32 tensor (~2.4 GB/batch). Chunked avoids this.',
    code: `# Chunked cross-entropy — never materialises full [B, V] softmax
def fast_cross_entropy_loss(logits, labels, chunk_size=4096):
    # logits: [B*T, 154112]  — 154K GLM-5 vocab
    loss = 0.0
    for start in range(0, logits.shape[-1], chunk_size):
        chunk = logits[..., start:start+chunk_size]
        # online softmax: subtract running max for numerical stability
        loss += _ce_chunk_kernel(chunk, labels, start)
    return loss / logits.shape[0]`,
  },
  rope: {
    file: 'modeling/glm5_modeling.py',
    sig: 'apply_rotary_pos_emb(q, k, cos, sin, position_ids)',
    constraint: 'd_rope=64 is tiny (head_dim=128, only first 64 dims rotated). PyTorch fused ops fast enough.',
    code: `# RoPE on 64-dim subspace only — q/k heads are 128-dim, rope on [:64]
def rotate_half(x):
    x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
    return torch.cat((-x2, x1), dim=-1)

def apply_rotary_pos_emb(q, k, cos, sin, pos_ids):
    cos = cos[pos_ids].unsqueeze(1)   # [B, 1, T, 64]
    sin = sin[pos_ids].unsqueeze(1)
    q_rot = (q * cos) + (rotate_half(q) * sin)  # only first 64 dims
    k_rot = (k * cos) + (rotate_half(k) * sin)
    return q_rot, k_rot`,
  },
  moerouter: {
    file: 'modeling/glm5_moe.py',
    sig: 'Glm5MoeSparseMoeBlock.forward(hidden_states)',
    constraint: 'n_group=1, top_k=8 from 256 experts. Sigmoid routing (not softmax). 5-line router.',
    code: `# Sigmoid MoE router — GLM-5 uses sigmoid not softmax
# n_group=1 means no grouped routing; straight top-8 from all 256 experts
def route(hidden_states):
    router_logits = self.gate(hidden_states)          # [B*T, 256]
    scores = torch.sigmoid(router_logits)             # independent probabilities
    topk_w, topk_idx = torch.topk(scores, k=8, dim=-1)
    topk_w = topk_w / topk_w.sum(-1, keepdim=True)   # renormalise selected
    return topk_w, topk_idx`,
  },
  dsaindexer: {
    file: 'deep_gemm/fp8_mqa_logits.cu',
    sig: 'fp8_mqa_logits(q_fp8, dsa_weights, block_indices, out)',
    constraint: 'FP8 fused: computes QK^T for DSA block selection AND converts to FP8 in one pass. Feeds FlashMLA mask.',
    code: `// DeepGEMM FP8 DSA indexer — fused QK logits + FP8 quantisation
// Input q_fp8: [B, H, T, 128] in FP8 E4M3
// dsa_weights: learned per-head block importance weights
// Outputs: block_scores [B, H, T, N_blocks] — top-K blocks selected for sparse attn
__global__ void fp8_mqa_logits_kernel(
    const __nv_fp8_e4m3* q, const __nv_fp8_e4m3* w,
    const int* block_map, float* out,
    int B, int H, int T, int N_blocks) {
  // warp-level GEMM via mma.sync PTX — H100 FP8 tensor cores
  // fused scale * qk_score → directly emits FP8 block scores
}`,
  },
  dsasparse: {
    file: 'flash_mla/flash_mla_sparse.cu',
    sig: 'flash_mla_sparse_fwd(q, k, v, block_table, block_scores, seqlens)',
    constraint: 'Sparse block attention: skips zero-score blocks entirely. ~40-60% of KV blocks skipped per head.',
    code: `// FlashMLA sparse forward — only visits blocks selected by DSA indexer
// block_table: [B, H, MAX_BLOCKS] — which KV cache pages to attend to
// block_scores: [B, H, T, K] — scores for each selected block (from fp8_mqa_logits)
void flash_mla_sparse_fwd(
    const torch::Tensor& q,          // [B, H_q, T, 128]
    const torch::Tensor& k_cache,    // [num_pages, page_size, H_kv, 128]
    const torch::Tensor& v_cache,    // [num_pages, page_size, H_kv, 512]  d_v=512!
    const torch::Tensor& block_table,
    const torch::Tensor& block_scores,
    const torch::Tensor& seqlens_k);
// Key: d_v=512 is absorbed — MLA absorbs W_UV into V projection
// so each KV head stores 512-dim V, not 128. FlashMLA handles this natively.`,
  },
  mlaattn: {
    file: 'flash_mla/flash_mla_interface.py',
    sig: 'flash_mla_with_kvcache(q, kv_cache, block_table, cache_seqlens, ...)',
    constraint: 'd_v=512 absorbed projection: W_UV merged into V so FlashMLA emits full 512-dim output directly.',
    code: `# FlashMLA decode — replaces _eager_attention_forward for generation
# Critical: d_v=512 because MLA absorbs W_UV (up-projection) into V
# Standard FlashAttention assumes d_k == d_v, FlashMLA supports d_v != d_k
from flash_mla import flash_mla_with_kvcache, get_mla_metadata

metadata = get_mla_metadata(cache_seqlens, 64, 1)   # 64 KV heads, 1 group
out, lse = flash_mla_with_kvcache(
    q=q_nope,                  # [B, H_q, 1, 128]  — no rope on latent
    k_cache=kv_cache,          # paged: [pages, 64, 128+64]  nope+rope
    v_cache=kv_cache,          # same page, d_v=512 slice
    block_table=block_table,
    cache_seqlens=cache_seqlens,
    softmax_scale=scale,
    causal=True,
)  # out: [B, H_q, 1, 512] — already projected, no separate W_UV matmul`,
  },
  moegemm: {
    file: 'deep_gemm/m_grouped_fp8_gemm.py',
    sig: 'deep_gemm.m_grouped_fp8_gemm(x_fp8, w_fp8, expert_ids, out)',
    constraint: 'Groups all tokens by expert, does one batched FP8 GEMM. Replaces 256 separate F.linear calls.',
    code: `# DeepGEMM grouped FP8 GEMM — all 256 experts in one kernel launch
# vs naive: for exp_id in range(256): out[mask] = F.linear(x[mask], W[exp_id])
import deep_gemm

# Quantise activations to FP8 E4M3 with per-token scale
x_fp8, x_scale = per_token_quant_fp8(hidden_states)   # [B*T, d_model]
w_fp8 = expert_weights_fp8                             # [256, d_ff, d_model] pre-quantised

# Single kernel: groups tokens by expert_id, uses H100 FP8 tensor cores
out = deep_gemm.m_grouped_fp8_gemm(
    x_fp8, w_fp8, topk_ids,     # topk_ids: which expert each token goes to
    scales=(x_scale, w_scale),
)  # ~1550 TFLOPS on H100 vs ~100 TFLOPS per-expert loop`,
  },
  kvcache: {
    file: 'flash_mla/flash_mla_interface.py',
    sig: 'allocate_kv_cache(max_pages, page_size=64, dtype=torch.float8_e4m3fn)',
    constraint: 'Paged FP8: 656-byte FlashMLA page format. Each page = 64 tokens × (128 nope + 64 rope) × FP8.',
    code: `# KV cache layout for FlashMLA paged attention
# Page format: 64 tokens × 64 KV heads × (128 nope + 64 rope) dims × FP8
# 656 bytes per token-slot = 64*(128+64)*1 byte (FP8=1 byte/elem) + scale metadata

# Allocation
kv_cache = torch.zeros(
    max_pages,          # number of pages in pool
    page_size,          # 64 tokens per page
    num_kv_heads,       # 64 GQA heads
    head_dim_nope + head_dim_rope,  # 128 + 64 = 192
    dtype=torch.float8_e4m3fn,
    device='cuda',
)
block_table = torch.zeros(batch, max_pages_per_seq, dtype=torch.int32)`,
  },
  fp8quant: {
    file: 'deep_gemm/fp8_utils.py + flash_mla/fp8_cache.cu',
    sig: 'per_token_quant_fp8(x) / interleaved_fp8_kv(k, v)',
    constraint: 'Two distinct FP8 formats: DeepGEMM uses separate E4M3 scales; FlashMLA uses interleaved scale layout.',
    code: `# FORMAT 1: DeepGEMM FP8 — separate scale tensors (per-token for activations)
def per_token_quant_fp8(x: torch.Tensor):
    # x: [B*T, d_model]  float16/bf16
    scale = x.abs().max(dim=-1).values / 448.0   # 448 = FP8 E4M3 max
    x_fp8 = (x / scale.unsqueeze(-1)).to(torch.float8_e4m3fn)
    return x_fp8, scale   # scale: [B*T]  — stored separately

# FORMAT 2: FlashMLA interleaved — scale packed inline with data
# Every 16 FP8 values share one FP16 scale, interleaved in memory
# Layout: [val0..val15][scale_fp16][val16..val31][scale_fp16]...
# Reason: FlashMLA's WGMMA instruction reads scale inline during GEMM`,
  },
  mtp: {
    file: 'N/A — Phase 2 backend integration',
    sig: 'vllm.engine.GLM5Engine / TRT-LLM GLM5Runner',
    constraint: 'MTP adds 1 draft token head per module (GLM-5 has 1 MTP module). Phase 2: vLLM speculative decoding.',
    code: `# GLM-5 MTP — 1 speculation module, predicts next+1 token
# Phase 1 (current): single-token decode via FlashMLA
# Phase 2 (planned): speculative decoding via vLLM or TRT-LLM backend

# vLLM integration (planned):
from vllm import LLM, SamplingParams
llm = LLM(
    model="THUDM/GLM-5-744B",
    speculative_model="[ngram]",   # or MTP draft head
    num_speculative_tokens=1,
    tensor_parallel_size=8,        # 8x H100 for 744B
)

# TRT-LLM path uses GLM-5's MTP head weight directly
# draft_head: Linear(d_model=7168, vocab=154112) — shares embedding`,
  },
};

// ─── Main table data ──────────────────────────────────────────────────────────
const ROWS = [
  {
    id: 'rmsnorm',
    name: 'RMSNorm',
    pytorch:  { tier: 'pytorch',  text: 'RMSNorm.forward()', sub: 'manual variance + norm' },
    triton:   { tier: 'triton',   text: 'fast_rms_layernorm()', sub: 'Unsloth single-pass' },
    cuda:     { tier: 'kept',     text: 'Same Triton', sub: 'already optimal' },
  },
  {
    id: 'swiglu',
    name: 'SwiGLU',
    pytorch:  { tier: 'pytorch',  text: 'F.silu(gate) * up', sub: 'two separate ops' },
    triton:   { tier: 'triton',   text: 'swiglu_fg_kernel()', sub: 'Unsloth fused' },
    cuda:     { tier: 'kept',     text: 'Same Triton', sub: 'already optimal' },
  },
  {
    id: 'crossentropy',
    name: 'Cross-Entropy',
    pytorch:  { tier: 'pytorch',  text: 'F.cross_entropy()', sub: 'materialises [B, 154K] tensor' },
    triton:   { tier: 'triton',   text: 'fast_cross_entropy_loss()', sub: 'chunked 154K vocab' },
    cuda:     { tier: 'kept',     text: 'Same Triton', sub: 'already optimal' },
  },
  {
    id: 'rope',
    name: 'RoPE (64-dim)',
    pytorch:  { tier: 'pytorch',  text: 'rotate_half + cat', sub: 'd_rope=64 only' },
    triton:   { tier: 'kept',     text: 'Same PyTorch', sub: 'small enough' },
    cuda:     { tier: 'kept',     text: 'Same PyTorch', sub: 'small enough' },
  },
  {
    id: 'moerouter',
    name: 'MoE Router',
    pytorch:  { tier: 'pytorch',  text: 'sigmoid → topk(8)', sub: '256 experts' },
    triton:   { tier: 'kept',     text: 'Same PyTorch', sub: 'n_group=1, 5 lines' },
    cuda:     { tier: 'kept',     text: 'Same PyTorch', sub: 'n_group=1, 5 lines' },
  },
  {
    id: 'dsaindexer',
    name: 'DSA Indexer',
    pytorch:  { tier: 'pytorch',  text: 'einsum + relu + weights', sub: 'block importance scores' },
    triton:   { tier: 'kept',     text: 'Same PyTorch', sub: '' },
    cuda:     { tier: 'deepgemm', text: 'fp8_mqa_logits', sub: 'FP8 fused QK+quant' },
  },
  {
    id: 'dsasparse',
    name: 'DSA Sparse Attn',
    pytorch:  { tier: 'pytorch',  text: 'matmul + mask + softmax', sub: 'dense baseline' },
    triton:   { tier: 'kept',     text: 'Same PyTorch', sub: '' },
    cuda:     { tier: 'flashmla', text: 'flash_mla_sparse_fwd', sub: 'skips zero-score blocks' },
  },
  {
    id: 'mlaattn',
    name: 'MLA Attention',
    pytorch:  { tier: 'pytorch',  text: '_eager_attention_forward', sub: 'standard eager' },
    triton:   { tier: 'kept',     text: 'Same PyTorch', sub: '' },
    cuda:     { tier: 'flashmla', text: 'flash_mla_with_kvcache', sub: 'absorbed d_v=512' },
  },
  {
    id: 'moegemm',
    name: 'MoE GEMM',
    pytorch:  { tier: 'pytorch',  text: 'per-expert F.linear loop', sub: '256 kernels launched' },
    triton:   { tier: 'triton',   text: 'grouped GEMM kernel', sub: 'Unsloth Triton' },
    cuda:     { tier: 'deepgemm', text: 'm_grouped_fp8_gemm', sub: 'FP8 batched, 1550 TFLOPS' },
  },
  {
    id: 'kvcache',
    name: 'KV Cache',
    pytorch:  { tier: 'pytorch',  text: 'simple concat', sub: 'fp16/bf16' },
    triton:   { tier: 'kept',     text: 'Same concat', sub: '' },
    cuda:     { tier: 'flashmla', text: 'Paged FP8', sub: '656-byte FlashMLA format' },
  },
  {
    id: 'fp8quant',
    name: 'FP8 Quantization',
    pytorch:  { tier: 'pytorch',  text: 'N/A', sub: 'fp16/bf16 only' },
    triton:   { tier: 'pytorch',  text: 'N/A', sub: '' },
    cuda:     { tier: 'deepgemm', text: 'Two formats', sub: 'FlashMLA interleaved + DeepGEMM separate' },
  },
  {
    id: 'mtp',
    name: 'MTP Speculation',
    pytorch:  { tier: 'pytorch',  text: 'N/A', sub: 'single-token only' },
    triton:   { tier: 'pytorch',  text: 'N/A', sub: '' },
    cuda:     { tier: 'kept',     text: 'Phase 2', sub: 'vLLM / TRT-LLM backend' },
  },
];

const PERF = [
  {
    label: 'FlashMLA decode',
    kernel: 'flash_mla_with_kvcache',
    vs: 'eager _eager_attention_forward',
    speedup: '~23×',
    detail: '660 TFLOPS vs ~30 TFLOPS eager on H100',
    color: CYAN,
    bar: 0.92,
  },
  {
    label: 'DeepGEMM indexer',
    kernel: 'fp8_mqa_logits',
    vs: 'einsum + relu + weights',
    speedup: '~18×',
    detail: 'FP8 tensor cores vs BF16 einsum on H100',
    color: LIME,
    bar: 0.72,
  },
  {
    label: 'DeepGEMM MoE GEMM',
    kernel: 'm_grouped_fp8_gemm',
    vs: 'per-expert F.linear loop',
    speedup: '~15×',
    detail: '1550 TFLOPS FP8 vs ~100 TFLOPS per-expert dispatch',
    color: LIME,
    bar: 0.60,
  },
  {
    label: 'FlashMLA sparse',
    kernel: 'flash_mla_sparse_fwd',
    vs: 'dense matmul + mask',
    speedup: '~4–8×',
    detail: '40–60% KV blocks skipped per DSA mask on avg',
    color: CYAN,
    bar: 0.32,
  },
];

const GPU_TIERS = [
  {
    sm: 'SM80',
    chip: 'A100',
    color: GRAY,
    kernels: ['PyTorch baseline', 'Unsloth Triton (RMSNorm, SwiGLU, CE, grouped GEMM)'],
    note: 'No FP8 tensor cores — DeepGEMM and FlashMLA FP8 paths disabled',
  },
  {
    sm: 'SM90',
    chip: 'H100',
    color: AMBER,
    kernels: ['All SM80 kernels', 'FlashMLA (flash_mla_with_kvcache, flash_mla_sparse_fwd)', 'DeepGEMM (fp8_mqa_logits, m_grouped_fp8_gemm)', 'Paged FP8 KV cache'],
    note: 'Full kernel stack — recommended inference target',
  },
  {
    sm: 'SM100',
    chip: 'B200',
    color: PINK,
    kernels: ['All SM90 kernels', 'SM100 sparse prefill optimizations', 'Extended FP8 E5M2 accumulation paths'],
    note: 'Future-proof — SM100 paths gated at runtime via __CUDA_ARCH__',
  },
];

const RESOURCES = [
  { label: 'FlashMLA — DeepSeek MLA kernel for H100 (official)',         url: 'https://github.com/deepseek-ai/FlashMLA',                          color: CYAN   },
  { label: 'DeepGEMM — FP8 grouped GEMM for MoE (DeepSeek)',             url: 'https://github.com/deepseek-ai/DeepGEMM',                          color: LIME   },
  { label: 'Unsloth — optimized Triton kernels (RMSNorm, SwiGLU, CE)',   url: 'https://github.com/unslothai/unsloth',                             color: AMBER  },
  { label: 'GLM-5 Model Card (THUDM)',                                   url: 'https://huggingface.co/THUDM/GLM-5',                               color: PURPLE },
  { label: 'DeepSeek-V3 Technical Report — FP8 training + MoE details',  url: 'https://arxiv.org/abs/2412.19437',                                 color: PINK   },
  { label: 'vLLM speculative decoding docs',                             url: 'https://docs.vllm.ai/en/latest/features/spec_decode.html',         color: GRAY   },
];

// ─── Page component ───────────────────────────────────────────────────────────
export default function OptimizationPage() {
  const [expanded, setExpanded]       = useState(null);   // row id
  const [showCode, setShowCode]       = useState(false);
  const [activeTab, setActiveTab]     = useState('table'); // 'table' | 'perf' | 'gpu'

  function toggle(id) {
    setExpanded(prev => (prev === id ? null : id));
  }

  return (
    <div style={{ background: '#0d0f14', minHeight: '100vh', color: '#e0e0e0', fontFamily: S, padding: '32px 20px' }}>
      <div style={{ maxWidth: 1040, margin: '0 auto' }}>

        {/* ── Header ── */}
        <div style={{ marginBottom: 28 }}>
          <span style={{ fontFamily: M, fontSize: 11, color: AMBER, letterSpacing: 2, textTransform: 'uppercase' }}>
            GLM-5 · Kernel Stack
          </span>
          <h1 style={{ margin: '6px 0 6px', fontSize: 28, fontWeight: 700, color: '#f0f0f0', letterSpacing: -0.5 }}>
            Optimization Stack
          </h1>
          <p style={{ margin: 0, color: 'rgba(255,255,255,.45)', fontSize: 14, maxWidth: 680, lineHeight: 1.6 }}>
            Full progression from raw PyTorch through Unsloth Triton kernels to hand-tuned CUDA kernels (FlashMLA + DeepGEMM). 744B MoE · 78 layers · 154K vocab · H100-first.
          </p>
        </div>

        {/* ── Key insight callout ── */}
        <div style={{
          marginBottom: 24, padding: '14px 18px', borderRadius: 8,
          background: `${CYAN}0c`, border: `1px solid ${CYAN}33`,
        }}>
          <div style={{ fontSize: 11, fontFamily: M, color: CYAN, marginBottom: 5, letterSpacing: 1, textTransform: 'uppercase' }}>
            Key Insight
          </div>
          <div style={{ fontSize: 13, color: '#d0d0d0', lineHeight: 1.65 }}>
            Only <strong style={{ color: CYAN }}>4 components</strong> change from Triton to CUDA kernels:{' '}
            <span style={{ fontFamily: M, color: LIME }}>DSA Indexer</span>,{' '}
            <span style={{ fontFamily: M, color: CYAN }}>DSA Sparse Attention</span>,{' '}
            <span style={{ fontFamily: M, color: CYAN }}>MLA Attention</span>, and{' '}
            <span style={{ fontFamily: M, color: LIME }}>MoE GEMM</span>.{' '}
            Everything else is kept as-is because Triton or PyTorch is already fast enough for those ops.
            The CUDA tier is about unlocking <strong style={{ color: AMBER }}>FP8 tensor cores</strong> and{' '}
            <strong style={{ color: CYAN }}>paged KV cache</strong> — not rewriting everything.
          </div>
        </div>

        {/* ── Tabs ── */}
        <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
          {[
            { id: 'table', label: 'Component Table' },
            { id: 'perf',  label: 'Performance Estimates' },
            { id: 'gpu',   label: 'GPU Requirements' },
          ].map(({ id, label }) => (
            <button key={id} onClick={() => setActiveTab(id)} style={{
              padding: '7px 18px', borderRadius: 6,
              border: `1px solid ${activeTab === id ? CYAN : 'rgba(255,255,255,.1)'}`,
              background: activeTab === id ? `${CYAN}12` : 'rgba(255,255,255,.03)',
              color: activeTab === id ? CYAN : '#777',
              fontFamily: M, fontSize: 11, cursor: 'pointer',
            }}>
              {label}
            </button>
          ))}

          {/* Show code paths toggle — only relevant on table tab */}
          {activeTab === 'table' && (
            <button onClick={() => setShowCode(v => !v)} style={{
              marginLeft: 'auto', padding: '7px 18px', borderRadius: 6,
              border: `1px solid ${showCode ? AMBER + '88' : 'rgba(255,255,255,.1)'}`,
              background: showCode ? `${AMBER}14` : 'rgba(255,255,255,.03)',
              color: showCode ? AMBER : '#666',
              fontFamily: M, fontSize: 11, cursor: 'pointer',
            }}>
              {showCode ? 'Hide code paths' : 'Show code paths'}
            </button>
          )}
        </div>

        {/* ══════════════════════════════════════════════════════
            TAB: Component Table
            ════════════════════════════════════════════════════ */}
        {activeTab === 'table' && (
          <div>
            {/* Legend */}
            <div style={{ display: 'flex', gap: 10, marginBottom: 16, flexWrap: 'wrap', alignItems: 'center' }}>
              <span style={{ fontSize: 11, color: '#555', fontFamily: M }}>Legend:</span>
              {[
                { tier: 'pytorch',  desc: 'Raw PyTorch' },
                { tier: 'triton',   desc: 'Unsloth Triton' },
                { tier: 'flashmla', desc: 'FlashMLA CUDA' },
                { tier: 'deepgemm', desc: 'DeepGEMM FP8' },
                { tier: 'kept',     desc: 'Kept as-is (Triton/PT)' },
              ].map(({ tier, desc }) => (
                <div key={tier} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
                  <TierBadge tier={tier} />
                  <span style={{ fontSize: 10, color: '#555' }}>{desc}</span>
                </div>
              ))}
            </div>

            {/* Column headers */}
            <div style={{
              display: 'grid', gridTemplateColumns: '160px 1fr 1fr 1fr',
              gap: 1, marginBottom: 2,
            }}>
              {['Component', 'PyTorch (raw)', 'Triton (Unsloth)', 'CUDA Kernels'].map((h, i) => (
                <div key={h} style={{
                  padding: '8px 12px', fontSize: 10, fontFamily: M,
                  color: i === 0 ? '#555' : [GRAY, AMBER, CYAN][i - 1],
                  textTransform: 'uppercase', letterSpacing: 1,
                  background: 'rgba(255,255,255,.02)',
                  borderRadius: i === 0 ? '6px 0 0 6px' : i === 3 ? '0 6px 6px 0' : 0,
                  borderTop: '1px solid rgba(255,255,255,.06)',
                  borderBottom: '1px solid rgba(255,255,255,.06)',
                  borderLeft: i === 0 ? '1px solid rgba(255,255,255,.06)' : 'none',
                  borderRight: i === 3 ? '1px solid rgba(255,255,255,.06)' : 'none',
                }}>
                  {h}
                </div>
              ))}
            </div>

            {/* Rows */}
            {ROWS.map((row) => {
              const isOpen = expanded === row.id;
              const det = DETAIL[row.id];
              return (
                <div key={row.id} style={{ marginBottom: 2 }}>
                  {/* Main row */}
                  <div
                    onClick={() => toggle(row.id)}
                    style={{
                      display: 'grid', gridTemplateColumns: '160px 1fr 1fr 1fr',
                      gap: 1, cursor: 'pointer',
                      background: isOpen ? 'rgba(34,211,238,.04)' : 'rgba(255,255,255,.015)',
                      border: `1px solid ${isOpen ? CYAN + '33' : 'rgba(255,255,255,.05)'}`,
                      borderRadius: isOpen ? '6px 6px 0 0' : 6,
                      transition: 'all .15s',
                    }}
                  >
                    {/* Component name */}
                    <div style={{ padding: '10px 12px', display: 'flex', alignItems: 'center', gap: 6 }}>
                      <span style={{
                        fontSize: 10, color: isOpen ? CYAN : '#888', fontFamily: M, fontWeight: 600,
                      }}>
                        {row.name}
                      </span>
                      <span style={{ fontSize: 10, color: isOpen ? CYAN + 'aa' : '#444', fontFamily: M }}>
                        {isOpen ? '▲' : '▼'}
                      </span>
                    </div>
                    {/* Three tier cells */}
                    {[row.pytorch, row.triton, row.cuda].map((cell, ci) => (
                      <div key={ci} style={{ padding: '10px 12px', borderLeft: '1px solid rgba(255,255,255,.04)' }}>
                        <CellContent tier={cell.tier} text={cell.text} sub={cell.sub} />
                      </div>
                    ))}
                  </div>

                  {/* Expanded detail panel */}
                  {isOpen && det && (
                    <div style={{
                      padding: '16px 18px',
                      background: 'rgba(34,211,238,.02)',
                      border: `1px solid ${CYAN}22`,
                      borderTop: 'none',
                      borderRadius: '0 0 6px 6px',
                      marginBottom: 4,
                    }}>
                      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16, marginBottom: 12 }}>
                        <div>
                          <div style={{ fontSize: 10, fontFamily: M, color: '#555', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>File</div>
                          <div style={{ fontFamily: M, fontSize: 11, color: AMBER }}>{det.file}</div>
                        </div>
                        <div>
                          <div style={{ fontSize: 10, fontFamily: M, color: '#555', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>Key Constraint</div>
                          <div style={{ fontSize: 12, color: '#bbb', lineHeight: 1.5 }}>{det.constraint}</div>
                        </div>
                      </div>
                      <div style={{ marginBottom: showCode ? 12 : 0 }}>
                        <div style={{ fontSize: 10, fontFamily: M, color: '#555', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>API Signature</div>
                        <div style={{ fontFamily: M, fontSize: 11, color: CYAN, padding: '6px 10px', background: 'rgba(0,0,0,.3)', borderRadius: 5, display: 'inline-block' }}>
                          {det.sig}
                        </div>
                      </div>
                      {showCode && (
                        <div style={{ marginTop: 12 }}>
                          <div style={{ fontSize: 10, fontFamily: M, color: '#555', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 6 }}>Code Path</div>
                          <HL code={det.code} />
                        </div>
                      )}
                    </div>
                  )}
                </div>
              );
            })}

            {/* Math: what actually changes */}
            <MathBlock label="Components that change from Triton → CUDA (the 4)">
              <Var color={LIME}>DSA Indexer</Var>
              {' → '}
              <Var color={LIME}>fp8_mqa_logits</Var>
              {'   '}
              <Var color={CYAN}>DSA Sparse</Var>
              {' → '}
              <Var color={CYAN}>flash_mla_sparse_fwd</Var>
              <br />
              <Var color={CYAN}>MLA Attn</Var>
              {' → '}
              <Var color={CYAN}>flash_mla_with_kvcache</Var>
              {'   '}
              <Var color={LIME}>MoE GEMM</Var>
              {' → '}
              <Var color={LIME}>m_grouped_fp8_gemm</Var>
            </MathBlock>
          </div>
        )}

        {/* ══════════════════════════════════════════════════════
            TAB: Performance Estimates
            ════════════════════════════════════════════════════ */}
        {activeTab === 'perf' && (
          <div>
            <div style={{ fontSize: 13, color: '#888', lineHeight: 1.65, marginBottom: 20, maxWidth: 680 }}>
              Theoretical peak speedups on H100 (SM90). Numbers derived from H100 FP8 tensor core throughput (1979 TFLOPS) vs H100 BF16/FP16 (989 TFLOPS) and per-op memory-bandwidth analysis. Actual speedup depends on sequence length and batch size.
            </div>

            {PERF.map((p) => (
              <div key={p.label} style={{ ...CARD, marginBottom: 12 }}>
                <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', marginBottom: 10, flexWrap: 'wrap', gap: 8 }}>
                  <div>
                    <div style={{ fontFamily: M, fontSize: 13, color: p.color, fontWeight: 700, marginBottom: 3 }}>
                      {p.label}
                    </div>
                    <div style={{ fontFamily: M, fontSize: 10, color: '#555' }}>
                      {p.kernel} <span style={{ color: '#444' }}>vs</span> {p.vs}
                    </div>
                  </div>
                  <div style={{
                    fontFamily: M, fontSize: 26, fontWeight: 700, color: p.color,
                    padding: '4px 14px', borderRadius: 6,
                    background: `${p.color}12`, border: `1px solid ${p.color}33`,
                  }}>
                    {p.speedup}
                  </div>
                </div>
                {/* Bar */}
                <div style={{ height: 6, borderRadius: 3, background: 'rgba(255,255,255,.05)', marginBottom: 8, overflow: 'hidden' }}>
                  <div style={{
                    height: '100%', width: `${p.bar * 100}%`,
                    background: p.color, borderRadius: 3,
                    opacity: 0.7,
                    transition: 'width 0.4s ease',
                  }} />
                </div>
                <div style={{ fontSize: 11, color: '#777', fontFamily: M }}>{p.detail}</div>
              </div>
            ))}

            {/* Math: FlashMLA TFLOPS derivation */}
            <MathBlock label="FlashMLA decode throughput (H100 FP8)">
              <Var color={CYAN}>TFLOPS</Var>
              {' = '}
              <Var color={AMBER}>2</Var>
              {' × '}
              <Var>B</Var>
              {' × '}
              <Var>H<Sub>q</Sub></Var>
              {' × '}
              <Var>S<Sub>kv</Sub></Var>
              {' × '}
              <Var>d<Sub>head</Sub></Var>
              {' / '}
              <Var color={PINK}>t<Sub>kernel</Sub></Var>
              <br />
              <span style={{ fontSize: 13, color: '#666' }}>
                ≈ 660 TFLOPS effective (H100 FP8) vs ~30 TFLOPS eager BF16
              </span>
            </MathBlock>

            <MathBlock label="DeepGEMM MoE GEMM throughput">
              <Var color={LIME}>TFLOPS</Var>
              {' = '}
              <Var>E<Sub>active</Sub></Var>
              {' × '}
              <Var>2</Var>
              {' × '}
              <Var>d<Sub>model</Sub></Var>
              {' × '}
              <Var>d<Sub>ff</Sub></Var>
              {' × '}
              <Var>tokens</Var>
              {' / '}
              <Var color={PINK}>t</Var>
              <br />
              <span style={{ fontSize: 13, color: '#666' }}>
                ≈ 1550 TFLOPS FP8 (8 active experts × d_ff=4096 × d_model=7168)
              </span>
            </MathBlock>

            {/* Stacked bar: relative contribution */}
            <div style={{ ...CARD, marginTop: 4 }}>
              <div style={{ fontSize: 11, fontFamily: M, color: '#555', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>
                Relative decode latency contribution (per layer, prefill excluded)
              </div>
              {[
                { label: 'MLA Attention',   pct: 42, color: CYAN   },
                { label: 'MoE GEMM',        pct: 31, color: LIME   },
                { label: 'DSA Sparse Attn', pct: 14, color: CYAN   },
                { label: 'DSA Indexer',     pct:  8, color: LIME   },
                { label: 'RMSNorm+SwiGLU',  pct:  5, color: AMBER  },
              ].map(({ label, pct, color }) => (
                <div key={label} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                  <div style={{ fontFamily: M, fontSize: 10, color: '#666', minWidth: 130 }}>{label}</div>
                  <div style={{ flex: 1, height: 14, borderRadius: 3, background: 'rgba(255,255,255,.04)', overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${pct}%`, background: color, opacity: 0.65, borderRadius: 3 }} />
                  </div>
                  <div style={{ fontFamily: M, fontSize: 11, color, minWidth: 34, textAlign: 'right' }}>{pct}%</div>
                </div>
              ))}
            </div>
          </div>
        )}

        {/* ══════════════════════════════════════════════════════
            TAB: GPU Requirements
            ════════════════════════════════════════════════════ */}
        {activeTab === 'gpu' && (
          <div>
            <div style={{ fontSize: 13, color: '#888', lineHeight: 1.65, marginBottom: 20, maxWidth: 680 }}>
              The optimization stack is tiered by CUDA compute capability. Each tier is a strict superset of the previous — SM90 can run all SM80 kernels but adds FP8 tensor core paths. The runtime selects paths via <code style={{ fontFamily: M, color: AMBER, fontSize: 12 }}>__CUDA_ARCH__</code> guards.
            </div>

            {GPU_TIERS.map((tier, ti) => (
              <div key={tier.sm} style={{
                ...CARD,
                marginBottom: 12,
                borderColor: `${tier.color}33`,
                background: `${tier.color}06`,
              }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 12 }}>
                  <div style={{
                    fontFamily: M, fontSize: 18, fontWeight: 800, color: tier.color,
                    padding: '6px 14px', borderRadius: 6,
                    background: `${tier.color}14`, border: `1px solid ${tier.color}44`,
                    letterSpacing: 1,
                  }}>
                    {tier.sm}
                  </div>
                  <div>
                    <div style={{ fontFamily: M, fontSize: 14, color: '#e0e0e0', fontWeight: 700 }}>{tier.chip}</div>
                    <div style={{ fontSize: 11, color: '#555' }}>{tier.note}</div>
                  </div>
                  {ti > 0 && (
                    <div style={{ marginLeft: 'auto', fontFamily: M, fontSize: 10, color: tier.color, padding: '3px 8px', borderRadius: 4, border: `1px solid ${tier.color}44`, whiteSpace: 'nowrap' }}>
                      + {ti === 1 ? '4' : '2'} new paths
                    </div>
                  )}
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {tier.kernels.map((k, ki) => (
                    <div key={ki} style={{ display: 'flex', alignItems: 'flex-start', gap: 8 }}>
                      <span style={{ color: tier.color, fontFamily: M, fontSize: 11, opacity: 0.6, flexShrink: 0 }}>→</span>
                      <span style={{ fontFamily: M, fontSize: 11, color: ki === 0 && ti > 0 ? '#444' : '#bbb' }}>{k}</span>
                    </div>
                  ))}
                </div>
              </div>
            ))}

            {/* Compatibility matrix */}
            <div style={{ ...CARD, marginTop: 8 }}>
              <div style={{ fontSize: 11, fontFamily: M, color: '#555', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>
                Kernel Compatibility Matrix
              </div>
              <div style={{ overflowX: 'auto' }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontFamily: M, fontSize: 11 }}>
                  <thead>
                    <tr>
                      {['Kernel', 'SM80 (A100)', 'SM90 (H100)', 'SM100 (B200)'].map((h, i) => (
                        <th key={h} style={{
                          padding: '8px 12px', textAlign: i === 0 ? 'left' : 'center',
                          color: '#555', fontSize: 10, textTransform: 'uppercase', letterSpacing: 1,
                          borderBottom: '1px solid rgba(255,255,255,.07)',
                          fontWeight: 600,
                        }}>
                          {h}
                        </th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {[
                      { kernel: 'Unsloth RMSNorm (Triton)',     sm80: true,  sm90: true,  sm100: true  },
                      { kernel: 'Unsloth SwiGLU (Triton)',       sm80: true,  sm90: true,  sm100: true  },
                      { kernel: 'Unsloth Cross-Entropy (Triton)',sm80: true,  sm90: true,  sm100: true  },
                      { kernel: 'Unsloth grouped GEMM (Triton)', sm80: true,  sm90: true,  sm100: true  },
                      { kernel: 'flash_mla_with_kvcache',        sm80: false, sm90: true,  sm100: true  },
                      { kernel: 'flash_mla_sparse_fwd',          sm80: false, sm90: true,  sm100: true  },
                      { kernel: 'fp8_mqa_logits (DeepGEMM)',     sm80: false, sm90: true,  sm100: true  },
                      { kernel: 'm_grouped_fp8_gemm (DeepGEMM)', sm80: false, sm90: true,  sm100: true  },
                      { kernel: 'Paged FP8 KV cache',            sm80: false, sm90: true,  sm100: true  },
                      { kernel: 'SM100 sparse prefill',          sm80: false, sm90: false, sm100: true  },
                    ].map((r, i) => (
                      <tr key={r.kernel} style={{ background: i % 2 === 0 ? 'transparent' : 'rgba(255,255,255,.012)' }}>
                        <td style={{ padding: '8px 12px', color: '#bbb', fontSize: 11 }}>{r.kernel}</td>
                        {[r.sm80, r.sm90, r.sm100].map((ok, ci) => (
                          <td key={ci} style={{ padding: '8px 12px', textAlign: 'center' }}>
                            <span style={{
                              fontFamily: M, fontSize: 13,
                              color: ok ? LIME : '#333',
                            }}>
                              {ok ? '✓' : '—'}
                            </span>
                          </td>
                        ))}
                      </tr>
                    ))}
                  </tbody>
                </table>
              </div>
            </div>

            {/* CUDA arch guard code snippet */}
            <div style={{ marginTop: 14 }}>
              <HL code={`// Runtime arch detection — flashmla / deepgemm gating
#if defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 900
  // SM90+ path: use FP8 tensor cores via WGMMA + TMA
  flash_mla_with_kvcache_sm90(q, kv_cache, ...);
#elif defined(__CUDA_ARCH__) && __CUDA_ARCH__ >= 800
  // SM80 fallback: BF16 eager attention + Triton grouped GEMM
  eager_attention_bf16(q, k, v, ...);
#endif

// Python side: detected at import time
import torch
SM = torch.cuda.get_device_capability()
USE_FLASHMLA   = SM >= (9, 0)   # H100+
USE_DEEPGEMM   = SM >= (9, 0)   # H100+ (FP8 TC)
USE_SM100_OPTS = SM >= (10, 0)  # B200+`} />
            </div>
          </div>
        )}

        {/* ── Resources ── */}
        <div style={{ ...CARD, marginTop: 28 }}>
          <div style={{ fontSize: 10, fontFamily: M, color: '#444', textTransform: 'uppercase', letterSpacing: 1.5, marginBottom: 12 }}>
            Resources
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
            {RESOURCES.map(({ label, url, color }) => (
              <a key={url} href={url} target="_blank" rel="noopener noreferrer" style={{
                display: 'flex', alignItems: 'flex-start', gap: 8,
                color, fontSize: 12, textDecoration: 'none', lineHeight: 1.5,
              }}>
                <span style={{ opacity: 0.4, flexShrink: 0, fontFamily: M }}>→</span>
                <span style={{ borderBottom: `1px solid ${color}33` }}>{label}</span>
              </a>
            ))}
          </div>
        </div>

      </div>
    </div>
  );
}
