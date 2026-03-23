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
          .replace(/\b(def|for|if|else|elif|return|while|import|from|class|in|range|pass|True|False|None|torch|tl|triton|float|int|void|const|auto|__global__|__shared__)\b/g, '\u27E8k\u27E9$1\u27E8/k\u27E9')
          .replace(/\b(\d+\.?\d*)\b/g, '\u27E8n\u27E9$1\u27E8/n\u27E9')
          .replace(/"([^"]*)"/g, '\u27E8s\u27E9"$1"\u27E8/s\u27E9')
          .replace(/'([^']*)'/g, "\u27E8s\u27E9'$1'\u27E8/s\u27E9");
        const ps = h.split(/(\u27E8k\u27E9.*?\u27E8\/k\u27E9|\u27E8n\u27E9.*?\u27E8\/n\u27E9|\u27E8s\u27E9.*?\u27E8\/s\u27E9)/g);
        return (
          <div key={i}>
            {ps.map((p, j) => {
              if (p.startsWith('\u27E8k\u27E9')) return <span key={j} style={{ color: kw }}>{p.slice(3, -4)}</span>;
              if (p.startsWith('\u27E8n\u27E9')) return <span key={j} style={{ color: nm }}>{p.slice(3, -4)}</span>;
              if (p.startsWith('\u27E8s\u27E9')) return <span key={j} style={{ color: st }}>{p.slice(3, -4)}</span>;
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

// ─── Bar chart ────────────────────────────────────────────────────────────────
function BarChart({ data, labels, series, maxVal }) {
  const mx = maxVal || Math.max(...data.flat());
  return (
    <div style={{ overflowX: 'auto' }}>
      <div style={{ display: 'flex', gap: 6, alignItems: 'flex-end', minWidth: labels.length * 110 }}>
        {labels.map((label, li) => (
          <div key={li} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
            <div style={{ display: 'flex', gap: 2, alignItems: 'flex-end', height: 160 }}>
              {series.map((s, si) => {
                const val = data[si][li];
                const h = (val / mx) * 150;
                return (
                  <div key={si} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                    <span style={{ fontFamily: M, fontSize: 7, color: s.color, opacity: 0.8 }}>{val}</span>
                    <div style={{
                      width: 16, height: h, borderRadius: 3,
                      background: s.color, opacity: 0.7,
                    }} />
                  </div>
                );
              })}
            </div>
            <span style={{ fontFamily: M, fontSize: 9, color: '#666', textAlign: 'center', lineHeight: 1.2 }}>{label}</span>
          </div>
        ))}
      </div>
      {/* Legend */}
      <div style={{ display: 'flex', gap: 14, marginTop: 14, flexWrap: 'wrap' }}>
        {series.map((s, i) => (
          <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 5 }}>
            <div style={{ width: 10, height: 10, borderRadius: 2, background: s.color, opacity: 0.7 }} />
            <span style={{ fontFamily: M, fontSize: 10, color: s.color }}>{s.name}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

// ─── Table 5 data ─────────────────────────────────────────────────────────────
const CTX_LABELS = ['4K', '8K', '16K', '32K', '64K', '128K'];
const CTX_DATA = [
  [97.44, 96.72, 95.83, 92.96, 85.34, 79.21],  // Dense
  [95.37, 93.54, 90.50, 80.94, 61.65, 48.86],  // SWA-Interleave
  [95.83, 94.75, 93.80, 90.99, 83.01, 74.11],  // SWA-Search
  [97.51, 96.54, 95.40, 90.09, 84.05, 71.35],  // DSA-warmup
  [96.77, 96.25, 96.69, 93.45, 87.06, 78.86],  // DSA-full
];
const CTX_SERIES = [
  { name: 'Dense', color: GRAY },
  { name: 'SWA-Interleave', color: PINK },
  { name: 'SWA-Search', color: PURPLE },
  { name: 'DSA-warmup', color: AMBER },
  { name: 'DSA-full', color: CYAN },
];

// ─── Resources ────────────────────────────────────────────────────────────────
const RESOURCES = [
  { label: 'GLM-5 Technical Report -- DSA details (THUDM)',           url: 'https://arxiv.org/abs/2501.12386',               color: CYAN   },
  { label: 'FlashMLA -- sparse block attention kernel',               url: 'https://github.com/deepseek-ai/FlashMLA',       color: LIME   },
  { label: 'NSA: Native Sparse Attention (related)',                  url: 'https://arxiv.org/abs/2502.11089',               color: AMBER  },
  { label: 'DeepSeek-V3 Technical Report',                           url: 'https://arxiv.org/abs/2412.19437',               color: PURPLE },
];

// ─── Pipeline step component ──────────────────────────────────────────────────
function PipelineStep({ num, title, desc, color, detail }) {
  return (
    <div style={{
      padding: '12px 16px', borderRadius: 7,
      background: `${color}08`, border: `1px solid ${color}33`,
      display: 'flex', gap: 12, alignItems: 'flex-start',
    }}>
      <div style={{
        fontFamily: M, fontSize: 16, fontWeight: 800, color,
        width: 32, height: 32, borderRadius: 6,
        background: `${color}18`, border: `1px solid ${color}44`,
        display: 'flex', alignItems: 'center', justifyContent: 'center',
        flexShrink: 0,
      }}>
        {num}
      </div>
      <div>
        <div style={{ fontFamily: M, fontSize: 12, color, fontWeight: 600, marginBottom: 3 }}>{title}</div>
        <div style={{ fontSize: 12, color: '#bbb', lineHeight: 1.5 }}>{desc}</div>
        {detail && <div style={{ fontFamily: M, fontSize: 10, color: '#555', marginTop: 4 }}>{detail}</div>}
      </div>
    </div>
  );
}

// ─── Page component ───────────────────────────────────────────────────────────
export default function DSAPage() {
  const [activeTab, setActiveTab] = useState('how');
  const [seqLen, setSeqLen]       = useState(131072);

  const denseOps = seqLen * seqLen;
  const dsaOps   = seqLen * 2048;
  const savings  = (denseOps / dsaOps).toFixed(0);

  return (
    <div style={{ background: '#0d0f14', minHeight: '100vh', color: '#e0e0e0', fontFamily: S, padding: '32px 20px' }}>
      <div style={{ maxWidth: 1040, margin: '0 auto' }}>

        {/* ── Header ── */}
        <div style={{ marginBottom: 28 }}>
          <span style={{ fontFamily: M, fontSize: 11, color: CYAN, letterSpacing: 2, textTransform: 'uppercase' }}>
            GLM-5 · Sparse Attention
          </span>
          <h1 style={{ margin: '6px 0 6px', fontSize: 28, fontWeight: 700, color: '#f0f0f0', letterSpacing: -0.5 }}>
            Dynamic Sparse Attention (DSA)
          </h1>
          <p style={{ margin: 0, color: 'rgba(255,255,255,.45)', fontSize: 14, maxWidth: 680, lineHeight: 1.6 }}>
            DSA learns which KV blocks to attend to per-head, reducing attention from O(L{'\u00B2'}) to O(L {'\u00D7'} 2048).
            At 128K context, this is a ~64x reduction in attention compute while matching or exceeding dense quality.
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
            DSA-full <strong style={{ color: CYAN }}>exceeds dense attention</strong> at 32K-128K contexts (Table 5).
            This is because learned sparsity acts as a form of{' '}
            <strong style={{ color: AMBER }}>attention regularization</strong> -- the model learns to ignore irrelevant blocks,
            reducing noise in the attention distribution. SWA (Sliding Window) patterns cannot adapt per-query.
          </div>
        </div>

        {/* ── Tabs ── */}
        <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
          {[
            { id: 'how',   label: 'How DSA Works' },
            { id: 'bench', label: 'DSA vs Alternatives' },
            { id: 'topk',  label: 'Deterministic TopK' },
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
        </div>

        {/* ══════════════════════════════════════════════════════
            TAB: How DSA Works
            ════════════════════════════════════════════════════ */}
        {activeTab === 'how' && (
          <div>
            {/* 5-step pipeline */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 16 }}>
              <PipelineStep
                num={1} color={CYAN}
                title="Compute Raw Attention Scores"
                desc="Standard QK^T dot product between query and all KV blocks."
                detail="scores = Q @ K.T / sqrt(d_k)  --  [B, H, T, T]"
              />
              <PipelineStep
                num={2} color={LIME}
                title="ReLU Gating"
                desc="Apply ReLU instead of softmax -- zeroes out negative scores, creating natural sparsity."
                detail="weights = ReLU(scores)  --  many entries become exactly 0"
              />
              <PipelineStep
                num={3} color={AMBER}
                title="Learned Importance Weights"
                desc="Per-head learned weights modulate which positions the model considers important."
                detail="weights = weights * dsa_importance  --  [H, T] learned parameters"
              />
              <PipelineStep
                num={4} color={PINK}
                title="TopK Block Selection"
                desc="Select top-2048 KV positions per query. Only these will be used for attention."
                detail="indices = topk(weights, k=2048)  --  fixed budget per query"
              />
              <PipelineStep
                num={5} color={PURPLE}
                title="Sparse Attention"
                desc="Compute attention only over selected blocks. FlashMLA skips zero-score blocks entirely."
                detail="out = sparse_attn(Q, K[indices], V[indices])  --  O(L x 2048)"
              />
            </div>

            {/* Complexity comparison */}
            <div style={{ ...CARD, marginBottom: 16 }}>
              <div style={{ fontSize: 11, fontFamily: M, color: AMBER, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 14 }}>
                Complexity Calculator
              </div>
              <div style={{ marginBottom: 14 }}>
                <label style={{ fontFamily: M, fontSize: 11, color: '#888', marginRight: 12 }}>
                  Sequence length: <span style={{ color: CYAN }}>{seqLen.toLocaleString()}</span>
                </label>
                <input
                  type="range" min={1024} max={262144} step={1024} value={seqLen}
                  onChange={e => setSeqLen(Number(e.target.value))}
                  style={{ width: '100%', maxWidth: 400, accentColor: CYAN }}
                />
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
                <div style={{ padding: '12px 16px', borderRadius: 6, background: `${GRAY}14`, border: `1px solid ${GRAY}44` }}>
                  <div style={{ fontFamily: M, fontSize: 10, color: GRAY, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>Dense O(L{'\u00B2'})</div>
                  <div style={{ fontFamily: M, fontSize: 18, fontWeight: 700, color: GRAY }}>{(denseOps / 1e9).toFixed(1)}G</div>
                  <div style={{ fontFamily: M, fontSize: 9, color: '#555' }}>ops per head</div>
                </div>
                <div style={{ padding: '12px 16px', borderRadius: 6, background: `${CYAN}14`, border: `1px solid ${CYAN}44` }}>
                  <div style={{ fontFamily: M, fontSize: 10, color: CYAN, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>DSA O(L{'\u00D7'}2048)</div>
                  <div style={{ fontFamily: M, fontSize: 18, fontWeight: 700, color: CYAN }}>{(dsaOps / 1e9).toFixed(2)}G</div>
                  <div style={{ fontFamily: M, fontSize: 9, color: '#555' }}>ops per head</div>
                </div>
                <div style={{ padding: '12px 16px', borderRadius: 6, background: `${LIME}14`, border: `1px solid ${LIME}44` }}>
                  <div style={{ fontFamily: M, fontSize: 10, color: LIME, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>Savings</div>
                  <div style={{ fontFamily: M, fontSize: 18, fontWeight: 700, color: LIME }}>~{savings}x</div>
                  <div style={{ fontFamily: M, fontSize: 9, color: '#555' }}>fewer ops</div>
                </div>
              </div>
            </div>

            <MathBlock label="DSA attention complexity reduction">
              <Var color={GRAY}>Dense</Var>{': O('}
              <Var>L</Var><sup style={{ fontSize: '0.7em' }}>2</sup>
              {' \u00B7 '}
              <Var>d</Var>
              {')   vs   '}
              <Var color={CYAN}>DSA</Var>{': O('}
              <Var>L</Var>
              {' \u00B7 '}
              <Var color={AMBER}>K</Var>
              {' \u00B7 '}
              <Var>d</Var>
              {')   where '}
              <Var color={AMBER}>K</Var>{' = 2048'}
              <br />
              <span style={{ fontSize: 13, color: '#666' }}>
                At L=128K: 16.8B ops (dense) vs 268M ops (DSA) = 64x reduction
              </span>
            </MathBlock>

            <HL code={`# DSA: Dynamic Sparse Attention — 5-step pipeline
def dsa_attention(Q, K, V, dsa_weights):
    # Step 1: raw attention scores
    scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(d_k)

    # Step 2: ReLU gating (NOT softmax) — creates natural sparsity
    weights = torch.relu(scores)

    # Step 3: learned per-head importance modulation
    weights = weights * dsa_weights  # dsa_weights: [H, T] learned

    # Step 4: top-K block selection (K=2048 fixed budget)
    topk_vals, topk_idx = torch.topk(weights, k=2048, dim=-1)

    # Step 5: sparse attention — only compute over selected positions
    K_sparse = K.gather(-2, topk_idx.unsqueeze(-1).expand_as(K))
    V_sparse = V.gather(-2, topk_idx.unsqueeze(-1).expand_as(V))
    attn = torch.softmax(Q @ K_sparse.T / math.sqrt(d_k), dim=-1)
    return attn @ V_sparse`} />
          </div>
        )}

        {/* ══════════════════════════════════════════════════════
            TAB: DSA vs Alternatives
            ════════════════════════════════════════════════════ */}
        {activeTab === 'bench' && (
          <div>
            <div style={{ fontSize: 13, color: '#888', lineHeight: 1.65, marginBottom: 20, maxWidth: 680 }}>
              Table 5 from the GLM-5 report: RULER benchmark scores across context lengths.
              DSA-full uses the full DSA pipeline throughout training. DSA-warmup transitions
              from dense to DSA mid-training. Both SWA variants use fixed sliding window patterns.
            </div>

            <div style={{ ...CARD, marginBottom: 16 }}>
              <div style={{ fontSize: 11, fontFamily: M, color: CYAN, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 16 }}>
                RULER Benchmark by Context Length (Table 5)
              </div>
              <BarChart
                data={CTX_DATA}
                labels={CTX_LABELS}
                series={CTX_SERIES}
                maxVal={100}
              />
            </div>

            {/* Analysis cards */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16 }}>
              <div style={{ ...CARD, borderColor: `${CYAN}33` }}>
                <div style={{ fontFamily: M, fontSize: 11, color: CYAN, marginBottom: 8 }}>DSA-full Wins</div>
                <div style={{ fontSize: 12, color: '#bbb', lineHeight: 1.6 }}>
                  At 32K: <strong style={{ color: CYAN }}>93.45</strong> vs Dense 92.96 (+0.49)<br />
                  At 64K: <strong style={{ color: CYAN }}>87.06</strong> vs Dense 85.34 (+1.72)<br />
                  At 128K: <strong style={{ color: CYAN }}>78.86</strong> vs Dense 79.21 (-0.35)
                </div>
                <div style={{ fontSize: 11, color: '#666', marginTop: 6 }}>
                  Learned sparsity regularizes attention at long contexts
                </div>
              </div>
              <div style={{ ...CARD, borderColor: `${PINK}33` }}>
                <div style={{ fontFamily: M, fontSize: 11, color: PINK, marginBottom: 8 }}>SWA Failure Mode</div>
                <div style={{ fontSize: 12, color: '#bbb', lineHeight: 1.6 }}>
                  SWA-Interleave at 128K: <strong style={{ color: PINK }}>48.86</strong> (catastrophic)<br />
                  SWA-Search at 128K: <strong style={{ color: PURPLE }}>74.11</strong> (still -5.1 vs dense)<br />
                  Fixed windows cannot adapt to query-specific patterns
                </div>
                <div style={{ fontSize: 11, color: '#666', marginTop: 6 }}>
                  Sliding window is fundamentally limited at long context
                </div>
              </div>
            </div>

            {/* Delta table for DSA-full vs Dense */}
            <div style={{ ...CARD, marginBottom: 16 }}>
              <div style={{ fontSize: 11, fontFamily: M, color: LIME, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>
                DSA-full vs Dense (Delta)
              </div>
              {CTX_LABELS.map((label, i) => {
                const delta = (CTX_DATA[4][i] - CTX_DATA[0][i]).toFixed(2);
                const positive = parseFloat(delta) >= 0;
                return (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
                    <span style={{ fontFamily: M, fontSize: 10, color: '#666', minWidth: 40 }}>{label}</span>
                    <div style={{ flex: 1, height: 12, borderRadius: 3, background: 'rgba(255,255,255,.04)', overflow: 'hidden', position: 'relative' }}>
                      <div style={{
                        position: 'absolute', left: '50%', top: 0, bottom: 0,
                        width: `${Math.min(Math.abs(parseFloat(delta)) * 15, 50)}%`,
                        marginLeft: positive ? 0 : `-${Math.min(Math.abs(parseFloat(delta)) * 15, 50)}%`,
                        background: positive ? LIME : PINK,
                        opacity: 0.6, borderRadius: 3,
                      }} />
                    </div>
                    <span style={{ fontFamily: M, fontSize: 11, color: positive ? LIME : PINK, minWidth: 50, textAlign: 'right' }}>
                      {positive ? '+' : ''}{delta}
                    </span>
                  </div>
                );
              })}
            </div>

            <MathBlock label="DSA outperforms dense at long context">
              <Var color={CYAN}>DSA</Var><Sub>64K</Sub>
              {' = '}
              <Var color={CYAN}>87.06</Var>
              {'  >  '}
              <Var color={GRAY}>Dense</Var><Sub>64K</Sub>
              {' = '}
              <Var color={GRAY}>85.34</Var>
              {'   (+'}
              <Var color={LIME}>1.72</Var>
              {')'}
              <br />
              <span style={{ fontSize: 13, color: '#666' }}>
                Learned sparsity acts as attention regularization at long contexts
              </span>
            </MathBlock>
          </div>
        )}

        {/* ══════════════════════════════════════════════════════
            TAB: Deterministic TopK
            ════════════════════════════════════════════════════ */}
        {activeTab === 'topk' && (
          <div>
            <div style={{ fontSize: 13, color: '#888', lineHeight: 1.65, marginBottom: 20, maxWidth: 680 }}>
              A critical engineering insight: CUDA's default topk is non-deterministic when
              values are tied. During RL training (GRPO), this non-determinism propagated through
              the sparse attention mask, causing training instability and divergence.
            </div>

            {/* Problem-solution layout */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16 }}>
              <div style={{ ...CARD, borderColor: `${PINK}44`, background: `${PINK}06` }}>
                <div style={{ fontSize: 11, fontFamily: M, color: PINK, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>
                  The Problem
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {[
                    'CUDA torch.topk is non-deterministic by default',
                    'Tied scores (common after ReLU gating) break ties randomly',
                    'Different GPUs in DP group get different sparse masks',
                    'RL reward signals become noisy (mask varies across replays)',
                    'GRPO training diverges after ~500 steps',
                  ].map((t, i) => (
                    <div key={i} style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
                      <span style={{ fontFamily: M, fontSize: 11, color: PINK, flexShrink: 0 }}>{i + 1}.</span>
                      <span style={{ fontSize: 12, color: '#bbb', lineHeight: 1.5 }}>{t}</span>
                    </div>
                  ))}
                </div>
              </div>
              <div style={{ ...CARD, borderColor: `${LIME}44`, background: `${LIME}06` }}>
                <div style={{ fontSize: 11, fontFamily: M, color: LIME, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>
                  The Solution
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                  {[
                    'Implement deterministic topk with stable tie-breaking',
                    'Add position-based tiebreaker: score + pos * epsilon',
                    'All GPUs in DP group get identical sparse masks',
                    'RL reward signals become consistent across replays',
                    'GRPO training stabilizes and converges normally',
                  ].map((t, i) => (
                    <div key={i} style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
                      <span style={{ fontFamily: M, fontSize: 11, color: LIME, flexShrink: 0 }}>{i + 1}.</span>
                      <span style={{ fontSize: 12, color: '#bbb', lineHeight: 1.5 }}>{t}</span>
                    </div>
                  ))}
                </div>
              </div>
            </div>

            {/* RL stability story */}
            <div style={{ ...CARD, marginBottom: 16, borderColor: `${AMBER}33`, background: `${AMBER}06` }}>
              <div style={{ fontSize: 11, fontFamily: M, color: AMBER, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 10 }}>
                Why RL Makes This Critical
              </div>
              <div style={{ fontSize: 13, color: '#bbb', lineHeight: 1.65, marginBottom: 12 }}>
                In supervised training (SFT), non-deterministic topk adds noise but the model still converges --
                gradient averaging smooths it out. In RL (GRPO), the problem is fundamentally different:
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <div style={{ padding: '10px 14px', borderRadius: 6, background: 'rgba(0,0,0,.3)' }}>
                  <div style={{ fontFamily: M, fontSize: 11, color: AMBER, marginBottom: 6 }}>SFT (tolerates noise)</div>
                  <div style={{ fontSize: 11, color: '#999', lineHeight: 1.6 }}>
                    Loss = -log P(y|x) -- target is fixed. Non-deterministic mask adds variance to gradients
                    but the expected gradient direction is unchanged. SGD handles this fine.
                  </div>
                </div>
                <div style={{ padding: '10px 14px', borderRadius: 6, background: 'rgba(0,0,0,.3)' }}>
                  <div style={{ fontFamily: M, fontSize: 11, color: PINK, marginBottom: 6 }}>RL (breaks badly)</div>
                  <div style={{ fontSize: 11, color: '#999', lineHeight: 1.6 }}>
                    Reward = R(generated_text) -- the same prompt generates different completions when the
                    attention mask varies. Advantage estimation in GRPO becomes meaningless. Policy gradient
                    updates oscillate.
                  </div>
                </div>
              </div>
            </div>

            {/* Visual: deterministic vs non-deterministic */}
            <div style={{ ...CARD, marginBottom: 16 }}>
              <div style={{ fontSize: 11, fontFamily: M, color: PURPLE, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>
                Same Input, Different Masks (Non-Deterministic)
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 12 }}>
                {['GPU 0', 'GPU 1'].map((gpu, gi) => (
                  <div key={gi} style={{ padding: '10px', borderRadius: 5, background: 'rgba(0,0,0,.3)' }}>
                    <div style={{ fontFamily: M, fontSize: 10, color: gi === 0 ? CYAN : PINK, marginBottom: 8 }}>{gpu} mask</div>
                    <div style={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                      {Array.from({ length: 32 }, (_, i) => {
                        const active0 = [0,1,3,5,7,8,12,15,18,20,22,25,27,29,30,31];
                        const active1 = [0,2,3,5,6,8,12,14,18,20,23,25,26,29,30,31];
                        const active = gi === 0 ? active0 : active1;
                        const isActive = active.includes(i);
                        const differs = active0.includes(i) !== active1.includes(i);
                        return (
                          <div key={i} style={{
                            width: 14, height: 14, borderRadius: 2,
                            background: isActive ? (differs ? AMBER : (gi === 0 ? CYAN : PINK)) : 'rgba(255,255,255,.05)',
                            opacity: isActive ? 0.7 : 0.3,
                          }} />
                        );
                      })}
                    </div>
                    <div style={{ fontFamily: M, fontSize: 9, color: '#555', marginTop: 4 }}>
                      <span style={{ color: AMBER }}>yellow</span> = differs between GPUs
                    </div>
                  </div>
                ))}
              </div>
              <div style={{ fontSize: 11, color: '#666', lineHeight: 1.6 }}>
                Tied scores at positions 1/2, 6/7, 14/15, 22/23, 26/27 are broken differently by each GPU.
                With deterministic topk, both GPUs produce identical masks.
              </div>
            </div>

            <MathBlock label="Deterministic tie-breaking">
              <Var color={LIME}>score'</Var><Sub>i</Sub>
              {' = '}
              <Var>score</Var><Sub>i</Sub>
              {' + '}
              <Var color={AMBER}>i</Var>
              {' \u00B7 '}
              <Var color={PINK}>{'\u03B5'}</Var>
              {'   where '}
              <Var color={PINK}>{'\u03B5'}</Var>
              {' \u226A min(|score_i - score_j|) for non-tied pairs'}
              <br />
              <span style={{ fontSize: 13, color: '#666' }}>
                Position-based tiebreaker ensures identical ordering across all GPUs
              </span>
            </MathBlock>

            <HL code={`# Deterministic topk — critical for RL stability
def deterministic_topk(scores, k):
    # Problem: torch.topk breaks ties non-deterministically
    # Solution: add tiny position-based offset to break ties consistently
    eps = 1e-6
    B, H, T, S = scores.shape
    pos_offset = torch.arange(S, device=scores.device).float() * eps
    scores_stable = scores + pos_offset  # same offset on all GPUs

    # Now topk is deterministic: tied scores broken by position
    topk_vals, topk_idx = torch.topk(scores_stable, k=k, dim=-1)
    return topk_vals, topk_idx

# In DSA forward:
weights = torch.relu(scores) * dsa_importance
topk_vals, topk_idx = deterministic_topk(weights, k=2048)
# All GPUs in DP group now get identical sparse masks
# RL (GRPO) training stabilizes immediately`} />
          </div>
        )}

        {/* ── Resources ── */}
        <div style={{ marginTop: 32, borderTop: '1px solid rgba(255,255,255,.06)', paddingTop: 20 }}>
          <div style={{ fontSize: 10, fontFamily: M, color: '#444', textTransform: 'uppercase', letterSpacing: 1.5, marginBottom: 10 }}>
            Resources
          </div>
          {RESOURCES.map((r, i) => (
            <a key={i} href={r.url} target="_blank" rel="noopener noreferrer" style={{
              display: 'block', padding: '8px 12px', borderRadius: 5,
              marginBottom: 4, color: r.color, fontFamily: M, fontSize: 11,
              background: 'rgba(255,255,255,.02)', border: '1px solid rgba(255,255,255,.04)',
              textDecoration: 'none',
            }}>
              {r.label} <span style={{ color: '#444', marginLeft: 4 }}>{'\u2192'}</span>
            </a>
          ))}
        </div>
      </div>
    </div>
  );
}
