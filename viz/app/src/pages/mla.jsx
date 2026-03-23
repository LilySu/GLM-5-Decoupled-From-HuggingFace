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

// ─── Bar chart component ──────────────────────────────────────────────────────
function BarChart({ data, labels, series, maxVal }) {
  const mx = maxVal || Math.max(...data.flat());
  return (
    <div style={{ overflowX: 'auto' }}>
      <div style={{ display: 'flex', gap: 6, alignItems: 'flex-end', minWidth: labels.length * 100 }}>
        {labels.map((label, li) => (
          <div key={li} style={{ flex: 1, display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 4 }}>
            <div style={{ display: 'flex', gap: 2, alignItems: 'flex-end', height: 160 }}>
              {series.map((s, si) => {
                const val = data[si][li];
                const h = (val / mx) * 150;
                return (
                  <div key={si} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                    <span style={{ fontFamily: M, fontSize: 8, color: s.color, opacity: 0.8 }}>{val}</span>
                    <div style={{
                      width: 22, height: h, borderRadius: 3,
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

// ─── Flow node component ──────────────────────────────────────────────────────
function FlowNode({ label, dim, color, sub }) {
  return (
    <div style={{
      padding: '8px 14px', borderRadius: 6,
      background: `${color}10`, border: `1px solid ${color}44`,
      textAlign: 'center', minWidth: 100,
    }}>
      <div style={{ fontFamily: M, fontSize: 11, color, fontWeight: 600 }}>{label}</div>
      {dim && <div style={{ fontFamily: M, fontSize: 9, color: '#888', marginTop: 2 }}>[{dim}]</div>}
      {sub && <div style={{ fontSize: 9, color: '#555', marginTop: 2 }}>{sub}</div>}
    </div>
  );
}

function Arrow({ color }) {
  return <div style={{ fontFamily: M, fontSize: 16, color: color || '#444', alignSelf: 'center' }}>{'\u2192'}</div>;
}

// ─── Benchmark data (Table 1) ─────────────────────────────────────────────────
const BENCH_LABELS = ['Hellaswag', 'MMLU', 'C-Eval', 'RACE', 'BBH', 'GSM8K', 'HumanEval'];
const BENCH_DATA = [
  [77.3, 61.2, 60.0, 79.6, 53.3, 47.6, 38.5],  // GQA-8
  [77.3, 61.5, 59.7, 77.8, 48.9, 46.2, 33.5],  // MLA
  [77.8, 62.5, 62.1, 79.9, 51.8, 45.0, 36.7],  // MLA+Muon
];
const BENCH_SERIES = [
  { name: 'GQA-8', color: GRAY },
  { name: 'MLA', color: AMBER },
  { name: 'MLA+Muon Split', color: CYAN },
];

// ─── Resources ────────────────────────────────────────────────────────────────
const RESOURCES = [
  { label: 'FlashMLA -- DeepSeek MLA kernel for H100 (official)',     url: 'https://github.com/deepseek-ai/FlashMLA',       color: CYAN   },
  { label: 'DeepSeek-V2 -- MLA architecture paper',                  url: 'https://arxiv.org/abs/2405.04434',               color: AMBER  },
  { label: 'GLM-5 Technical Report (THUDM)',                         url: 'https://arxiv.org/abs/2501.12386',               color: PURPLE },
  { label: 'Muon Optimizer -- Jordan et al.',                        url: 'https://arxiv.org/abs/2502.16982',               color: LIME   },
];

// ─── Page component ───────────────────────────────────────────────────────────
export default function MLAPage() {
  const [activeTab, setActiveTab] = useState('flow');
  const [seqLen, setSeqLen]       = useState(4096);

  const mhaBytesPerToken = 2 * 64 * (128 + 128);  // 2 (K+V) * 64 heads * 128 dim * 2 bytes (BF16) ... simplified: 64 heads * 256 * 4 = 65536
  const mlaBytesPerToken = 576 * 2;                // compressed_kv=512 + k_pe=64 = 576 dims * 2 bytes
  const mhaTotal = (mhaBytesPerToken * seqLen / 1024 / 1024).toFixed(1);
  const mlaTotal = (mlaBytesPerToken * seqLen / 1024 / 1024).toFixed(1);
  const reduction = (mhaBytesPerToken / mlaBytesPerToken).toFixed(1);

  return (
    <div style={{ background: '#0d0f14', minHeight: '100vh', color: '#e0e0e0', fontFamily: S, padding: '32px 20px' }}>
      <div style={{ maxWidth: 1040, margin: '0 auto' }}>

        {/* ── Header ── */}
        <div style={{ marginBottom: 28 }}>
          <span style={{ fontFamily: M, fontSize: 11, color: CYAN, letterSpacing: 2, textTransform: 'uppercase' }}>
            GLM-5 · Attention
          </span>
          <h1 style={{ margin: '6px 0 6px', fontSize: 28, fontWeight: 700, color: '#f0f0f0', letterSpacing: -0.5 }}>
            Multi-Head Latent Attention + Muon Split
          </h1>
          <p style={{ margin: 0, color: 'rgba(255,255,255,.45)', fontSize: 14, maxWidth: 680, lineHeight: 1.6 }}>
            MLA compresses KV cache from 65,536 to 1,152 bytes per token (56.9x reduction) by projecting keys and values into a shared low-rank latent space. The Muon optimizer recovers quality lost vs GQA.
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
            MLA alone underperforms GQA-8 on reasoning benchmarks (BBH, GSM8K, HumanEval).
            The <strong style={{ color: LIME }}>Muon optimizer</strong> applied to attention projection weights
            (<span style={{ fontFamily: M, color: CYAN }}>q_a_proj</span>,{' '}
            <span style={{ fontFamily: M, color: CYAN }}>kv_a_proj</span>) recovers and often exceeds GQA quality,
            while keeping the <strong style={{ color: AMBER }}>56.9x KV cache reduction</strong>.
          </div>
        </div>

        {/* ── Tabs ── */}
        <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
          {[
            { id: 'flow',    label: 'Data Flow' },
            { id: 'muon',    label: 'Muon Split' },
            { id: 'absorb',  label: 'Weight Absorption' },
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
            TAB: Data Flow
            ════════════════════════════════════════════════════ */}
        {activeTab === 'flow' && (
          <div>
            {/* Q path */}
            <div style={{ ...CARD, marginBottom: 12 }}>
              <div style={{ fontSize: 11, fontFamily: M, color: PINK, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 14 }}>
                Q Path (Query Projection)
              </div>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', marginBottom: 12 }}>
                <FlowNode label="hidden" dim="6144" color={GRAY} />
                <Arrow />
                <FlowNode label="q_a_proj" dim="2048" color={AMBER} sub="down-project" />
                <Arrow />
                <FlowNode label="RMSNorm" dim="2048" color={PURPLE} />
                <Arrow />
                <FlowNode label="q_b_proj" dim="16384" color={AMBER} sub="up-project" />
                <Arrow />
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  <FlowNode label="q_nope" dim="128 x 192" color={CYAN} sub="no position" />
                  <FlowNode label="q_rope" dim="128 x 64" color={PINK} sub="rotary" />
                </div>
              </div>
              <MathBlock label="Q projection decomposition">
                <Var color={CYAN}>Q</Var> = <Var>W</Var><Sub>q_b</Sub> {'\u00B7'} RMSNorm(<Var>W</Var><Sub>q_a</Sub> {'\u00B7'} <Var color={AMBER}>h</Var>)
                {'  \u2208  '}
                <Var color={PINK}>R</Var><sup style={{ fontSize: '0.7em' }}>128 x 256</sup>
                {'  (split \u2192 nope[192] + rope[64])'}
              </MathBlock>
            </div>

            {/* KV path */}
            <div style={{ ...CARD, marginBottom: 12 }}>
              <div style={{ fontSize: 11, fontFamily: M, color: LIME, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 14 }}>
                KV Path (Key-Value Compression)
              </div>
              <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap', marginBottom: 12 }}>
                <FlowNode label="hidden" dim="6144" color={GRAY} />
                <Arrow />
                <FlowNode label="kv_a_proj" dim="576" color={LIME} sub="compress" />
                <Arrow />
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  <FlowNode label="compressed_kv" dim="512" color={CYAN} sub="cached!" />
                  <FlowNode label="k_pe" dim="64" color={PINK} sub="rope, cached!" />
                </div>
                <Arrow />
                <FlowNode label="kv_b_proj" dim="28672" color={LIME} sub="up-project" />
                <Arrow />
                <div style={{ display: 'flex', flexDirection: 'column', gap: 4 }}>
                  <FlowNode label="k_nope" dim="128 x 192" color={CYAN} />
                  <FlowNode label="v" dim="128 x 256" color={AMBER} />
                </div>
              </div>
              <MathBlock label="KV cache: only store the compressed latent">
                <Var color={LIME}>KV Cache</Var>{' = ['}
                <Var color={CYAN}>c</Var><Sub>kv</Sub>{' (512), '}
                <Var color={PINK}>k</Var><Sub>pe</Sub>{' (64)] = '}
                <Var color={AMBER}>576</Var>{' dims/token'}
                <br />
                <span style={{ fontSize: 13, color: '#666' }}>
                  vs MHA: 2 x 64 heads x 128 dim = 16,384 dims/token
                </span>
              </MathBlock>
            </div>

            {/* KV cache calculator */}
            <div style={{ ...CARD, marginBottom: 12 }}>
              <div style={{ fontSize: 11, fontFamily: M, color: AMBER, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 14 }}>
                KV Cache Size Calculator
              </div>
              <div style={{ marginBottom: 14 }}>
                <label style={{ fontFamily: M, fontSize: 11, color: '#888', marginRight: 12 }}>
                  Sequence length: <span style={{ color: CYAN }}>{seqLen.toLocaleString()}</span>
                </label>
                <input
                  type="range" min={512} max={131072} step={512} value={seqLen}
                  onChange={e => setSeqLen(Number(e.target.value))}
                  style={{ width: '100%', maxWidth: 400, accentColor: CYAN }}
                />
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
                <div style={{ padding: '12px 16px', borderRadius: 6, background: `${GRAY}14`, border: `1px solid ${GRAY}44` }}>
                  <div style={{ fontFamily: M, fontSize: 10, color: GRAY, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>MHA (Standard)</div>
                  <div style={{ fontFamily: M, fontSize: 22, fontWeight: 700, color: GRAY }}>{mhaTotal} MB</div>
                  <div style={{ fontFamily: M, fontSize: 9, color: '#555' }}>65,536 bytes/token</div>
                </div>
                <div style={{ padding: '12px 16px', borderRadius: 6, background: `${CYAN}14`, border: `1px solid ${CYAN}44` }}>
                  <div style={{ fontFamily: M, fontSize: 10, color: CYAN, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>MLA (GLM-5)</div>
                  <div style={{ fontFamily: M, fontSize: 22, fontWeight: 700, color: CYAN }}>{mlaTotal} MB</div>
                  <div style={{ fontFamily: M, fontSize: 9, color: '#555' }}>1,152 bytes/token</div>
                </div>
                <div style={{ padding: '12px 16px', borderRadius: 6, background: `${LIME}14`, border: `1px solid ${LIME}44` }}>
                  <div style={{ fontFamily: M, fontSize: 10, color: LIME, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>Reduction</div>
                  <div style={{ fontFamily: M, fontSize: 22, fontWeight: 700, color: LIME }}>{reduction}x</div>
                  <div style={{ fontFamily: M, fontSize: 9, color: '#555' }}>smaller KV cache</div>
                </div>
              </div>
              {/* Visual bar comparison */}
              <div style={{ marginTop: 16 }}>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
                  <span style={{ fontFamily: M, fontSize: 10, color: GRAY, minWidth: 40 }}>MHA</span>
                  <div style={{ flex: 1, height: 16, borderRadius: 3, background: GRAY, opacity: 0.5 }} />
                </div>
                <div style={{ display: 'flex', alignItems: 'center', gap: 10 }}>
                  <span style={{ fontFamily: M, fontSize: 10, color: CYAN, minWidth: 40 }}>MLA</span>
                  <div style={{ width: `${(1 / reduction) * 100}%`, height: 16, borderRadius: 3, background: CYAN, opacity: 0.7 }} />
                </div>
              </div>
            </div>

            <HL code={`# MLA: Q and KV projection paths
# Q path: down-project → RMSNorm → up-project → split
q_compressed = self.q_a_proj(hidden)         # [B, T, 2048]
q_compressed = self.q_a_layernorm(q_compressed)
q = self.q_b_proj(q_compressed)              # [B, T, 16384]
q_nope, q_pe = q.split([192, 64], dim=-1)   # per-head split

# KV path: down-project → split latent + rope key
kv_compressed = self.kv_a_proj(hidden)       # [B, T, 576]
compressed_kv, k_pe = kv_compressed.split([512, 64], dim=-1)
# CACHE only compressed_kv (512) + k_pe (64) = 576 dims

# Decode: up-project from cache on the fly
kv = self.kv_b_proj(compressed_kv)           # [B, T, 28672]
k_nope, v = kv.split([128*192, 128*256])`} />
          </div>
        )}

        {/* ══════════════════════════════════════════════════════
            TAB: Muon Split
            ════════════════════════════════════════════════════ */}
        {activeTab === 'muon' && (
          <div>
            <div style={{ fontSize: 13, color: '#888', lineHeight: 1.65, marginBottom: 20, maxWidth: 680 }}>
              Table 1 from the GLM-5 report: MLA alone loses quality vs GQA-8 on reasoning tasks.
              Applying the Muon optimizer specifically to the attention projection matrices (q_a_proj, q_b_proj,
              kv_a_proj, kv_b_proj) recovers and often exceeds GQA performance.
            </div>

            <div style={{ ...CARD, marginBottom: 16 }}>
              <div style={{ fontSize: 11, fontFamily: M, color: AMBER, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 16 }}>
                Benchmark Comparison (Table 1)
              </div>
              <BarChart
                data={BENCH_DATA}
                labels={BENCH_LABELS}
                series={BENCH_SERIES}
                maxVal={100}
              />
            </div>

            {/* Delta analysis */}
            <div style={{ ...CARD, marginBottom: 16 }}>
              <div style={{ fontSize: 11, fontFamily: M, color: LIME, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>
                MLA + Muon vs GQA-8 (Delta)
              </div>
              {BENCH_LABELS.map((label, i) => {
                const delta = (BENCH_DATA[2][i] - BENCH_DATA[0][i]).toFixed(1);
                const positive = parseFloat(delta) >= 0;
                return (
                  <div key={i} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 6 }}>
                    <span style={{ fontFamily: M, fontSize: 10, color: '#666', minWidth: 90 }}>{label}</span>
                    <div style={{ flex: 1, height: 12, borderRadius: 3, background: 'rgba(255,255,255,.04)', overflow: 'hidden', position: 'relative' }}>
                      <div style={{
                        position: 'absolute', left: '50%', top: 0, bottom: 0,
                        width: `${Math.abs(parseFloat(delta)) * 4}%`,
                        marginLeft: positive ? 0 : `-${Math.abs(parseFloat(delta)) * 4}%`,
                        background: positive ? LIME : PINK,
                        opacity: 0.6, borderRadius: 3,
                      }} />
                    </div>
                    <span style={{ fontFamily: M, fontSize: 11, color: positive ? LIME : PINK, minWidth: 40, textAlign: 'right' }}>
                      {positive ? '+' : ''}{delta}
                    </span>
                  </div>
                );
              })}
            </div>

            {/* Muon split explanation */}
            <div style={{ ...CARD, marginBottom: 16 }}>
              <div style={{ fontSize: 11, fontFamily: M, color: PURPLE, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>
                What is Muon Split?
              </div>
              <div style={{ fontSize: 13, color: '#bbb', lineHeight: 1.65, marginBottom: 12 }}>
                GLM-5 uses two different optimizers for different parameter groups:
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12 }}>
                <div style={{ padding: '10px 14px', borderRadius: 6, background: `${AMBER}10`, border: `1px solid ${AMBER}33` }}>
                  <div style={{ fontFamily: M, fontSize: 11, color: AMBER, marginBottom: 6 }}>AdamW</div>
                  <div style={{ fontSize: 12, color: '#999', lineHeight: 1.5 }}>
                    Embeddings, output head, RMSNorm, MoE router, MoE expert FFN weights
                  </div>
                </div>
                <div style={{ padding: '10px 14px', borderRadius: 6, background: `${LIME}10`, border: `1px solid ${LIME}33` }}>
                  <div style={{ fontFamily: M, fontSize: 11, color: LIME, marginBottom: 6 }}>Muon</div>
                  <div style={{ fontSize: 12, color: '#999', lineHeight: 1.5 }}>
                    q_a_proj, q_b_proj, kv_a_proj, kv_b_proj -- all MLA projection matrices
                  </div>
                </div>
              </div>
            </div>

            <MathBlock label="Muon update rule (orthogonalized momentum)">
              <Var color={LIME}>G</Var><Sub>orth</Sub>
              {' = NewtonSchulz('}
              <Var>G</Var>
              {')   \u2014   '}
              <Var color={AMBER}>W</Var>
              {' \u2192 '}
              <Var color={AMBER}>W</Var>
              {' - \u03B7 \u00B7 '}
              <Var color={LIME}>G</Var><Sub>orth</Sub>
              <br />
              <span style={{ fontSize: 13, color: '#666' }}>
                Newton-Schulz orthogonalization steers gradients onto the Stiefel manifold
              </span>
            </MathBlock>

            <HL code={`# Muon optimizer — applied only to MLA projection weights
# Newton-Schulz iteration for approximate orthogonalization
def newton_schulz_step(G, steps=5):
    a, b, c = (3.4445, -4.7750, 2.0315)
    X = G / G.norm()
    for _ in range(steps):
        A = X @ X.T
        X = a * X + b * (A @ X) + c * (A @ (A @ X))
    return X

# In optimizer.step():
for p in muon_params:     # q_a_proj, q_b_proj, kv_a_proj, kv_b_proj
    G = p.grad
    G_orth = newton_schulz_step(G)
    p.data -= lr * G_orth`} />
          </div>
        )}

        {/* ══════════════════════════════════════════════════════
            TAB: Weight Absorption
            ════════════════════════════════════════════════════ */}
        {activeTab === 'absorb' && (
          <div>
            <div style={{ fontSize: 13, color: '#888', lineHeight: 1.65, marginBottom: 20, maxWidth: 680 }}>
              Weight absorption merges the up-projection matrices into the attention computation,
              eliminating a matmul at decode time. FlashMLA natively supports the asymmetric
              d_v=512 that results from this absorption.
            </div>

            {/* Normal vs absorbed comparison */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16 }}>
              <div style={{ ...CARD, borderColor: `${GRAY}44` }}>
                <div style={{ fontSize: 11, fontFamily: M, color: GRAY, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>
                  Normal MLA (without absorption)
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {[
                    { step: '1', text: 'Load compressed_kv from cache', dim: '[T, 512]' },
                    { step: '2', text: 'kv_b_proj up-project', dim: '[T, 512] \u2192 [T, 28672]' },
                    { step: '3', text: 'Split k_nope + v', dim: '[T, 128x192] + [T, 128x256]' },
                    { step: '4', text: 'Compute Q \u00B7 K^T', dim: '' },
                    { step: '5', text: 'Apply softmax + V matmul', dim: '' },
                  ].map(({ step, text, dim }) => (
                    <div key={step} style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
                      <span style={{ fontFamily: M, fontSize: 10, color: GRAY, flexShrink: 0, width: 16 }}>{step}.</span>
                      <div>
                        <span style={{ fontSize: 11, color: '#bbb' }}>{text}</span>
                        {dim && <span style={{ fontFamily: M, fontSize: 9, color: '#555', marginLeft: 6 }}>{dim}</span>}
                      </div>
                    </div>
                  ))}
                </div>
                <div style={{ marginTop: 10, padding: '6px 10px', borderRadius: 4, background: `${PINK}10`, border: `1px solid ${PINK}33` }}>
                  <span style={{ fontFamily: M, fontSize: 10, color: PINK }}>Extra matmul: kv_b_proj on every decode step</span>
                </div>
              </div>

              <div style={{ ...CARD, borderColor: `${CYAN}44` }}>
                <div style={{ fontSize: 11, fontFamily: M, color: CYAN, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>
                  Absorbed MLA (FlashMLA)
                </div>
                <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
                  {[
                    { step: '1', text: 'Absorb W_q_b into Q (offline)', dim: 'Q\' = Q \u00B7 W_q_b^T' },
                    { step: '2', text: 'Absorb W_kv_b into K,V (offline)', dim: 'precomputed' },
                    { step: '3', text: 'Compute Q\' \u00B7 compressed_kv^T', dim: 'direct on cache' },
                    { step: '4', text: 'Output has d_v = 512', dim: 'not 128' },
                    { step: '5', text: 'No kv_b_proj at decode!', dim: '' },
                  ].map(({ step, text, dim }) => (
                    <div key={step} style={{ display: 'flex', gap: 8, alignItems: 'flex-start' }}>
                      <span style={{ fontFamily: M, fontSize: 10, color: CYAN, flexShrink: 0, width: 16 }}>{step}.</span>
                      <div>
                        <span style={{ fontSize: 11, color: '#bbb' }}>{text}</span>
                        {dim && <span style={{ fontFamily: M, fontSize: 9, color: '#555', marginLeft: 6 }}>{dim}</span>}
                      </div>
                    </div>
                  ))}
                </div>
                <div style={{ marginTop: 10, padding: '6px 10px', borderRadius: 4, background: `${LIME}10`, border: `1px solid ${LIME}33` }}>
                  <span style={{ fontFamily: M, fontSize: 10, color: LIME }}>No decode-time up-projection needed</span>
                </div>
              </div>
            </div>

            {/* FlashMLA d_v=512 */}
            <div style={{ ...CARD, marginBottom: 16, borderColor: `${AMBER}33`, background: `${AMBER}06` }}>
              <div style={{ fontSize: 11, fontFamily: M, color: AMBER, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 10 }}>
                FlashMLA d_v = 512 Constraint
              </div>
              <div style={{ fontSize: 13, color: '#bbb', lineHeight: 1.65, marginBottom: 10 }}>
                Standard FlashAttention assumes d_k == d_v (typically 128). After weight absorption,
                MLA has <strong style={{ color: CYAN }}>d_k = 192</strong> (nope) but{' '}
                <strong style={{ color: AMBER }}>d_v = 512</strong> (absorbed V projection).
                FlashMLA is a modified FlashAttention kernel that natively handles this asymmetry.
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10 }}>
                {[
                  { label: 'd_k (nope)', val: '192', color: CYAN },
                  { label: 'd_k (rope)', val: '64', color: PINK },
                  { label: 'd_v (absorbed)', val: '512', color: AMBER },
                ].map(({ label, val, color }) => (
                  <div key={label} style={{ padding: '8px 12px', borderRadius: 5, background: `${color}10`, border: `1px solid ${color}33`, textAlign: 'center' }}>
                    <div style={{ fontFamily: M, fontSize: 9, color: '#666', marginBottom: 2 }}>{label}</div>
                    <div style={{ fontFamily: M, fontSize: 18, fontWeight: 700, color }}>{val}</div>
                  </div>
                ))}
              </div>
            </div>

            <MathBlock label="Weight absorption eliminates decode-time up-projection">
              <Var color={CYAN}>Q'</Var>{' = '}
              <Var>W</Var><Sub>q_b</Sub><sup style={{ fontSize: '0.65em' }}>T</sup>{' \u00B7 '}
              <Var color={CYAN}>Q</Var>
              {'   \u21D2   '}
              <Var color={CYAN}>Q'</Var>{' \u00B7 '}
              <Var color={LIME}>c</Var><Sub>kv</Sub><sup style={{ fontSize: '0.65em' }}>T</sup>
              {'  =  '}
              <Var color={CYAN}>Q</Var>{' \u00B7 '}
              <Var>W</Var><Sub>q_b</Sub>{' \u00B7 '}
              <Var>W</Var><Sub>kv_b</Sub><sup style={{ fontSize: '0.65em' }}>T</sup>{' \u00B7 '}
              <Var color={LIME}>c</Var><Sub>kv</Sub><sup style={{ fontSize: '0.65em' }}>T</sup>
              <br />
              <span style={{ fontSize: 13, color: '#666' }}>
                Absorbed: W_UV merged into V, so FlashMLA emits d_v=512 output directly
              </span>
            </MathBlock>

            <HL code={`# Weight absorption — done once at model load, not per decode step
# Before (every decode step):
#   kv = self.kv_b_proj(compressed_kv)   # [B, T, 28672] -- expensive!
#   k_nope, v = kv.split(...)

# After absorption (at load time):
W_UK = kv_b_proj.weight[:n_heads*d_nope, :]  # [24576, 512]
W_UV = kv_b_proj.weight[n_heads*d_nope:, :]  # [32768, 512]

# Absorbed Q: merge W_UK into query projection
# Q_absorbed = Q @ W_UK.T  — precomputed into q_b_proj
# Now at decode: score = Q_absorbed @ compressed_kv.T  (no up-proj!)

# FlashMLA handles d_v=512 natively:
out = flash_mla_with_kvcache(
    q=q_absorbed,          # [B, H, 1, 192]
    kv_cache=kv_cache,     # paged [pages, 64, 576]
    block_table=block_table,
    cache_seqlens=seqlens,
)  # out: [B, H, 1, 512]  — absorbed d_v`} />
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
