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

// ─── Routing step visualisation ───────────────────────────────────────────────
function RoutingStep({ num, title, desc, color, detail }) {
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

// ─── dtype configs ────────────────────────────────────────────────────────────
const DTYPES = [
  { label: 'BF16', bytes: 2, color: AMBER },
  { label: 'FP8',  bytes: 1, color: CYAN },
  { label: 'INT4', bytes: 0.5, color: LIME },
];

// ─── Kernel stack performance ─────────────────────────────────────────────────
const KERNEL_PERF = [
  { label: 'PyTorch per-expert loop', tflops: 100, color: GRAY, detail: '256 separate F.linear calls, sequential dispatch' },
  { label: 'Triton grouped GEMM', tflops: 650, color: AMBER, detail: 'Unsloth Triton kernel, groups tokens by expert' },
  { label: 'DeepGEMM FP8', tflops: 1550, color: CYAN, detail: 'H100 FP8 tensor cores, single kernel launch for all 256 experts' },
];

// ─── Resources ────────────────────────────────────────────────────────────────
const RESOURCES = [
  { label: 'DeepGEMM -- FP8 grouped GEMM for MoE (DeepSeek)',      url: 'https://github.com/deepseek-ai/DeepGEMM',    color: CYAN   },
  { label: 'GLM-5 Technical Report -- MoE architecture (THUDM)',   url: 'https://arxiv.org/abs/2501.12386',            color: PURPLE },
  { label: 'DeepSeek-V3 -- MoE training details',                  url: 'https://arxiv.org/abs/2412.19437',            color: AMBER  },
  { label: 'Switch Transformers -- MoE foundations',                url: 'https://arxiv.org/abs/2101.03961',            color: LIME   },
];

// ─── Page component ───────────────────────────────────────────────────────────
export default function MoEPage() {
  const [activeTab, setActiveTab] = useState('routing');
  const [layers, setLayers]       = useState(75);
  const [dtypeIdx, setDtypeIdx]   = useState(0);

  const dtype = DTYPES[dtypeIdx];
  // Per-expert params: gate_proj [2048, 6144] + up_proj [2048, 6144] + down_proj [6144, 2048]
  const gateUpParams = 2048 * 6144 * 2;   // gate + up
  const downParams   = 6144 * 2048;
  const perExpertParams = gateUpParams + downParams;
  const perExpertBytes  = perExpertParams * dtype.bytes;
  const totalExperts    = 256;
  const moeLayerBytes   = perExpertBytes * totalExperts;
  const totalBytes      = moeLayerBytes * layers;
  const totalGB         = (totalBytes / 1e9).toFixed(1);
  const perExpertMB     = (perExpertBytes / 1e6).toFixed(1);
  const perLayerGB      = (moeLayerBytes / 1e9).toFixed(2);

  return (
    <div style={{ background: '#0d0f14', minHeight: '100vh', color: '#e0e0e0', fontFamily: S, padding: '32px 20px' }}>
      <div style={{ maxWidth: 1040, margin: '0 auto' }}>

        {/* ── Header ── */}
        <div style={{ marginBottom: 28 }}>
          <span style={{ fontFamily: M, fontSize: 11, color: LIME, letterSpacing: 2, textTransform: 'uppercase' }}>
            GLM-5 · Mixture of Experts
          </span>
          <h1 style={{ margin: '6px 0 6px', fontSize: 28, fontWeight: 700, color: '#f0f0f0', letterSpacing: -0.5 }}>
            MoE System: 256 Experts, Top-8 Routing
          </h1>
          <p style={{ margin: 0, color: 'rgba(255,255,255,.45)', fontSize: 14, maxWidth: 680, lineHeight: 1.6 }}>
            GLM-5 uses 256 FFN experts per MoE layer with flat sigmoid routing (n_group=1, no hierarchical selection).
            Each token activates 8 experts. The routing logic is 5 lines of PyTorch.
          </p>
        </div>

        {/* ── Key insight callout ── */}
        <div style={{
          marginBottom: 24, padding: '14px 18px', borderRadius: 8,
          background: `${LIME}0c`, border: `1px solid ${LIME}33`,
        }}>
          <div style={{ fontSize: 11, fontFamily: M, color: LIME, marginBottom: 5, letterSpacing: 1, textTransform: 'uppercase' }}>
            Key Insight
          </div>
          <div style={{ fontSize: 13, color: '#d0d0d0', lineHeight: 1.65 }}>
            GLM-5's MoE router is intentionally simple: <strong style={{ color: LIME }}>sigmoid scoring + flat top-8</strong>.
            With <span style={{ fontFamily: M, color: CYAN }}>n_group=1</span>, there is no hierarchical routing,
            no group-level selection, no auxiliary load-balancing loss.
            The bottleneck is not routing but the <strong style={{ color: AMBER }}>256 expert GEMMs</strong> --
            solved by DeepGEMM's FP8 grouped GEMM at <strong style={{ color: CYAN }}>1550 TFLOPS</strong>.
          </div>
        </div>

        {/* ── Tabs ── */}
        <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
          {[
            { id: 'routing', label: 'Routing Flow' },
            { id: 'memory',  label: 'Memory Calculator' },
            { id: 'kernel',  label: 'Kernel Stack' },
          ].map(({ id, label }) => (
            <button key={id} onClick={() => setActiveTab(id)} style={{
              padding: '7px 18px', borderRadius: 6,
              border: `1px solid ${activeTab === id ? LIME : 'rgba(255,255,255,.1)'}`,
              background: activeTab === id ? `${LIME}12` : 'rgba(255,255,255,.03)',
              color: activeTab === id ? LIME : '#777',
              fontFamily: M, fontSize: 11, cursor: 'pointer',
            }}>
              {label}
            </button>
          ))}
        </div>

        {/* ══════════════════════════════════════════════════════
            TAB: Routing Flow
            ════════════════════════════════════════════════════ */}
        {activeTab === 'routing' && (
          <div>
            {/* Step-through */}
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginBottom: 16 }}>
              <RoutingStep
                num={1} color={LIME}
                title="Linear Gate Projection"
                desc="Project hidden state through a linear layer to get scores for all 256 experts."
                detail="router_logits = self.gate(hidden)  --  [B*T, 256]"
              />
              <RoutingStep
                num={2} color={AMBER}
                title="Sigmoid Scoring (NOT softmax)"
                desc="Apply sigmoid independently to each expert score. Unlike softmax, experts do not compete -- each gets an independent probability."
                detail="scores = torch.sigmoid(router_logits)  --  [B*T, 256]"
              />
              <RoutingStep
                num={3} color={CYAN}
                title="Flat Top-8 Selection"
                desc="Select the 8 highest-scoring experts. n_group=1 means no hierarchical routing -- straight top-8 from all 256."
                detail="topk_w, topk_idx = torch.topk(scores, k=8, dim=-1)"
              />
              <RoutingStep
                num={4} color={PINK}
                title="Renormalize Weights"
                desc="Normalize the selected expert weights to sum to 1. This ensures the output scale is consistent."
                detail="topk_w = topk_w / topk_w.sum(-1, keepdim=True)"
              />
              <RoutingStep
                num={5} color={PURPLE}
                title="Dispatch + Combine"
                desc="Route each token to its 8 experts, compute FFN outputs, combine with renormalized weights."
                detail="out = sum(topk_w[i] * expert_i(x) for i in range(8))"
              />
            </div>

            {/* "5 lines" callout */}
            <div style={{
              ...CARD, marginBottom: 16,
              borderColor: `${LIME}44`, background: `${LIME}06`,
            }}>
              <div style={{ fontSize: 11, fontFamily: M, color: LIME, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 10 }}>
                GLM-5 routing is 5 lines of PyTorch
              </div>
              <div style={{ fontSize: 13, color: '#bbb', lineHeight: 1.65, marginBottom: 4 }}>
                No auxiliary load-balancing loss. No grouped routing. No expert capacity factor.
                The simplicity is intentional -- with 256 experts and sigmoid scoring, load
                naturally balances because the model can independently "turn on" multiple experts
                without the zero-sum competition of softmax.
              </div>
            </div>

            <MathBlock label="GLM-5 MoE routing (sigmoid, flat top-8)">
              <Var color={LIME}>s</Var>
              {' = \u03C3('}
              <Var>W</Var><Sub>gate</Sub>
              {' \u00B7 '}
              <Var color={AMBER}>h</Var>
              {')   \u2208 [0,1]'}
              <sup style={{ fontSize: '0.7em' }}>256</sup>
              <br />
              <Var color={CYAN}>w</Var><Sub>1..8</Sub>
              {', '}
              <Var color={CYAN}>e</Var><Sub>1..8</Sub>
              {' = top8('}
              <Var color={LIME}>s</Var>
              {')   \u2014   '}
              <Var color={PINK}>y</Var>
              {' = \u2211 '}
              <Var color={CYAN}>w\u0302</Var><Sub>i</Sub>
              {' \u00B7 FFN'}
              <Sub>e_i</Sub>
              {'('}
              <Var color={AMBER}>h</Var>
              {')'}
            </MathBlock>

            <HL code={`# GLM-5 MoE Router — sigmoid, flat top-8, n_group=1
# This is the ENTIRE routing logic. No auxiliary losses.
def route(self, hidden_states):
    router_logits = self.gate(hidden_states)          # [B*T, 256]
    scores = torch.sigmoid(router_logits)             # independent probabilities
    topk_w, topk_idx = torch.topk(scores, k=8, dim=-1)
    topk_w = topk_w / topk_w.sum(-1, keepdim=True)   # renormalise selected
    return topk_w, topk_idx

# Compare with DeepSeek-V3's hierarchical routing:
# 1. Group experts into 8 groups of 32
# 2. Score groups, select top-k groups
# 3. Within selected groups, select top-k experts
# GLM-5 skips all of this: n_group=1 → one flat pool of 256`} />

            {/* Visual: expert grid */}
            <div style={{ ...CARD, marginTop: 16 }}>
              <div style={{ fontSize: 11, fontFamily: M, color: PURPLE, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 10 }}>
                256 Experts — 8 Active (example routing)
              </div>
              <div style={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                {Array.from({ length: 256 }, (_, i) => {
                  const active = [12, 47, 89, 103, 145, 178, 210, 241];
                  const isActive = active.includes(i);
                  return (
                    <div key={i} style={{
                      width: 14, height: 14, borderRadius: 2,
                      background: isActive ? LIME : 'rgba(255,255,255,.05)',
                      opacity: isActive ? 0.85 : 0.3,
                      transition: 'all .15s',
                    }} title={`Expert ${i}${isActive ? ' (active)' : ''}`} />
                  );
                })}
              </div>
              <div style={{ fontFamily: M, fontSize: 9, color: '#555', marginTop: 6 }}>
                <span style={{ color: LIME }}>green</span> = active experts for this token (8 of 256 = 3.1% sparsity)
              </div>
            </div>
          </div>
        )}

        {/* ══════════════════════════════════════════════════════
            TAB: Memory Calculator
            ════════════════════════════════════════════════════ */}
        {activeTab === 'memory' && (
          <div>
            <div style={{ fontSize: 13, color: '#888', lineHeight: 1.65, marginBottom: 20, maxWidth: 680 }}>
              Each expert has three weight matrices (gate_proj, up_proj, down_proj) in a SwiGLU FFN.
              With 256 experts per layer and up to 75 MoE layers, expert weights dominate total model size.
            </div>

            {/* Controls */}
            <div style={{ ...CARD, marginBottom: 16 }}>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 16 }}>
                <div>
                  <label style={{ fontFamily: M, fontSize: 11, color: '#888', display: 'block', marginBottom: 6 }}>
                    MoE Layers: <span style={{ color: CYAN }}>{layers}</span>
                  </label>
                  <input
                    type="range" min={1} max={75} step={1} value={layers}
                    onChange={e => setLayers(Number(e.target.value))}
                    style={{ width: '100%', accentColor: CYAN }}
                  />
                </div>
                <div>
                  <label style={{ fontFamily: M, fontSize: 11, color: '#888', display: 'block', marginBottom: 6 }}>
                    Data Type: <span style={{ color: dtype.color }}>{dtype.label}</span>
                  </label>
                  <div style={{ display: 'flex', gap: 6 }}>
                    {DTYPES.map((d, i) => (
                      <button key={i} onClick={() => setDtypeIdx(i)} style={{
                        padding: '6px 14px', borderRadius: 5,
                        border: `1px solid ${dtypeIdx === i ? d.color : 'rgba(255,255,255,.1)'}`,
                        background: dtypeIdx === i ? `${d.color}14` : 'rgba(255,255,255,.03)',
                        color: dtypeIdx === i ? d.color : '#666',
                        fontFamily: M, fontSize: 11, cursor: 'pointer',
                      }}>
                        {d.label}
                      </button>
                    ))}
                  </div>
                </div>
              </div>
            </div>

            {/* Results */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr 1fr', gap: 10, marginBottom: 16 }}>
              {[
                { label: 'Per Expert', val: `${perExpertMB} MB`, sub: `${(perExpertParams / 1e6).toFixed(1)}M params`, color: AMBER },
                { label: 'Per Layer', val: `${perLayerGB} GB`, sub: `256 experts x ${perExpertMB} MB`, color: CYAN },
                { label: `Total (${layers} layers)`, val: `${totalGB} GB`, sub: `${dtype.label} weights only`, color: LIME },
                { label: 'Active / Token', val: `${(perExpertBytes * 8 / 1e6).toFixed(0)} MB`, sub: '8 experts active', color: PINK },
              ].map(({ label, val, sub, color }) => (
                <div key={label} style={{ padding: '12px 14px', borderRadius: 6, background: `${color}10`, border: `1px solid ${color}33` }}>
                  <div style={{ fontFamily: M, fontSize: 9, color: '#666', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>{label}</div>
                  <div style={{ fontFamily: M, fontSize: 18, fontWeight: 700, color }}>{val}</div>
                  <div style={{ fontFamily: M, fontSize: 9, color: '#555', marginTop: 2 }}>{sub}</div>
                </div>
              ))}
            </div>

            {/* Weight breakdown */}
            <div style={{ ...CARD, marginBottom: 16 }}>
              <div style={{ fontSize: 11, fontFamily: M, color: AMBER, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>
                Per-Expert Weight Breakdown
              </div>
              {[
                { name: 'gate_proj', shape: '[2048, 6144]', params: 2048 * 6144, color: LIME },
                { name: 'up_proj', shape: '[2048, 6144]', params: 2048 * 6144, color: CYAN },
                { name: 'down_proj', shape: '[6144, 2048]', params: 6144 * 2048, color: AMBER },
              ].map(({ name, shape, params, color }) => (
                <div key={name} style={{ display: 'flex', alignItems: 'center', gap: 10, marginBottom: 8 }}>
                  <span style={{ fontFamily: M, fontSize: 11, color, minWidth: 90 }}>{name}</span>
                  <span style={{ fontFamily: M, fontSize: 10, color: '#555', minWidth: 110 }}>{shape}</span>
                  <div style={{ flex: 1, height: 14, borderRadius: 3, background: 'rgba(255,255,255,.04)', overflow: 'hidden' }}>
                    <div style={{ height: '100%', width: `${(params / perExpertParams) * 100}%`, background: color, opacity: 0.6, borderRadius: 3 }} />
                  </div>
                  <span style={{ fontFamily: M, fontSize: 10, color: '#666', minWidth: 60, textAlign: 'right' }}>{(params * dtype.bytes / 1e6).toFixed(1)} MB</span>
                </div>
              ))}
            </div>

            <MathBlock label="Expert parameter count">
              <Var color={AMBER}>params</Var><Sub>expert</Sub>
              {' = '}
              <Var>d</Var><Sub>model</Sub>
              {' \u00D7 '}
              <Var>d</Var><Sub>ff</Sub>
              {' \u00D7 '}
              <Var color={CYAN}>3</Var>
              {' = 2048 \u00D7 6144 \u00D7 3 = '}
              <Var color={LIME}>37.7M</Var>
              <br />
              <Var color={PINK}>total</Var>
              {' = 37.7M \u00D7 256 experts \u00D7 '}
              <Var>{layers}</Var>
              {' layers = '}
              <Var color={PINK}>{(perExpertParams * totalExperts * layers / 1e9).toFixed(1)}B</Var>
              {' params'}
            </MathBlock>

            <HL code={`# Per-expert SwiGLU FFN structure
# d_model = 2048 (intermediate per-expert), d_ff = 6144
class MoEExpert(nn.Module):
    def __init__(self, d_model=2048, d_ff=6144):
        self.gate_proj = nn.Linear(d_model, d_ff, bias=False)  # [2048, 6144]
        self.up_proj   = nn.Linear(d_model, d_ff, bias=False)  # [2048, 6144]
        self.down_proj = nn.Linear(d_ff, d_model, bias=False)  # [6144, 2048]

    def forward(self, x):
        return self.down_proj(F.silu(self.gate_proj(x)) * self.up_proj(x))

# 256 experts x 3 matrices = 768 weight tensors per MoE layer
# 75 MoE layers x 768 = 57,600 weight tensors total`} />
          </div>
        )}

        {/* ══════════════════════════════════════════════════════
            TAB: Kernel Stack
            ════════════════════════════════════════════════════ */}
        {activeTab === 'kernel' && (
          <div>
            <div style={{ fontSize: 13, color: '#888', lineHeight: 1.65, marginBottom: 20, maxWidth: 680 }}>
              The MoE GEMM kernel is the single most performance-critical component.
              Each forward pass dispatches tokens to 8 of 256 experts -- without fused grouped GEMM,
              this means 256 separate kernel launches per layer.
            </div>

            {/* Performance bars */}
            <div style={{ ...CARD, marginBottom: 16 }}>
              <div style={{ fontSize: 11, fontFamily: M, color: CYAN, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 16 }}>
                MoE GEMM Performance (H100)
              </div>
              {KERNEL_PERF.map((k) => (
                <div key={k.label} style={{ marginBottom: 16 }}>
                  <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                    <div>
                      <span style={{ fontFamily: M, fontSize: 12, color: k.color, fontWeight: 600 }}>{k.label}</span>
                    </div>
                    <div style={{
                      fontFamily: M, fontSize: 20, fontWeight: 700, color: k.color,
                      padding: '2px 10px', borderRadius: 5,
                      background: `${k.color}12`, border: `1px solid ${k.color}33`,
                    }}>
                      {k.tflops} TFLOPS
                    </div>
                  </div>
                  <div style={{ height: 10, borderRadius: 4, background: 'rgba(255,255,255,.05)', overflow: 'hidden', marginBottom: 4 }}>
                    <div style={{
                      height: '100%', width: `${(k.tflops / 1550) * 100}%`,
                      background: k.color, opacity: 0.7, borderRadius: 4,
                      transition: 'width 0.4s ease',
                    }} />
                  </div>
                  <div style={{ fontFamily: M, fontSize: 10, color: '#555' }}>{k.detail}</div>
                </div>
              ))}
            </div>

            {/* Three-tier comparison */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 10, marginBottom: 16 }}>
              <div style={{ ...CARD, borderColor: `${GRAY}44` }}>
                <div style={{ fontFamily: M, fontSize: 11, color: GRAY, marginBottom: 8 }}>Tier 1: PyTorch Loop</div>
                <div style={{ fontSize: 12, color: '#999', lineHeight: 1.6, marginBottom: 8 }}>
                  For each expert, gather tokens routed to it, call F.linear, scatter results back.
                  256 kernel launches per MoE layer.
                </div>
                <div style={{ fontFamily: M, fontSize: 9, color: '#555' }}>
                  ~100 TFLOPS -- kernel launch overhead dominates
                </div>
              </div>
              <div style={{ ...CARD, borderColor: `${AMBER}44` }}>
                <div style={{ fontFamily: M, fontSize: 11, color: AMBER, marginBottom: 8 }}>Tier 2: Triton Grouped</div>
                <div style={{ fontSize: 12, color: '#999', lineHeight: 1.6, marginBottom: 8 }}>
                  Unsloth Triton kernel groups all tokens by expert and dispatches a single grouped GEMM.
                  BF16 compute, 1 kernel launch.
                </div>
                <div style={{ fontFamily: M, fontSize: 9, color: '#555' }}>
                  ~650 TFLOPS -- BF16 tensor cores, good occupancy
                </div>
              </div>
              <div style={{ ...CARD, borderColor: `${CYAN}44` }}>
                <div style={{ fontFamily: M, fontSize: 11, color: CYAN, marginBottom: 8 }}>Tier 3: DeepGEMM FP8</div>
                <div style={{ fontSize: 12, color: '#999', lineHeight: 1.6, marginBottom: 8 }}>
                  DeepGEMM's m_grouped_fp8_gemm uses H100 FP8 tensor cores. Per-token activation
                  quantization + pre-quantized expert weights.
                </div>
                <div style={{ fontFamily: M, fontSize: 9, color: '#555' }}>
                  ~1550 TFLOPS -- near peak H100 FP8 throughput
                </div>
              </div>
            </div>

            <MathBlock label="DeepGEMM grouped FP8 throughput">
              <Var color={CYAN}>TFLOPS</Var>
              {' = '}
              <Var>E</Var><Sub>active</Sub>
              {' \u00D7 2 \u00D7 '}
              <Var>d</Var><Sub>model</Sub>
              {' \u00D7 '}
              <Var>d</Var><Sub>ff</Sub>
              {' \u00D7 '}
              <Var>tokens</Var>
              {' / '}
              <Var color={PINK}>t</Var>
              <br />
              <span style={{ fontSize: 13, color: '#666' }}>
                8 active experts x 2 x 2048 x 6144 = 201M FLOPs per token per MoE layer
              </span>
            </MathBlock>

            <HL code={`# Tier 1: PyTorch per-expert loop (baseline, ~100 TFLOPS)
for expert_id in range(256):
    mask = (topk_ids == expert_id).any(-1)
    if mask.any():
        out[mask] += w[mask, expert_id] * expert[expert_id](x[mask])

# Tier 2: Triton grouped GEMM (~650 TFLOPS)
# Groups tokens by expert, single kernel launch
triton_grouped_gemm(x, expert_weights, topk_ids, topk_weights, out)

# Tier 3: DeepGEMM FP8 (~1550 TFLOPS)
x_fp8, x_scale = per_token_quant_fp8(hidden_states)
out = deep_gemm.m_grouped_fp8_gemm(
    x_fp8, expert_weights_fp8, topk_ids,
    scales=(x_scale, w_scale),
)  # Single kernel, FP8 tensor cores, all 256 experts`} />
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
