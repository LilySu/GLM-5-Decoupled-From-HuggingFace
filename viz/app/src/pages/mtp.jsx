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

// ─── Token block component ────────────────────────────────────────────────────
function TokenBlock({ text, color, faded, small }) {
  return (
    <span style={{
      display: 'inline-block',
      padding: small ? '3px 6px' : '5px 10px',
      borderRadius: 4,
      background: faded ? 'rgba(255,255,255,.04)' : `${color}18`,
      border: `1px solid ${faded ? 'rgba(255,255,255,.08)' : color + '55'}`,
      fontFamily: M, fontSize: small ? 9 : 11,
      color: faded ? '#555' : color,
      fontWeight: 500,
    }}>
      {text}
    </span>
  );
}

// ─── Resources ────────────────────────────────────────────────────────────────
const RESOURCES = [
  { label: 'GLM-5 Technical Report -- MTP details (THUDM)',          url: 'https://arxiv.org/abs/2501.12386',                              color: CYAN   },
  { label: 'DeepSeek-V3 -- Multi-Token Prediction architecture',     url: 'https://arxiv.org/abs/2412.19437',                              color: AMBER  },
  { label: 'Better & Faster LLMs via Multi-Token Prediction (Meta)', url: 'https://arxiv.org/abs/2404.19737',                              color: PURPLE },
  { label: 'vLLM Speculative Decoding docs',                         url: 'https://docs.vllm.ai/en/latest/features/spec_decode.html',      color: LIME   },
];

// ─── Page component ───────────────────────────────────────────────────────────
export default function MTPPage() {
  const [activeTab, setActiveTab] = useState('how');
  const [acceptLen, setAcceptLen] = useState(2.76);
  const [pipeStep, setPipeStep]   = useState(0);

  const baseLatency = 30;  // ms per token (hypothetical)
  const standardTps = (1000 / baseLatency).toFixed(1);
  const mtpTps = ((1000 / baseLatency) * acceptLen).toFixed(1);
  const speedup = acceptLen.toFixed(2);

  return (
    <div style={{ background: '#0d0f14', minHeight: '100vh', color: '#e0e0e0', fontFamily: S, padding: '32px 20px' }}>
      <div style={{ maxWidth: 1040, margin: '0 auto' }}>

        {/* ── Header ── */}
        <div style={{ marginBottom: 28 }}>
          <span style={{ fontFamily: M, fontSize: 11, color: PURPLE, letterSpacing: 2, textTransform: 'uppercase' }}>
            GLM-5 · Speculative Decoding
          </span>
          <h1 style={{ margin: '6px 0 6px', fontSize: 28, fontWeight: 700, color: '#f0f0f0', letterSpacing: -0.5 }}>
            Multi-Token Prediction + Speculation
          </h1>
          <p style={{ margin: 0, color: 'rgba(255,255,255,.45)', fontSize: 14, maxWidth: 680, lineHeight: 1.6 }}>
            MTP trains the model to predict multiple future tokens simultaneously. At inference,
            the MTP head acts as a built-in draft model for speculative decoding, achieving
            2.76 average accepted tokens per step (vs DeepSeek's 2.55).
          </p>
        </div>

        {/* ── Key insight callout ── */}
        <div style={{
          marginBottom: 24, padding: '14px 18px', borderRadius: 8,
          background: `${PURPLE}0c`, border: `1px solid ${PURPLE}33`,
        }}>
          <div style={{ fontSize: 11, fontFamily: M, color: PURPLE, marginBottom: 5, letterSpacing: 1, textTransform: 'uppercase' }}>
            Key Insight
          </div>
          <div style={{ fontSize: 13, color: '#d0d0d0', lineHeight: 1.65 }}>
            MTP is <strong style={{ color: PURPLE }}>free at inference</strong> -- the draft head shares the
            main model's hidden states. Unlike separate draft models, there is no additional memory cost
            and the drafting overhead is a single linear projection.
            GLM-5 achieves <strong style={{ color: CYAN }}>2.76x</strong> average speedup vs
            DeepSeek-V3's <strong style={{ color: AMBER }}>2.55x</strong> with the same MTP approach
            but 3 shared draft layers instead of 1.
          </div>
        </div>

        {/* ── Tabs ── */}
        <div style={{ display: 'flex', gap: 8, marginBottom: 20, flexWrap: 'wrap' }}>
          {[
            { id: 'how',      label: 'How MTP Works' },
            { id: 'accept',   label: 'Accept Length' },
            { id: 'pipeline', label: 'Inference Pipeline' },
          ].map(({ id, label }) => (
            <button key={id} onClick={() => setActiveTab(id)} style={{
              padding: '7px 18px', borderRadius: 6,
              border: `1px solid ${activeTab === id ? PURPLE : 'rgba(255,255,255,.1)'}`,
              background: activeTab === id ? `${PURPLE}12` : 'rgba(255,255,255,.03)',
              color: activeTab === id ? PURPLE : '#777',
              fontFamily: M, fontSize: 11, cursor: 'pointer',
            }}>
              {label}
            </button>
          ))}
        </div>

        {/* ══════════════════════════════════════════════════════
            TAB: How MTP Works
            ════════════════════════════════════════════════════ */}
        {activeTab === 'how' && (
          <div>
            {/* Standard vs MTP comparison */}
            <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 12, marginBottom: 16 }}>
              <div style={{ ...CARD, borderColor: `${GRAY}44` }}>
                <div style={{ fontSize: 11, fontFamily: M, color: GRAY, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>
                  Standard Autoregressive Decode
                </div>
                <div style={{ fontSize: 12, color: '#999', lineHeight: 1.6, marginBottom: 12 }}>
                  One token per forward pass. Each step runs the full model (78 layers, 744B params)
                  to produce a single next-token prediction.
                </div>
                <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 8 }}>
                  {['The', 'cat', 'sat', 'on', 'the', 'mat'].map((t, i) => (
                    <div key={i} style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                      <span style={{ fontFamily: M, fontSize: 8, color: '#555' }}>step {i + 1}</span>
                      <TokenBlock text={t} color={GRAY} />
                    </div>
                  ))}
                </div>
                <div style={{ fontFamily: M, fontSize: 10, color: GRAY }}>
                  6 tokens = 6 forward passes
                </div>
              </div>

              <div style={{ ...CARD, borderColor: `${PURPLE}44` }}>
                <div style={{ fontSize: 11, fontFamily: M, color: PURPLE, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>
                  MTP Speculative Decode
                </div>
                <div style={{ fontSize: 12, color: '#999', lineHeight: 1.6, marginBottom: 12 }}>
                  Draft multiple tokens using 3 lightweight shared layers, then verify all at once.
                  Accept 2-3 tokens per step on average.
                </div>
                <div style={{ display: 'flex', gap: 4, flexWrap: 'wrap', marginBottom: 8 }}>
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                    <span style={{ fontFamily: M, fontSize: 8, color: '#555' }}>step 1</span>
                    <div style={{ display: 'flex', gap: 2 }}>
                      <TokenBlock text="The" color={LIME} />
                      <TokenBlock text="cat" color={CYAN} small />
                      <TokenBlock text="sat" color={CYAN} small />
                    </div>
                  </div>
                  <div style={{ display: 'flex', flexDirection: 'column', alignItems: 'center', gap: 2 }}>
                    <span style={{ fontFamily: M, fontSize: 8, color: '#555' }}>step 2</span>
                    <div style={{ display: 'flex', gap: 2 }}>
                      <TokenBlock text="on" color={LIME} />
                      <TokenBlock text="the" color={CYAN} small />
                      <TokenBlock text="mat" color={CYAN} small />
                    </div>
                  </div>
                </div>
                <div style={{ fontFamily: M, fontSize: 10, color: PURPLE }}>
                  6 tokens = 2 forward passes (~3x faster)
                </div>
              </div>
            </div>

            {/* Architecture diagram */}
            <div style={{ ...CARD, marginBottom: 16 }}>
              <div style={{ fontSize: 11, fontFamily: M, color: AMBER, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 14 }}>
                MTP Architecture: 3 Shared Draft Layers
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 8 }}>
                {/* Main model */}
                <div style={{ padding: '10px 16px', borderRadius: 6, background: `${GRAY}10`, border: `1px solid ${GRAY}33` }}>
                  <div style={{ fontFamily: M, fontSize: 11, color: GRAY, marginBottom: 4 }}>Main Model (78 layers)</div>
                  <div style={{ fontSize: 11, color: '#666' }}>
                    Full transformer: MLA attention + MoE FFN. Produces hidden state h_L for position t.
                  </div>
                </div>
                <div style={{ fontFamily: M, fontSize: 14, color: '#333', textAlign: 'center' }}>{'\u2193'} hidden state h_L</div>
                {/* MTP modules */}
                {[1, 2, 3].map((k) => (
                  <div key={k} style={{
                    padding: '10px 16px', borderRadius: 6,
                    background: `${PURPLE}08`, border: `1px solid ${PURPLE}33`,
                    marginLeft: k * 20,
                  }}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
                      <div>
                        <div style={{ fontFamily: M, fontSize: 11, color: PURPLE }}>MTP Module {k}</div>
                        <div style={{ fontSize: 10, color: '#555' }}>
                          Shared transformer layers + projection head. Predicts token at position t+{k}.
                        </div>
                      </div>
                      <div style={{
                        fontFamily: M, fontSize: 9, color: CYAN,
                        padding: '3px 8px', borderRadius: 4,
                        background: `${CYAN}10`, border: `1px solid ${CYAN}33`,
                      }}>
                        t+{k}
                      </div>
                    </div>
                  </div>
                ))}
              </div>
            </div>

            <MathBlock label="MTP training objective">
              <Var color={PURPLE}>L</Var><Sub>MTP</Sub>
              {' = '}
              <Var>L</Var><Sub>main</Sub>
              {' + '}
              <Var color={AMBER}>{'\u03BB'}</Var>
              {' \u2211'}
              <Sub>k=1..3</Sub>
              {' '}
              <Var>L</Var><Sub>k</Sub>
              {'   where '}
              <Var>L</Var><Sub>k</Sub>
              {' = CE(MTP'}
              <Sub>k</Sub>
              {'(h'}
              <Sub>L</Sub>
              {'), y'}
              <Sub>t+k</Sub>
              {')'}
              <br />
              <span style={{ fontSize: 13, color: '#666' }}>
                Each MTP module predicts k tokens ahead; shared layers reduce parameter overhead
              </span>
            </MathBlock>

            <HL code={`# MTP: Multi-Token Prediction — 3 shared draft layers
class MTPModule(nn.Module):
    def __init__(self, config):
        # Shared transformer layers (lightweight — same arch, fewer layers)
        self.shared_layers = nn.ModuleList([
            TransformerLayer(config) for _ in range(3)
        ])
        # Projection: hidden → vocab (shares embedding weights)
        self.lm_head = nn.Linear(config.d_model, config.vocab_size, bias=False)
        self.lm_head.weight = model.embed_tokens.weight  # weight tying

    def forward(self, hidden_state):
        h = hidden_state
        for layer in self.shared_layers:
            h = layer(h)
        return self.lm_head(h)  # logits for next+k token

# Training: predict tokens at positions t+1, t+2, t+3
# Inference: use as draft model for speculative decoding`} />
          </div>
        )}

        {/* ══════════════════════════════════════════════════════
            TAB: Accept Length
            ════════════════════════════════════════════════════ */}
        {activeTab === 'accept' && (
          <div>
            <div style={{ fontSize: 13, color: '#888', lineHeight: 1.65, marginBottom: 20, maxWidth: 680 }}>
              Average accepted token length measures how many draft tokens pass verification per step.
              Higher is better -- it directly translates to throughput improvement. GLM-5 achieves
              2.76 vs DeepSeek-V3's 2.55 thanks to 3 shared draft layers.
            </div>

            {/* Bar comparison */}
            <div style={{ ...CARD, marginBottom: 16 }}>
              <div style={{ fontSize: 11, fontFamily: M, color: CYAN, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 16 }}>
                Average Accepted Length
              </div>
              <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
                {[
                  { label: 'DeepSeek-V3', val: 2.55, color: AMBER, layers: '1 MTP module' },
                  { label: 'GLM-5', val: 2.76, color: CYAN, layers: '3 shared draft layers' },
                ].map(({ label, val, color, layers }) => (
                  <div key={label}>
                    <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', marginBottom: 6 }}>
                      <div>
                        <span style={{ fontFamily: M, fontSize: 13, color, fontWeight: 600 }}>{label}</span>
                        <span style={{ fontFamily: M, fontSize: 10, color: '#555', marginLeft: 8 }}>{layers}</span>
                      </div>
                      <div style={{
                        fontFamily: M, fontSize: 22, fontWeight: 700, color,
                        padding: '3px 12px', borderRadius: 5,
                        background: `${color}12`, border: `1px solid ${color}33`,
                      }}>
                        {val}
                      </div>
                    </div>
                    <div style={{ height: 16, borderRadius: 4, background: 'rgba(255,255,255,.05)', overflow: 'hidden' }}>
                      <div style={{
                        height: '100%', width: `${(val / 4) * 100}%`,
                        background: color, opacity: 0.65, borderRadius: 4,
                      }} />
                    </div>
                  </div>
                ))}
              </div>
            </div>

            {/* Speedup calculator */}
            <div style={{ ...CARD, marginBottom: 16 }}>
              <div style={{ fontSize: 11, fontFamily: M, color: LIME, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 14 }}>
                Speedup Calculator
              </div>
              <div style={{ marginBottom: 14 }}>
                <label style={{ fontFamily: M, fontSize: 11, color: '#888', display: 'block', marginBottom: 6 }}>
                  Average accepted length: <span style={{ color: CYAN }}>{acceptLen.toFixed(2)}</span>
                </label>
                <input
                  type="range" min={1} max={4} step={0.01} value={acceptLen}
                  onChange={e => setAcceptLen(Number(e.target.value))}
                  style={{ width: '100%', maxWidth: 400, accentColor: CYAN }}
                />
              </div>
              <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 12 }}>
                <div style={{ padding: '12px 14px', borderRadius: 6, background: `${GRAY}14`, border: `1px solid ${GRAY}44` }}>
                  <div style={{ fontFamily: M, fontSize: 9, color: '#666', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>Standard Decode</div>
                  <div style={{ fontFamily: M, fontSize: 20, fontWeight: 700, color: GRAY }}>{standardTps}</div>
                  <div style={{ fontFamily: M, fontSize: 9, color: '#555' }}>tokens/sec (1 per step)</div>
                </div>
                <div style={{ padding: '12px 14px', borderRadius: 6, background: `${CYAN}14`, border: `1px solid ${CYAN}44` }}>
                  <div style={{ fontFamily: M, fontSize: 9, color: '#666', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>MTP Decode</div>
                  <div style={{ fontFamily: M, fontSize: 20, fontWeight: 700, color: CYAN }}>{mtpTps}</div>
                  <div style={{ fontFamily: M, fontSize: 9, color: '#555' }}>tokens/sec (effective)</div>
                </div>
                <div style={{ padding: '12px 14px', borderRadius: 6, background: `${LIME}14`, border: `1px solid ${LIME}44` }}>
                  <div style={{ fontFamily: M, fontSize: 9, color: '#666', textTransform: 'uppercase', letterSpacing: 1, marginBottom: 4 }}>Speedup</div>
                  <div style={{ fontFamily: M, fontSize: 20, fontWeight: 700, color: LIME }}>{speedup}x</div>
                  <div style={{ fontFamily: M, fontSize: 9, color: '#555' }}>effective throughput gain</div>
                </div>
              </div>
            </div>

            <MathBlock label="Speculative decoding speedup">
              <Var color={LIME}>speedup</Var>
              {' \u2248 '}
              <Var color={CYAN}>E</Var>
              {'[accepted tokens per step]  =  '}
              <Var color={CYAN}>{acceptLen.toFixed(2)}</Var>
              {'x'}
              <br />
              <span style={{ fontSize: 13, color: '#666' }}>
                GLM-5: 2.76x vs DeepSeek-V3: 2.55x (+8.2% more tokens per step)
              </span>
            </MathBlock>

            {/* Why GLM-5 is better */}
            <div style={{ ...CARD, marginBottom: 16, borderColor: `${PURPLE}33`, background: `${PURPLE}06` }}>
              <div style={{ fontSize: 11, fontFamily: M, color: PURPLE, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 10 }}>
                Why GLM-5 MTP Accepts More Tokens
              </div>
              <div style={{ fontSize: 13, color: '#bbb', lineHeight: 1.65 }}>
                GLM-5 uses <strong style={{ color: PURPLE }}>3 shared transformer layers</strong> in its MTP draft head
                (vs DeepSeek-V3's single MTP module). The additional draft layers improve
                draft quality -- the draft distribution more closely matches the main model,
                so more tokens pass the speculative verification check. The overhead of 3 extra
                lightweight layers is negligible compared to the 78-layer main model.
              </div>
            </div>
          </div>
        )}

        {/* ══════════════════════════════════════════════════════
            TAB: Inference Pipeline
            ════════════════════════════════════════════════════ */}
        {activeTab === 'pipeline' && (
          <div>
            <div style={{ fontSize: 13, color: '#888', lineHeight: 1.65, marginBottom: 20, maxWidth: 680 }}>
              Step through one iteration of speculative decoding. The draft model proposes
              multiple tokens, the main model verifies them in parallel, and accepted tokens
              are appended to the sequence.
            </div>

            {/* Step controls */}
            <div style={{ display: 'flex', gap: 8, marginBottom: 16, flexWrap: 'wrap' }}>
              {[
                { id: 0, label: 'Draft' },
                { id: 1, label: 'Verify' },
                { id: 2, label: 'Accept/Reject' },
                { id: 3, label: 'Update' },
              ].map(({ id, label }) => (
                <button key={id} onClick={() => setPipeStep(id)} style={{
                  padding: '7px 18px', borderRadius: 6,
                  border: `1px solid ${pipeStep === id ? PURPLE : 'rgba(255,255,255,.1)'}`,
                  background: pipeStep === id ? `${PURPLE}12` : 'rgba(255,255,255,.03)',
                  color: pipeStep === id ? PURPLE : '#777',
                  fontFamily: M, fontSize: 11, cursor: 'pointer',
                }}>
                  Step {id + 1}: {label}
                </button>
              ))}
            </div>

            {/* Step 1: Draft */}
            {pipeStep === 0 && (
              <div style={{ ...CARD, marginBottom: 16, borderColor: `${CYAN}33` }}>
                <div style={{ fontSize: 11, fontFamily: M, color: CYAN, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>
                  Step 1: Draft — MTP head proposes tokens
                </div>
                <div style={{ fontSize: 12, color: '#bbb', lineHeight: 1.6, marginBottom: 14 }}>
                  The MTP draft head runs 3 shared transformer layers on the last hidden state.
                  It produces logits for positions t+1, t+2, t+3 and samples draft tokens.
                </div>
                <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap', marginBottom: 10 }}>
                  <span style={{ fontFamily: M, fontSize: 10, color: '#555' }}>Context:</span>
                  <TokenBlock text="The" color={GRAY} />
                  <TokenBlock text="quick" color={GRAY} />
                  <TokenBlock text="brown" color={GRAY} />
                  <span style={{ fontFamily: M, fontSize: 12, color: '#333' }}>{'\u2192'}</span>
                  <span style={{ fontFamily: M, fontSize: 10, color: '#555' }}>Draft:</span>
                  <TokenBlock text="fox" color={CYAN} />
                  <TokenBlock text="jumps" color={CYAN} />
                  <TokenBlock text="over" color={CYAN} />
                </div>
                <div style={{ fontFamily: M, fontSize: 10, color: '#555' }}>
                  Cost: 3 lightweight transformer layers (~0.5% of main model FLOPs)
                </div>
              </div>
            )}

            {/* Step 2: Verify */}
            {pipeStep === 1 && (
              <div style={{ ...CARD, marginBottom: 16, borderColor: `${AMBER}33` }}>
                <div style={{ fontSize: 11, fontFamily: M, color: AMBER, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>
                  Step 2: Verify — Main model checks all drafts in parallel
                </div>
                <div style={{ fontSize: 12, color: '#bbb', lineHeight: 1.6, marginBottom: 14 }}>
                  Run the full 78-layer model on the context + draft tokens. This produces the
                  "true" next-token distributions for each position. Crucially, this is done in
                  a single forward pass (not one per draft token).
                </div>
                <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap', marginBottom: 10 }}>
                  <span style={{ fontFamily: M, fontSize: 10, color: '#555' }}>Verify:</span>
                  <TokenBlock text="The" color={GRAY} />
                  <TokenBlock text="quick" color={GRAY} />
                  <TokenBlock text="brown" color={GRAY} />
                  <TokenBlock text="fox" color={AMBER} />
                  <TokenBlock text="jumps" color={AMBER} />
                  <TokenBlock text="over" color={AMBER} />
                </div>
                <div style={{ fontFamily: M, fontSize: 10, color: '#555' }}>
                  Cost: 1 forward pass through full model (same as generating 1 token normally)
                </div>
              </div>
            )}

            {/* Step 3: Accept/Reject */}
            {pipeStep === 2 && (
              <div style={{ ...CARD, marginBottom: 16, borderColor: `${LIME}33` }}>
                <div style={{ fontSize: 11, fontFamily: M, color: LIME, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>
                  Step 3: Accept/Reject — Compare draft vs main model distributions
                </div>
                <div style={{ fontSize: 12, color: '#bbb', lineHeight: 1.6, marginBottom: 14 }}>
                  For each draft token, check if it matches what the main model would have sampled.
                  Accept tokens left-to-right until the first rejection. On rejection, sample from
                  the adjusted distribution.
                </div>
                <div style={{ display: 'flex', gap: 6, alignItems: 'center', flexWrap: 'wrap', marginBottom: 10 }}>
                  <span style={{ fontFamily: M, fontSize: 10, color: '#555' }}>Result:</span>
                  <TokenBlock text="fox" color={LIME} />
                  <span style={{ fontFamily: M, fontSize: 8, color: LIME }}>accepted</span>
                  <TokenBlock text="jumps" color={LIME} />
                  <span style={{ fontFamily: M, fontSize: 8, color: LIME }}>accepted</span>
                  <TokenBlock text="over" color={PINK} />
                  <span style={{ fontFamily: M, fontSize: 8, color: PINK }}>rejected</span>
                  <span style={{ fontFamily: M, fontSize: 12, color: '#333' }}>{'\u2192'}</span>
                  <TokenBlock text="leaps" color={AMBER} />
                  <span style={{ fontFamily: M, fontSize: 8, color: AMBER }}>resampled</span>
                </div>
                <div style={{ fontFamily: M, fontSize: 10, color: '#555' }}>
                  Accepted 2 + resampled 1 = 3 tokens from 1 verification pass
                </div>
              </div>
            )}

            {/* Step 4: Update */}
            {pipeStep === 3 && (
              <div style={{ ...CARD, marginBottom: 16, borderColor: `${PURPLE}33` }}>
                <div style={{ fontSize: 11, fontFamily: M, color: PURPLE, textTransform: 'uppercase', letterSpacing: 1, marginBottom: 12 }}>
                  Step 4: Update — Append accepted tokens, repeat
                </div>
                <div style={{ fontSize: 12, color: '#bbb', lineHeight: 1.6, marginBottom: 14 }}>
                  Append all accepted tokens + the resampled token to the sequence. Update the
                  KV cache. Start the next draft cycle from the new position.
                </div>
                <div style={{ display: 'flex', gap: 4, alignItems: 'center', flexWrap: 'wrap', marginBottom: 10 }}>
                  <TokenBlock text="The" color={GRAY} />
                  <TokenBlock text="quick" color={GRAY} />
                  <TokenBlock text="brown" color={GRAY} />
                  <TokenBlock text="fox" color={LIME} />
                  <TokenBlock text="jumps" color={LIME} />
                  <TokenBlock text="leaps" color={AMBER} />
                  <span style={{ fontFamily: M, fontSize: 12, color: '#333' }}>{'\u2192'}</span>
                  <span style={{ fontFamily: M, fontSize: 10, color: CYAN }}>draft next 3...</span>
                </div>
                <div style={{ fontFamily: M, fontSize: 10, color: '#555' }}>
                  Generated 3 tokens in 1 main-model forward pass. Repeat until done.
                </div>
              </div>
            )}

            <MathBlock label="Speculative decoding guarantee">
              <span style={{ fontSize: 14, color: '#999' }}>
                P(accept token k) = min(1, P<Sub>main</Sub>(t<Sub>k</Sub>) / P<Sub>draft</Sub>(t<Sub>k</Sub>))
              </span>
              <br />
              <span style={{ fontSize: 13, color: '#666' }}>
                Output distribution is identical to the main model -- speculation is lossless
              </span>
            </MathBlock>

            <HL code={`# Speculative decoding with MTP draft head
def speculative_decode(model, mtp_head, input_ids, max_tokens):
    generated = []
    while len(generated) < max_tokens:
        # Step 1: Draft — MTP head proposes K tokens
        hidden = model.get_hidden(input_ids)
        draft_tokens = []
        h = hidden[:, -1:]
        for mtp_layer in mtp_head.shared_layers:
            h = mtp_layer(h)
            logits = mtp_head.lm_head(h)
            token = torch.argmax(logits, dim=-1)
            draft_tokens.append(token)

        # Step 2: Verify — single forward pass with drafts
        all_tokens = torch.cat([input_ids, *draft_tokens], dim=-1)
        main_logits = model(all_tokens).logits

        # Step 3: Accept/Reject
        accepted = []
        for k, draft_tok in enumerate(draft_tokens):
            pos = input_ids.shape[-1] + k - 1
            if matches(main_logits[:, pos], draft_tok):
                accepted.append(draft_tok)
            else:
                accepted.append(resample(main_logits[:, pos]))
                break

        # Step 4: Update
        input_ids = torch.cat([input_ids, *accepted], dim=-1)
        generated.extend(accepted)`} />
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
