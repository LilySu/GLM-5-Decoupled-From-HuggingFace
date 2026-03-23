import { useState, useEffect } from 'react';

const M = "'JetBrains Mono','Fira Code',monospace";
const S = "'Inter','Segoe UI',sans-serif";

const modules = import.meta.glob('./pages/*.jsx');

const PAGES = [
  { id: 'optimization', name: 'Optimization Stack', desc: 'PyTorch → Triton → CUDA kernels', color: '#22d3ee' },
  { id: 'mla', name: 'MLA + Muon Split', desc: 'Multi-Latent Attention deep dive', color: '#84cc16' },
  { id: 'dsa', name: 'DSA Sparse Attention', desc: 'Dynamic sparse vs other methods', color: '#f59e0b' },
  { id: 'moe', name: 'MoE System', desc: '256 experts, sigmoid routing', color: '#a855f7' },
  { id: 'mtp', name: 'MTP + Speculation', desc: 'Multi-token prediction', color: '#f472b6' },
];

export default function App() {
  const [selected, setSelected] = useState('optimization');
  const [Comp, setComp] = useState(null);
  const [loading, setLoading] = useState(false);

  useEffect(() => {
    setLoading(true);
    const path = `./pages/${selected}.jsx`;
    if (modules[path]) {
      modules[path]().then(m => { setComp(() => m.default); setLoading(false); }).catch(() => { setComp(null); setLoading(false); });
    } else { setComp(null); setLoading(false); }
  }, [selected]);

  return (
    <div style={{ display: 'flex', height: '100vh', background: '#0d0f14', color: '#e0e0e0', fontFamily: S }}>
      <div style={{ width: 260, borderRight: '1px solid #1f2433', overflowY: 'auto', flexShrink: 0, background: '#0a0c12' }}>
        <div style={{ padding: '20px 16px 14px', borderBottom: '1px solid #1f2433' }}>
          <div style={{ fontSize: 11, color: '#f59e0b', fontFamily: M, letterSpacing: 2, textTransform: 'uppercase', marginBottom: 4 }}>GLM-5</div>
          <h1 style={{ fontSize: 16, fontWeight: 700, color: '#f0f0f0', margin: 0, fontFamily: S }}>Architecture Viz</h1>
          <p style={{ fontSize: 10, color: '#555', margin: '4px 0 0', fontFamily: S }}>744B MoE · MLA · DSA · 78 layers</p>
        </div>
        {PAGES.map(p => (
          <button key={p.id} onClick={() => setSelected(p.id)} style={{
            display: 'block', width: '100%', textAlign: 'left', padding: '12px 16px',
            background: selected === p.id ? `${p.color}0a` : 'transparent',
            border: 'none', borderLeft: selected === p.id ? `3px solid ${p.color}` : '3px solid transparent',
            cursor: 'pointer', transition: 'all .15s',
          }}>
            <div style={{ fontSize: 12, fontWeight: selected === p.id ? 700 : 500, color: selected === p.id ? p.color : '#888', fontFamily: S }}>{p.name}</div>
            <div style={{ fontSize: 10, color: '#444', fontFamily: M, marginTop: 2 }}>{p.desc}</div>
          </button>
        ))}
        <div style={{ padding: '16px', fontSize: 9, color: '#333', fontFamily: M, borderTop: '1px solid #1f2433', marginTop: 12 }}>
          cd viz/app && npm run dev
        </div>
      </div>
      <div style={{ flex: 1, overflowY: 'auto', background: '#0d0f14' }}>
        {loading ? (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
            <p style={{ color: '#555', fontSize: 12, fontFamily: M }}>Loading...</p>
          </div>
        ) : Comp ? <Comp /> : (
          <div style={{ display: 'flex', alignItems: 'center', justifyContent: 'center', height: '100%' }}>
            <p style={{ color: '#444', fontSize: 12, fontFamily: M }}>Page not found</p>
          </div>
        )}
      </div>
    </div>
  );
}
