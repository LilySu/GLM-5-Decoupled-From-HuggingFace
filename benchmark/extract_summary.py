"""Extract compact summary from all benchmark JSON files."""
import json
import glob
import os

root = "/workspace/GLM-5-Decoupled-From-HuggingFace"
files = [os.path.join(root, "bench_results.json")]
files += sorted(glob.glob(os.path.join(root, "results", "**", "*.json"), recursive=True))

for f in files:
    if not os.path.isfile(f):
        continue
    try:
        d = json.load(open(f))
        results = d.get("results", [])
        meta = d.get("metadata", {})
        env = d.get("environment", {})
        gpu = d.get("gpu", env.get("gpu_name", "?"))
        exp = d.get("experiment", d.get("config", "?"))
        print(f"FILE: {os.path.relpath(f, root)}")
        print(f"  GPU: {gpu} | Experiment: {exp} | Results: {len(results)}")
        for r in results:
            name = r.get("name", "?")
            impl = r.get("impl", r.get("batch", "?"))
            med = r.get("median_ms", 0)
            tf = r.get("tflops", 0) or 0
            mfu = r.get("mfu_pct", 0) or 0
            bw = r.get("bandwidth_gb_s", 0) or 0
            sol = r.get("hbm_sol_pct", 0) or 0
            mem = r.get("peak_memory_gb", 0) or 0
            oom = r.get("is_oom", False)
            err = r.get("error", "")
            cfg = r.get("config", {})
            if isinstance(cfg, dict):
                tokens = cfg.get("n_tokens", cfg.get("batch_size", ""))
                prec = cfg.get("precision", "")
                extra = f"T={tokens} {prec}" if tokens else ""
            else:
                extra = str(cfg)[:30]
            if oom:
                print(f"  {name:35s} {str(impl):20s} OOM")
            elif err:
                print(f"  {name:35s} {str(impl):20s} ERR:{err[:40]}")
            else:
                print(f"  {name:35s} {str(impl):20s} {med:9.3f}ms {tf:8.1f}TF {mfu:5.1f}%MFU {mem:5.1f}GB {extra}")
        print()
    except Exception as e:
        print(f"FILE: {f} ERROR: {e}")
        print()
