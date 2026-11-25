#!/usr/bin/env python3
"""Shmoo: model-guided sfc_ca_gemm runs vs. sfc_ca_perf_model predictions, plus a correlation SVG.
Usage: SHMOO_MODEL_ARGS="--bw ... --macs ... --eff-br ... --c-bw ... --freq ..." python3 shmoo.py [out_dir]
       (run from the sfc_ca_gemm directory; SHMOO_THREADS defaults to all CPUs)
"""
import csv, math, os, re, shlex, subprocess, sys

M_LIST = [8192, 16384, 32768]
N_LIST = [1024, 2048, 4096, 8192]
K_LIST = [2048, 4096]

THREADS = int(os.environ.get("SHMOO_THREADS", os.cpu_count()))
TARGET_SECONDS = 4.0      # timed work per real run
# Platform calibration (per-sharer bandwidth, MACs, efficiency, C bandwidth, frequency) comes from the user.
PLATFORM_ARGS = shlex.split(os.environ.get("SHMOO_MODEL_ARGS", ""))
MODEL_ARGS = ["--threads", str(THREADS)] + PLATFORM_ARGS + [
              "--overlap-c", "--sched", "greedy", "--step-max", "1", "--top", "1"]
RUN_ENV = {"OMP_NUM_THREADS": str(THREADS),
           "LIBXSMM_X86_AMX_GEMM_STREAMING_A": "1", "LIBXSMM_X86_AMX_GEMM_STREAMING_B": "1",
           "KMP_AFFINITY": "granularity=fine,compact,1,0"}


def run_model(M, N, K):
    out = subprocess.run(["./sfc_ca_perf_model", str(M), str(N), str(K)] + MODEL_ARGS,
                         capture_output=True, text=True, check=True).stdout
    row = next(l for l in out.splitlines() if re.match(r"^1\s+\d+\s+\d+,\d+", l))
    f = row.split()
    cmd = re.search(r"#1 \./sfc_ca_gemm (.+)", out).group(1).split()
    return {"K_layers": int(f[1]), "steps": f[2], "kbf": f[3], "patch": f[5], "grid": f[6],
            "model_pct": float(f[-2]), "model_tflops": float(f[-1]), "args": cmd}


def auto_layers(M, N, K):
    # Same rule as sfc_ca_gemm with n_layers = -1 (BF16 A/B/C, >= 5 GiB in total).
    per = 2.0 * (M * K + M * N + K * N) / 2**30
    n = 1
    while n * per < 5.0:
        n += 1
    return n


def run_real(M, N, K, model):
    n_layers = auto_layers(M, N, K)
    t_iter = n_layers * 2.0 * M * N * K / (model["model_tflops"] * 1e12)
    n_iters = max(20, min(20000, math.ceil(TARGET_SECONDS / t_iter)))
    args = list(model["args"])
    args[9] = str(n_iters)  # positional n_iters
    env = dict(os.environ, **RUN_ENV)
    res = subprocess.run(["./sfc_ca_gemm"] + args, capture_output=True, text=True, env=env)
    out = res.stdout
    m = re.search(r"Measured freq ([\d.]+) GHz -> ([\d.]+) FLOPs/cycle/core -> ([\d.]+)% of peak", out)
    t = re.search(r"Time is ([\d.]+) ms \(([\d.]+) GFLOPS\)", out)
    if not m or not t:
        raise RuntimeError("could not parse sfc_ca_gemm output:\n" + out[-2000:] + res.stderr[-2000:])
    return {"n_layers": n_layers, "n_iters": n_iters, "meas_ghz": float(m.group(1)),
            "meas_pct": float(m.group(3)), "meas_tflops": float(t.group(2)) / 1000.0,
            "cmd": "./sfc_ca_gemm " + " ".join(args)}


def pearson(x, y):
    n = len(x)
    mx, my = sum(x) / n, sum(y) / n
    sxy = sum((a - mx) * (b - my) for a, b in zip(x, y))
    sxx = sum((a - mx) ** 2 for a in x)
    syy = sum((b - my) ** 2 for b in y)
    return sxy / math.sqrt(sxx * syy) if sxx > 0 and syy > 0 else float("nan")


def write_svg(path, rows):
    W, H, L, R, T, B = 760, 640, 80, 220, 50, 70
    xs = [r["model_pct"] for r in rows]
    ys = [r["meas_pct"] for r in rows]
    lo = 10 * math.floor(min(xs + ys) / 10)
    hi = 10 * math.ceil(max(xs + ys) / 10)
    pw, ph = W - L - R, H - T - B
    X = lambda v: L + (v - lo) / (hi - lo) * pw
    Y = lambda v: T + ph - (v - lo) / (hi - lo) * ph
    colors = {1024: "#1f77b4", 2048: "#ff7f0e", 4096: "#2ca02c", 8192: "#d62728"}
    shapes = {2048: "circle", 4096: "square"}
    r = pearson(xs, ys)
    mae = sum(abs(a - b) for a, b in zip(xs, ys)) / len(xs)
    s = [f'<svg xmlns="http://www.w3.org/2000/svg" width="{W}" height="{H}" font-family="sans-serif" font-size="12">',
         f'<rect width="{W}" height="{H}" fill="white"/>',
         f'<text x="{L + pw / 2}" y="25" text-anchor="middle" font-size="15">SFC-CA GEMM: model vs. measured % of peak '
         f'(BF16, {THREADS} cores)</text>']
    for v in range(lo, hi + 1, 10):
        s.append(f'<line x1="{X(v)}" y1="{T}" x2="{X(v)}" y2="{T + ph}" stroke="#eee"/>')
        s.append(f'<line x1="{L}" y1="{Y(v)}" x2="{L + pw}" y2="{Y(v)}" stroke="#eee"/>')
        s.append(f'<text x="{X(v)}" y="{T + ph + 18}" text-anchor="middle">{v}</text>')
        s.append(f'<text x="{L - 8}" y="{Y(v) + 4}" text-anchor="end">{v}</text>')
    s.append(f'<rect x="{L}" y="{T}" width="{pw}" height="{ph}" fill="none" stroke="black"/>')
    for d, dash in ((0, ""), (5, ' stroke-dasharray="4,4"'), (-5, ' stroke-dasharray="4,4"')):
        s.append(f'<line x1="{X(lo)}" y1="{Y(lo + d)}" x2="{X(hi - max(d, 0))}" y2="{Y(hi + min(d, 0))}" '
                 f'stroke="#888"{dash}/>' if d >= 0 else
                 f'<line x1="{X(lo - d)}" y1="{Y(lo)}" x2="{X(hi)}" y2="{Y(hi + d)}" stroke="#888"{dash}/>')
    for row in rows:
        cx, cy, c = X(row["model_pct"]), Y(row["meas_pct"]), colors[row["N"]]
        tip = f'{row["M"]}x{row["N"]}x{row["K"]}: model {row["model_pct"]:.1f}%, measured {row["meas_pct"]:.1f}%'
        if shapes[row["K"]] == "circle":
            s.append(f'<circle cx="{cx}" cy="{cy}" r="6" fill="{c}" fill-opacity="0.8"><title>{tip}</title></circle>')
        else:
            s.append(f'<rect x="{cx - 6}" y="{cy - 6}" width="12" height="12" fill="{c}" fill-opacity="0.8">'
                     f'<title>{tip}</title></rect>')
        s.append(f'<text x="{cx + 8}" y="{cy - 6}" font-size="9" fill="#444">{row["M"] // 1024}k</text>')
    s.append(f'<text x="{L + pw / 2}" y="{H - 25}" text-anchor="middle" font-size="13">model % of peak</text>')
    s.append(f'<text x="20" y="{T + ph / 2}" text-anchor="middle" font-size="13" '
             f'transform="rotate(-90 20 {T + ph / 2})">measured % of peak (at measured freq)</text>')
    lx = L + pw + 20
    s.append(f'<text x="{lx}" y="{T + 10}" font-weight="bold">N</text>')
    for i, (n, c) in enumerate(colors.items()):
        s.append(f'<circle cx="{lx + 6}" cy="{T + 30 + 20 * i}" r="6" fill="{c}"/>'
                 f'<text x="{lx + 18}" y="{T + 34 + 20 * i}">{n}</text>')
    s.append(f'<text x="{lx}" y="{T + 130}" font-weight="bold">K</text>')
    s.append(f'<circle cx="{lx + 6}" cy="{T + 150}" r="6" fill="#666"/><text x="{lx + 18}" y="{T + 154}">2048</text>')
    s.append(f'<rect x="{lx}" y="{T + 164}" width="12" height="12" fill="#666"/><text x="{lx + 18}" y="{T + 174}">4096</text>')
    s.append(f'<text x="{lx}" y="{T + 210}">label = M (k)</text>')
    s.append(f'<text x="{lx}" y="{T + 230}">solid: y = x, dashed: +/-5</text>')
    s.append(f'<text x="{lx}" y="{T + 270}" font-weight="bold">Pearson r = {r:.3f}</text>')
    s.append(f'<text x="{lx}" y="{T + 290}" font-weight="bold">MAE = {mae:.2f} pts</text>')
    s.append(f'<text x="{lx}" y="{T + 310}">points = {len(rows)}</text>')
    s.append('</svg>')
    with open(path, "w") as f:
        f.write("\n".join(s))
    return r, mae


def main():
    out_dir = sys.argv[1] if len(sys.argv) > 1 else "shmoo"
    os.makedirs(out_dir, exist_ok=True)
    csv_path = os.path.join(out_dir, "shmoo_results.csv")
    fields = ["M", "N", "K", "K_layers", "steps", "kbf", "patch", "grid", "model_pct", "model_tflops",
              "n_layers", "n_iters", "meas_ghz", "meas_pct", "meas_tflops", "cmd"]
    rows = []
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for M in M_LIST:
            for N in N_LIST:
                for K in K_LIST:
                    model = run_model(M, N, K)
                    real = run_real(M, N, K, model)
                    row = {"M": M, "N": N, "K": K, **{k: model[k] for k in fields if k in model}, **real}
                    rows.append(row)
                    w.writerow({k: row[k] for k in fields})
                    f.flush()
                    print(f"{M:>6} {N:>5} {K:>5}  KL={row['K_layers']:<2} steps={row['steps']:<4} kbf={row['kbf']:<7} "
                          f"model {row['model_pct']:5.1f}%  meas {row['meas_pct']:5.1f}% @ {row['meas_ghz']:.2f} GHz  "
                          f"(diff {row['meas_pct'] - row['model_pct']:+5.1f})", flush=True)
    r, mae = write_svg(os.path.join(out_dir, "shmoo_correlation.svg"), rows)
    print(f"\nPearson r = {r:.3f}   MAE = {mae:.2f} pts   -> {csv_path}, {out_dir}/shmoo_correlation.svg")


if __name__ == "__main__":
    main()
