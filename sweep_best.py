#!/usr/bin/env python3
"""Model-guided sweep of sfc_ca_gemm configurations.

For every shape the performance model proposes candidates; each candidate is measured with the
round-major A/B layout and C-tile prefetching, and K_layers > 1 candidates with both the end-of-GEMM
and the lock-based reduction. Results are appended to <out>/runs.csv (resumable).
Platform settings come from the environment: SWEEP_MODEL_ARGS (calibrated sfc_ca_perf_model options,
e.g. --threads/--bw/--macs/--eff-br/...), plus OMP_NUM_THREADS, KMP_AFFINITY and
SFC_CA_GEMM_FLOPS_PER_CYCLE for the driver; SWEEP_EST_TFLOPS only sizes the iteration counts.
  python3 sweep_best.py            # run (skips finished runs)
  python3 sweep_best.py --report   # write <out>/results.xlsx, results.png and results_measured.png
"""
import argparse
import csv
import math
import os
import re
import shlex
import subprocess

MODEL_ARGS = shlex.split(os.environ.get("SWEEP_MODEL_ARGS", "")) + ["--kbf-max", "16"]
RUN_ENV = {"LIBXSMM_X86_AMX_GEMM_STREAMING_A": "1", "LIBXSMM_X86_AMX_GEMM_STREAMING_B": "1",
           "SFC_CA_GEMM_ROUND_MAJOR": "1", "SFC_CA_GEMM_PF_C": "1"}
FIELDS = ["M", "N", "K", "KL", "ms", "ns", "kbf", "reduction", "model_pct", "model_rank",
          "meas_pct", "ghz", "gflops", "round_major", "iters", "status"]


def shapes():
    ks = {4096: [1024, 2048, 4096], 8192: [1024, 2048, 4096, 8192],
          16384: [1024, 2048, 4096, 8192, 16384], 32768: [1024, 2048, 4096, 8192, 16384, 32768]}
    out = []
    for n_kind in ("M", 2048, 1024):
        for m, kl in ks.items():
            for k in kl:
                out.append((m, m if n_kind == "M" else n_kind, k))
    return out


def model_candidates(m, n, k, kl_max, top):
    cmd = ["./sfc_ca_perf_model", str(m), str(n), str(k)] + MODEL_ARGS + ["--kl-max", str(kl_max), "--top", str(top)]
    res = []
    for line in subprocess.run(cmd, capture_output=True, text=True).stdout.splitlines():
        line = re.sub(r"\(same as.*\)\s*$", "", line).rstrip()
        mt = re.match(r"^\s*(\d+)\s+(\d+)\s+(\d+),(\d+)\s+(\d+)\(\d+\)\s", line)
        if mt:
            tok = line.split()
            res.append({"rank": int(mt.group(1)), "KL": int(mt.group(2)), "ms": int(mt.group(3)),
                        "ns": int(mt.group(4)), "kbf": int(mt.group(5)), "pct": float(tok[-2])})
    return res


def pick(m, n, k):
    """Model-ranked candidates restricted to round-major-compatible kbf: the top 2 with K_layers = 1
    and the best configuration of the two highest-ranked K_layers > 1 values."""
    allc = model_candidates(m, n, k, 8, 1000)
    kb = k // 32

    def rm_ok(c):
        kbl = -(-kb // c["KL"])
        brc = -(-kbl // c["kbf"])
        return c["kbf"] & (c["kbf"] - 1) == 0 and kbl % brc == 0 and kb % brc == 0

    valid = [c for c in allc if rm_ok(c)]
    chosen = [c for c in valid if c["KL"] == 1][:2]
    best_per_kl = {}
    for c in valid:
        if c["KL"] > 1 and c["KL"] not in best_per_kl:
            best_per_kl[c["KL"]] = c
    chosen += sorted(best_per_kl.values(), key=lambda c: c["rank"])[:2]
    best_model = max([x["pct"] for x in allc] or [float("nan")])
    return chosen, best_model


def iters_for(m, n, k):
    set_gb = 2.0 * (m * k + m * n + k * n) / 2**30
    layers = max(1, math.ceil(5.0 / set_gb))
    t_iter = layers * 2.0 * m * n * k / (float(os.environ.get("SWEEP_EST_TFLOPS", "100")) * 1e12)
    return int(min(50, max(3, math.ceil(1.0 / t_iter))))


def measure(m, n, k, c, reduction, iters):
    env = dict(os.environ, **RUN_ENV)
    env.pop("SFC_CA_GEMM_REDUCTION", None)
    if reduction == "locks":
        env["SFC_CA_GEMM_REDUCTION"] = "locks"
    cmd = ["./sfc_ca_gemm", str(m), str(n), str(k), "32", "32", "32", str(c["kbf"]), str(c["KL"]), "-1",
           str(iters), "0", "BF16", str(c["ms"]), str(c["ns"])]
    try:
        p = subprocess.run(cmd, env=env, capture_output=True, text=True, timeout=1800)
    except subprocess.TimeoutExpired:
        return {"status": "timeout"}
    out = p.stdout
    r = {"status": "ok" if p.returncode == 0 else f"exit {p.returncode}",
         "round_major": int("round-major A/B layout" in out)}
    mf = re.search(r"Measured freq ([\d.]+) GHz .*-> ([\d.]+)% of peak", out)
    mg = re.search(r"Time is .*\(([\d.]+) GFLOPS\)", out)
    if mf:
        r["ghz"], r["meas_pct"] = float(mf.group(1)), float(mf.group(2))
    if mg:
        r["gflops"] = float(mg.group(1))
    if not mf and r["status"] == "ok":
        r["status"] = "no result"
    return r


def run(out_dir):
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "runs.csv")
    done = set()
    if os.path.exists(path):
        with open(path) as f:
            for row in csv.DictReader(f):
                done.add(tuple(row[x] for x in ("M", "N", "K", "KL", "ms", "ns", "kbf", "reduction")))
    new = not os.path.exists(path)
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS + ["model_best_pct"])
        if new:
            w.writeheader()
        for m, n, k in shapes():
            cands, best_model = pick(m, n, k)
            iters = iters_for(m, n, k)
            for c in cands:
                for red in (["default", "locks"] if c["KL"] > 1 else ["none"]):
                    key = tuple(str(x) for x in (m, n, k, c["KL"], c["ms"], c["ns"], c["kbf"], red))
                    if key in done:
                        continue
                    r = measure(m, n, k, c, red, iters)
                    row = {"M": m, "N": n, "K": k, "KL": c["KL"], "ms": c["ms"], "ns": c["ns"], "kbf": c["kbf"],
                           "reduction": red, "model_pct": c["pct"], "model_rank": c["rank"], "iters": iters,
                           "model_best_pct": best_model, **r}
                    w.writerow(row)
                    f.flush()
                    print(f"{m}x{n}x{k} KL{c['KL']} {c['ms']},{c['ns']} kbf{c['kbf']} {red}: "
                          f"{r.get('meas_pct', '-')}% (model {c['pct']}%) {r['status']}", flush=True)


def report(out_dir):
    import openpyxl
    from openpyxl.styles import Font
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    with open(os.path.join(out_dir, "runs.csv")) as f:
        rows = list(csv.DictReader(f))
    order = {s: i for i, s in enumerate(shapes())}
    best = {}
    for r in rows:
        s = (int(r["M"]), int(r["N"]), int(r["K"]))
        pct = float(r["meas_pct"]) if r.get("meas_pct") else -1
        if s not in best or pct > best[s][0]:
            best[s] = (pct, r)
    shp = sorted(best, key=lambda s: order.get(s, 1e9))

    wb = openpyxl.Workbook()
    ws = wb.active
    ws.title = "best per shape"
    hdr = ["M", "N", "K", "best % of peak", "GFLOPS", "GHz", "K_layers", "m_step", "n_step", "kbf", "reduction",
           "model % (this config)", "model best %"]
    ws.append(hdr)
    for s in shp:
        pct, r = best[s]
        ws.append([*s, pct, float(r["gflops"] or 0), float(r["ghz"] or 0), int(r["KL"]), int(r["ms"]), int(r["ns"]),
                   int(r["kbf"]), r["reduction"], float(r["model_pct"]), float(r["model_best_pct"])])
    ws2 = wb.create_sheet("all runs")
    ws2.append(FIELDS + ["model_best_pct"])
    for r in sorted(rows, key=lambda r: order.get((int(r["M"]), int(r["N"]), int(r["K"])), 1e9)):
        ws2.append([r.get(x, "") for x in FIELDS + ["model_best_pct"]])
    for sh in (ws, ws2):
        for c in sh[1]:
            c.font = Font(bold=True)
    xlsx = os.path.join(out_dir, "results.xlsx")
    wb.save(xlsx)

    groups = [("N = M", lambda s: s[1] == s[0]), ("N = 2048", lambda s: s[1] == 2048 and s[0] != 2048),
              ("N = 1024", lambda s: s[1] == 1024 and s[0] != 1024)]
    pngs = []
    for with_model, name in ((True, "results.png"), (False, "results_measured.png")):
        fig, axes = plt.subplots(len(groups), 1, figsize=(16, 13))
        for ax, (title, sel) in zip(axes, groups):
            g = [s for s in shp if sel(s)]
            x = range(len(g))
            meas = [best[s][0] for s in g]
            bars = ax.bar(x, meas, color="tab:blue", label="measured (best config)")
            if with_model:
                ax.plot(x, [float(best[s][1]["model_best_pct"]) for s in g], "o-", color="tab:red",
                        label="model (best predicted)")
            for b, s in zip(bars, g):
                r = best[s][1]
                tag = f"KL{r['KL']} kbf{r['kbf']}" + (" L" if r["reduction"] == "locks" else "")
                ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 1, tag, ha="center", va="bottom", fontsize=6, rotation=90)
            ax.set_xticks(list(x))
            ax.set_xticklabels([f"{m}x{n}x{k}" for m, n, k in g], rotation=60, ha="right", fontsize=7)
            ax.set_ylabel("% of peak (measured freq)")
            ax.set_ylim(0, 105)
            ax.set_title(f"SFC-CA GEMM BF16, {title}")
            ax.grid(axis="y", alpha=0.3)
            ax.legend(loc="upper left", fontsize=8)
        fig.tight_layout()
        pngs.append(os.path.join(out_dir, name))
        fig.savefig(pngs[-1], dpi=130)
        plt.close(fig)
    print(f"wrote {xlsx}, {' and '.join(pngs)}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default="sweep_best")
    ap.add_argument("--report", action="store_true")
    a = ap.parse_args()
    report(a.out) if a.report else run(a.out)
