/******************************************************************************
* Copyright (c) Intel Corporation - All rights reserved.                      *
* This file is part of the LIBXSMM library.                                   *
*                                                                             *
* For information on the license, see the LICENSE file.                       *
* Further information: https://github.com/libxsmm/libxsmm/                    *
* SPDX-License-Identifier: BSD-3-Clause                                       *
******************************************************************************/
/* Evangelos Georganas (Intel Corp.)
******************************************************************************/

#include "sfc_ca_perf_model.hpp"

#include <array>
#include <atomic>
#include <cstdio>
#include <cstring>
#include <set>
#include <sstream>
#include <stdexcept>
#include <thread>

using namespace sfc_model;

static void usage(const char *prog) {
  printf(
    "Usage: %s M N K [options]\n"
    "  --bm/--bn/--bk <int>     tile sizes (default 32)\n"
    "  --threads <int>          number of cores/threads (default 64)\n"
    "  --bw \"s:B,...\"           per-core Bytes/cycle when a panel has s sharers,\n"
    "                           e.g. \"1:6,2:8,4:10,8:12,16:14,32:16\"\n"
    "  --bw-flat <B/cycle>      per-core Bytes/cycle independent of sharers (instead of --bw)\n"
    "  --c-bw <B/cycle>         per-core Bytes/cycle for C writes/reduction (default bw(1))\n"
    "  --overlap-c              exclude the final C write-backs from the critical path\n"
    "                           (C tiles evicted between K rounds and the K_layers reduction still counted)\n"
    "  --pf-smooth              ideal cross-BRGEMM prefetch: A/B loads overlap any compute of the core\n"
    "                           (time = max(total compute, total A/B load) instead of per-BRGEMM max)\n"
    "  --macs <MACs/cycle>      BF16 MACs/cycle/core (default 1024)\n"
    "  --eff <percent>          achieved %% of MAC peak inside BRGEMM (default 100)\n"
    "  --eff-br \"br:P,...\"       achieved %% of MAC peak per BRGEMM brcount (overrides --eff),\n"
    "                           e.g. \"8:70,16:80,32:90,64:95\"\n"
    "  --freq <GHz>             core frequency, only to report TFLOP/s\n"
    "  --l2-kb <KiB>            per-core fast memory, 0 = infinite (default 2048)\n"
    "  --kl-max <int>           max K_layers (default 16)\n"
    "  --step-max <int>         max m_step/n_step (default 4)\n"
    "  --kbf-max <int>          max k_block_factor (default 8)\n"
    "  --sched balanced|greedy  OpenMP static partition: libgomp|libomp (default balanced)\n"
    "  --in-bytes/--out-bytes   element sizes of A,B / C (default 2/2, BF16)\n"
    "  --top <int>              number of decompositions to report (default 5)\n"
    "  --eval KL,ms,ns,kbf      evaluate only this configuration\n"
    "  --dump-cores             with --eval: print every core's exact A/B/C partition\n"
    "  --dump-map <file>        write the core-ownership map of the --eval config (else of rank 1):\n"
    "                           *.svg -> colored SVG, '-' -> ASCII to stdout, otherwise ASCII file\n"
    "  --jobs <int>             host threads used by the search (default all)\n", prog);
}

static std::map<long, double> parse_bw(const std::string &s, const char *opt = "--bw") {
  std::map<long, double> m;
  std::stringstream ss(s);
  std::string item;
  while (std::getline(ss, item, ',')) {
    size_t colon = item.find(':');
    if (colon == std::string::npos) throw std::runtime_error(std::string("bad ") + opt + " entry: " + item);
    long sh = std::stol(item.substr(0, colon));
    double bw = std::stod(item.substr(colon + 1));
    if (sh < 1 || bw <= 0) throw std::runtime_error(std::string("bad ") + opt + " entry: " + item);
    m[sh] = bw;
  }
  return m;
}

// Owner core of every C tile, indexed [layer][im][in].
static std::vector<int> owner_grid(const Footprint &fp) {
  std::vector<int> own((size_t)fp.K_layers * fp.Mb * fp.Nb, -1);
  for (size_t t = 0; t < fp.cores.size(); t++)
    for (const Task &tk : fp.cores[t].tasks) own[((size_t)tk.layer * fp.Mb + tk.im) * fp.Nb + tk.in] = (int)t;
  return own;
}

// Cores reading each A panel (by_rows: one per im) or B panel (one per in) of layer l, sorted.
static std::vector<std::vector<int>> panel_readers(const std::vector<int> &own, long l, long Mb, long Nb, bool by_rows) {
  const long np = by_rows ? Mb : Nb, nq = by_rows ? Nb : Mb;
  std::vector<std::vector<int>> r(np);
  for (long i = 0; i < np; i++) {
    for (long j = 0; j < nq; j++)
      r[i].push_back(own[((size_t)l * Mb + (by_rows ? i : j)) * Nb + (by_rows ? j : i)]);
    std::sort(r[i].begin(), r[i].end());
    r[i].erase(std::unique(r[i].begin(), r[i].end()), r[i].end());
  }
  return r;
}

// Maximal runs [lo, hi) of consecutive panels read by the same set of cores.
static std::vector<std::pair<long, long>> reader_bands(const std::vector<std::vector<int>> &rd) {
  std::vector<std::pair<long, long>> b;
  for (long i = 0; i < (long)rd.size();) {
    long j = i + 1;
    while (j < (long)rd.size() && rd[j] == rd[i]) j++;
    b.push_back({i, j});
    i = j;
  }
  return b;
}

// "0-3,8,10-11", cut to max_len characters.
static std::string core_ranges(const std::vector<int> &v, size_t max_len = 60) {
  std::string s;
  for (size_t i = 0; i < v.size();) {
    size_t j = i;
    while (j + 1 < v.size() && v[j + 1] == v[j] + 1) j++;
    s += (s.empty() ? "" : ",") + std::to_string(v[i]) + (j > i ? "-" + std::to_string(v[j]) : "");
    i = j + 1;
  }
  if (s.size() > max_len) s = s.substr(0, max_len > 3 ? max_len - 3 : 0) + "...";
  return s;
}

static void write_ascii_map(FILE *f, const Footprint &fp) {
  const std::vector<int> own = owner_grid(fp);
  const long Mb = fp.Mb, Nb = fp.Nb;
  const int w = (int)std::max(std::to_string(fp.cores.size() - 1).size(), std::to_string(Nb - 1).size()) + 1;
  fprintf(f, "Core-ownership map: K_layers=%ld m_step=%ld n_step=%ld, rows=im (A panels), cols=in (B panels)\n",
          fp.K_layers, fp.m_step, fp.n_step);
  for (long l = 0; l < fp.K_layers; l++) {
    fprintf(f, "\nK-layer %ld: entry = core id, '|' column = A-panel sharers, last row = B-panel sharers\n%6s", l, "");
    for (long n = 0; n < Nb; n++) fprintf(f, "%*ld", w, n);
    fprintf(f, "  | A-sh\n");
    for (long m = 0; m < Mb; m++) {
      fprintf(f, "im%-4ld", m);
      for (long n = 0; n < Nb; n++) fprintf(f, "%*d", w, own[((size_t)l * Mb + m) * Nb + n]);
      fprintf(f, "  | %u\n", fp.sh_a[(size_t)l * Mb + m]);
    }
    fprintf(f, "%-6s", "B-sh");
    for (long n = 0; n < Nb; n++) fprintf(f, "%*u", w, fp.sh_b[(size_t)l * Nb + n]);
    fprintf(f, "\n");
    for (int is_a = 1; is_a >= 0; is_a--) {
      auto rd = panel_readers(own, l, Mb, Nb, is_a);
      fprintf(f, "%s decomposition of K-layer %ld (panels read by the same cores):\n", is_a ? "A" : "B", l);
      for (auto &b : reader_bands(rd)) {
        char rng[32];
        snprintf(rng, sizeof rng, "%s %ld-%ld", is_a ? "im" : "in", b.first, b.second - 1);
        fprintf(f, "  %-14s -> %2zu cores {%s}\n", rng, rd[b.first].size(), core_ranges(rd[b.first]).c_str());
      }
    }
  }
}

static bool write_svg_map(const std::string &path, const Footprint &fp, const Problem &p, long threads, long kbf) {
  FILE *f = fopen(path.c_str(), "w");
  if (!f) return false;
  const std::vector<int> own = owner_grid(fp);
  const long Mb = fp.Mb, Nb = fp.Nb, L = fp.K_layers, Kb = p.K / p.bk;
  const long Kb_pl = (Kb + L - 1) / L, Kb_last = Kb - (L - 1) * Kb_pl, brcount = (Kb_pl + kbf - 1) / kbf;
  const double s = std::max(1.0, std::min(24.0, 700.0 / (double)std::max(Mb, Nb)));
  const double kdim = std::max(90.0, std::min(Kb_pl * s, 0.35 * std::max(Mb, Nb) * s));
  const double gap = 34, pad = 6, title_h = 4, top = 76;
  const double pw = kdim + pad + Nb * s + gap, ph = title_h + kdim + pad + Mb * s + gap;
  const long per_row = std::max(1L, std::min<long>(L, (long)(1900.0 / pw)));
  const double W = std::max(per_row * pw + gap, 1000.0), H = top + ((L + per_row - 1) / per_row) * ph;
  auto core_fill = [](long t) {
    char b[48];
    snprintf(b, sizeof b, "hsl(%.1f,65%%,%d%%)", std::fmod(t * 137.508, 360.0), (t % 2) ? 58 : 72);
    return std::string(b);
  };
  auto team_fill = [](const std::vector<int> &v, long band) {
    uint64_t h = 1469598103934665603ULL;
    for (int x : v) h = (h ^ (uint64_t)x) * 1099511628211ULL;
    char b[48];
    snprintf(b, sizeof b, "hsl(%.1f,45%%,%d%%)", (double)(h % 3600) / 10.0, (band % 2) ? 70 : 82);
    return std::string(b);
  };
  auto text = [&](double x, double y, double fs, const char *anchor, const std::string &t, double rot = 0) {
    fprintf(f, "<text x=\"%.2f\" y=\"%.2f\" font-size=\"%.1f\" text-anchor=\"%s\" dominant-baseline=\"central\"", x, y,
            fs, anchor);
    if (rot != 0) fprintf(f, " transform=\"rotate(%.0f %.2f %.2f)\"", rot, x, y);
    fprintf(f, ">%s</text>\n", t.c_str());
  };

  fprintf(f, "<svg xmlns=\"http://www.w3.org/2000/svg\" width=\"%.0f\" height=\"%.0f\" shape-rendering=\"crispEdges\" "
             "font-family=\"monospace\" font-size=\"11\">\n<rect width=\"100%%\" height=\"100%%\" fill=\"white\"/>\n", W, H);
  fprintf(f, "<text x=\"%.0f\" y=\"20\" font-size=\"14\">M=%ld N=%ld K=%ld bm=%ld bn=%ld bk=%ld  T=%ld  K_layers=%ld "
             "m_step=%ld n_step=%ld kbf=%ld  patches=%s grid=%s</text>\n", gap, p.M, p.N, p.K, p.bm, p.bn, p.bk, threads,
          L, fp.m_step, fp.n_step, kbf, fp.patches.c_str(), fp.grid.c_str());
  fprintf(f, "<text x=\"%.0f\" y=\"40\">C: core that owns each tile.  A (left, rows im) / B (top, cols in): panels "
             "colored by the set of cores reading them, label = #sharers {cores}.</text>\n", gap);
  fprintf(f, "<text x=\"%.0f\" y=\"56\">Dashed: kbf K-rounds (%ld k-blocks each).  Hover a panel for its full core list.</text>\n",
          gap, brcount);

  for (long l = 0; l < L; l++) {
    const double ox = gap + (l % per_row) * pw, oy = top + (l / per_row) * ph + title_h;
    const double ax = ox, ay = oy + kdim + pad, bx = ox + kdim + pad, by = oy, cx = bx, cy = ay;
    const long Kb_l = (l == L - 1) ? Kb_last : Kb_pl;
    const double klen = kdim * (double)Kb_l / (double)Kb_pl;
    auto at = [&](long m, long n) { return own[((size_t)l * Mb + m) * Nb + n]; };
    fprintf(f, "<g>\n");
    text(ox + kdim / 2, oy + kdim / 2 - 16, 13, "middle", "K-layer " + std::to_string(l));
    text(ox + kdim / 2, oy + kdim / 2 + 2, 10, "middle",
         "k-blocks " + std::to_string(l * Kb_pl) + "-" + std::to_string(l * Kb_pl + Kb_l - 1));
    text(ox + kdim / 2, oy + kdim / 2 + 16, 10, "middle",
         std::to_string((Kb_l + brcount - 1) / brcount) + " K-round(s)");

    // ---- C: one rect per horizontal run of equal owner keeps the file small ----
    for (long m = 0; m < Mb; m++) {
      for (long n0 = 0; n0 < Nb;) {
        long n1 = n0 + 1;
        while (n1 < Nb && at(m, n1) == at(m, n0)) n1++;
        fprintf(f, "<rect x=\"%.2f\" y=\"%.2f\" width=\"%.2f\" height=\"%.2f\" fill=\"%s\"/>\n", cx + n0 * s,
                cy + m * s, (n1 - n0) * s, s, core_fill(at(m, n0)).c_str());
        n0 = n1;
      }
    }
    std::string d;
    char seg[96];
    for (long m = 0; m < Mb; m++) {
      for (long n = 0; n < Nb; n++) {
        if (n + 1 < Nb && at(m, n) != at(m, n + 1)) {
          snprintf(seg, sizeof seg, "M%.2f %.2fv%.2f", cx + (n + 1) * s, cy + m * s, s);
          d += seg;
        }
        if (m + 1 < Mb && at(m, n) != at(m + 1, n)) {
          snprintf(seg, sizeof seg, "M%.2f %.2fh%.2f", cx + n * s, cy + (m + 1) * s, s);
          d += seg;
        }
      }
    }
    fprintf(f, "<path d=\"%s\" stroke=\"black\" stroke-width=\"1\" fill=\"none\"/>\n", d.c_str());
    fprintf(f, "<rect x=\"%.2f\" y=\"%.2f\" width=\"%.2f\" height=\"%.2f\" stroke=\"black\" stroke-width=\"1.5\" fill=\"none\"/>\n",
            cx, cy, Nb * s, Mb * s);
    std::map<int, std::array<double, 3>> cen;
    for (long m = 0; m < Mb; m++)
      for (long n = 0; n < Nb; n++) {
        auto &c = cen[at(m, n)];
        c[0] += m; c[1] += n; c[2] += 1;
      }
    for (auto &kv : cen) {
      if (kv.second[2] * s * s < 300) continue;
      text(cx + (kv.second[1] / kv.second[2] + 0.5) * s, cy + (kv.second[0] / kv.second[2] + 0.5) * s, 11, "middle",
           std::to_string(kv.first));
    }

    // ---- A (rows im x this layer's K slab) and B (this layer's K slab x cols in) ----
    for (int is_a = 1; is_a >= 0; is_a--) {
      auto rd = panel_readers(own, l, Mb, Nb, is_a);
      auto bands = reader_bands(rd);
      for (size_t bi = 0; bi < bands.size(); bi++) {
        const long lo = bands[bi].first, hi = bands[bi].second;
        const auto &cores = rd[lo];
        const double x = is_a ? ax : bx + lo * s, y = is_a ? ay + lo * s : by + (kdim - klen);
        const double w = is_a ? klen : (hi - lo) * s, h = is_a ? (hi - lo) * s : klen;
        fprintf(f, "<rect x=\"%.2f\" y=\"%.2f\" width=\"%.2f\" height=\"%.2f\" fill=\"%s\" stroke=\"black\" "
                   "stroke-width=\"0.8\"><title>%s panels %s %ld-%ld, K-layer %ld: read by %zu cores {%s}</title></rect>\n",
                x, y, w, h, team_fill(cores, (long)bi).c_str(), is_a ? "A" : "B", is_a ? "im" : "in", lo, hi - 1, l,
                cores.size(), core_ranges(cores, 400).c_str());
        const double thick = is_a ? h : w;
        if (thick < 9) continue;
        const double fs = std::min(11.0, thick - 2);
        const size_t fit = (size_t)std::max(0.0, (is_a ? w : h) / (0.62 * fs)) - 1;
        std::string lab = std::to_string(cores.size()) + "x {" + core_ranges(cores, 400) + "}";
        if (lab.size() > fit) lab = std::to_string(cores.size()) + "x";
        if (is_a) text(x + w / 2, y + h / 2, fs, "middle", lab);
        else text(x + w / 2, y + h / 2, fs, "middle", lab, -90);
      }
      // K-round boundaries inside this layer's slab
      for (long ik = brcount; ik < Kb_l; ik += brcount) {
        double frac = (double)ik / (double)Kb_pl * kdim;
        if (is_a)
          fprintf(f, "<line x1=\"%.2f\" y1=\"%.2f\" x2=\"%.2f\" y2=\"%.2f\" stroke=\"#444\" stroke-dasharray=\"4 3\"/>\n",
                  ax + frac, ay, ax + frac, ay + Mb * s);
        else
          fprintf(f, "<line x1=\"%.2f\" y1=\"%.2f\" x2=\"%.2f\" y2=\"%.2f\" stroke=\"#444\" stroke-dasharray=\"4 3\"/>\n",
                  bx, by + (kdim - klen) + frac, bx + Nb * s, by + (kdim - klen) + frac);
      }
    }
    text(ax + klen / 2, ay + Mb * s + 12, 11, "middle", "A");
    text(bx - 10, by + kdim / 2, 11, "middle", "B");
    fprintf(f, "</g>\n");
  }
  fprintf(f, "</svg>\n");
  return fclose(f) == 0;
}

int main(int argc, char **argv) {
  if (argc < 4) { usage(argv[0]); return 1; }
  Problem p{atol(argv[1]), atol(argv[2]), atol(argv[3]), 32, 32, 32};
  Platform hw;
  SearchSpace sp;
  long top = 5;
  bool eval_one = false;
  bool dump_cores = false;
  std::string dump_map;
  Config one;
  unsigned jobs = std::max(1u, std::thread::hardware_concurrency());

  try {
    for (int i = 4; i < argc; i++) {
      std::string a = argv[i];
      if (a == "--dump-cores") { dump_cores = true; continue; }
      if (a == "--overlap-c") { hw.overlap_c_writes = true; continue; }
      if (a == "--pf-smooth") { hw.pf_smooth = true; continue; }
      if (i + 1 >= argc) throw std::runtime_error("missing value for " + a);
      std::string v = argv[++i];
      if (a == "--bm") p.bm = std::stol(v);
      else if (a == "--bn") p.bn = std::stol(v);
      else if (a == "--bk") p.bk = std::stol(v);
      else if (a == "--threads") hw.threads = std::stol(v);
      else if (a == "--bw") {
        if (!hw.bw_by_sharers.empty()) throw std::runtime_error("use only one of --bw / --bw-flat");
        hw.bw_by_sharers = parse_bw(v);
      }
      else if (a == "--bw-flat") {
        if (!hw.bw_by_sharers.empty()) throw std::runtime_error("use only one of --bw / --bw-flat");
        double bw = std::stod(v);
        if (bw <= 0) throw std::runtime_error("bad --bw-flat: " + v);
        hw.bw_by_sharers = {{1, bw}};  // bw() clamps, so one point is flat for all sharer counts
      }
      else if (a == "--c-bw") hw.c_bw = std::stod(v);
      else if (a == "--macs") hw.macs_per_cycle = std::stod(v);
      else if (a == "--eff") hw.efficiency = std::stod(v) / 100.0;
      else if (a == "--eff-br") {
        hw.eff_by_brcount = parse_bw(v, "--eff-br");
        for (auto &kv : hw.eff_by_brcount) kv.second /= 100.0;
      }
      else if (a == "--freq") hw.freq_ghz = std::stod(v);
      else if (a == "--l2-kb") hw.l2_bytes = std::stod(v) * 1024.0;
      else if (a == "--kl-max") sp.kl_max = std::stol(v);
      else if (a == "--step-max") sp.step_max = std::stol(v);
      else if (a == "--kbf-max") sp.kbf_max = std::stol(v);
      else if (a == "--sched") {
        if (v == "balanced") hw.sched = OmpSched::Balanced;
        else if (v == "greedy") hw.sched = OmpSched::Greedy;
        else throw std::runtime_error("bad --sched: " + v);
      }
      else if (a == "--in-bytes") p.in_bytes = std::stod(v);
      else if (a == "--out-bytes") p.out_bytes = std::stod(v);
      else if (a == "--top") top = std::stol(v);
      else if (a == "--eval") {
        if (sscanf(v.c_str(), "%ld,%ld,%ld,%ld", &one.K_layers, &one.m_step, &one.n_step, &one.kbf) != 4)
          throw std::runtime_error("bad --eval: " + v);
        eval_one = true;
      }
      else if (a == "--jobs") jobs = (unsigned)std::max(1L, std::stol(v));
      else if (a == "--dump-map") dump_map = v;
      else throw std::runtime_error("unknown option " + a);
    }
  } catch (const std::exception &e) {
    fprintf(stderr, "Error: %s\n", e.what());
    usage(argv[0]);
    return 1;
  }

  if (hw.bw_by_sharers.empty()) { fprintf(stderr, "Error: --bw or --bw-flat is required\n"); return 1; }
  if (p.M <= 0 || p.N <= 0 || p.K <= 0 || p.bm <= 0 || p.bn <= 0 || p.bk <= 0 ||
      p.M % p.bm || p.N % p.bn || p.K % p.bk) {
    fprintf(stderr, "Error: M,N,K must be positive multiples of bm,bn,bk\n");
    return 1;
  }
  if (hw.threads < 1 || hw.macs_per_cycle <= 0 || hw.efficiency <= 0) {
    fprintf(stderr, "Error: invalid threads/macs/eff\n");
    return 1;
  }
  const long Mb = p.M / p.bm, Nb = p.N / p.bn, Kb = p.K / p.bk;
  if (Mb >= (1L << 20) || Nb >= (1L << 20) || std::max(sp.kl_max, eval_one ? one.K_layers : 0L) >= (1L << 12)) {
    fprintf(stderr, "Error: problem too large for the model's cache keys\n");
    return 1;
  }

  // ---- Enumerate decompositions: one SFC footprint per (K_layers, m_step, n_step) ----
  std::vector<std::pair<long, long>> steps;
  if (eval_one) {
    if (one.m_step < 1 || one.n_step < 1 || Mb % one.m_step || Nb % one.n_step) {
      fprintf(stderr, "Error: m_step/n_step must divide Mb/Nb\n");
      return 1;
    }
    steps.push_back({one.m_step, one.n_step});
  } else {
    for (long ms = 1; ms <= sp.step_max; ms++)
      for (long ns = 1; ns <= sp.step_max; ns++)
        if (Mb % ms == 0 && Nb % ns == 0) steps.push_back({ms, ns});
  }

  std::map<std::pair<long, long>, std::vector<uint32_t>> maps;
  for (auto &st : steps) maps[st] = build_sfc_map(Mb, Nb, st.first, st.second);

  struct Job { long c, ms, ns; std::vector<long> kbfs; };
  std::vector<Job> work;
  for (auto &st : steps) {
    for (long c = eval_one ? one.K_layers : 1; c <= (eval_one ? one.K_layers : std::min(sp.kl_max, Kb)); c++) {
      long kbpl = (Kb + c - 1) / c;
      if (c < 1 || Kb - (c - 1) * kbpl <= 0) continue;
      Job jb{c, st.first, st.second, {}};
      if (eval_one) {
        jb.kbfs.push_back(one.kbf);
      } else {
        std::set<long> seen_brcount;  // kbf values with equal brcount are the same schedule
        for (long kbf = 1; kbf <= std::min(sp.kbf_max, kbpl); kbf++)
          if (seen_brcount.insert((kbpl + kbf - 1) / kbf).second) jb.kbfs.push_back(kbf);
      }
      work.push_back(jb);
    }
  }

  std::vector<std::vector<Result>> job_res(work.size());
  std::atomic<size_t> next{0};
  std::vector<std::thread> pool;
  for (unsigned j = 0; j < std::min<size_t>(jobs, work.size()); j++) {
    pool.emplace_back([&]() {
      for (size_t i; (i = next.fetch_add(1)) < work.size();) {
        const Job &jb = work[i];
        Footprint fp = build_footprint(Mb, Nb, jb.c, jb.ms, jb.ns, hw, maps.at({jb.ms, jb.ns}));
        for (long kbf : jb.kbfs) job_res[i].push_back(evaluate(p, hw, fp, kbf));
      }
    });
  }
  for (auto &th : pool) th.join();

  if (eval_one && dump_cores && !work.empty()) {
    Footprint fp = build_footprint(Mb, Nb, one.K_layers, one.m_step, one.n_step, hw,
                                   maps.at({one.m_step, one.n_step}));
    printf("Per-core partitions for K_layers=%ld m_step=%ld n_step=%ld (tile units):\n",
           one.K_layers, one.m_step, one.n_step);
    printf("%-5s %-15s %-7s %-17s %-17s %-6s %-6s %-11s %-11s %-5s %s\n", "core", "SFC tasks", "layer",
           "C rows(A panels)", "C cols(B panels)", "#A", "#B", "A-sharers", "B-sharers", "rect", "red tiles");
    for (long t = 0; t < (long)fp.cores.size(); t++) {
      const CorePartition &cp = fp.cores[t];
      if (cp.tasks.empty()) { printf("%-5ld idle\n", t); continue; }
      uint32_t amin = ~0u, amax = 0, bmin = ~0u, bmax = 0;
      for (const Task &tk : cp.tasks) {
        amin = std::min(amin, tk.sh_a); amax = std::max(amax, tk.sh_a);
        bmin = std::min(bmin, tk.sh_b); bmax = std::max(bmax, tk.sh_b);
      }
      char rng[32], lay[16], rows[32], cols[32], ash[16], bsh[16], red[32];
      snprintf(rng, sizeof rng, "[%ld,%ld)", cp.lo, cp.hi);
      snprintf(lay, sizeof lay, cp.layer_lo == cp.layer_hi ? "%ld" : "%ld-%ld", cp.layer_lo, cp.layer_hi);
      snprintf(rows, sizeof rows, "%ld..%ld", cp.im_lo, cp.im_hi);
      snprintf(cols, sizeof cols, "%ld..%ld", cp.in_lo, cp.in_hi);
      snprintf(ash, sizeof ash, "%u-%u", amin, amax);
      snprintf(bsh, sizeof bsh, "%u-%u", bmin, bmax);
      snprintf(red, sizeof red, "[%ld,%ld)", cp.red_lo, cp.red_hi);
      printf("%-5ld %-15s %-7s %-17s %-17s %-6ld %-6ld %-11s %-11s %-5s %s\n", t, rng, lay, rows, cols,
             cp.n_a, cp.n_b, ash, bsh, cp.rect ? "yes" : "no", one.K_layers > 1 ? red : "-");
    }
    printf("\n");
  }

  std::vector<Result> res;
  for (auto &v : job_res) res.insert(res.end(), v.begin(), v.end());
  res.erase(std::remove_if(res.begin(), res.end(), [](const Result &r) { return !r.valid; }), res.end());
  if (eval_one && res.empty()) {
    fprintf(stderr, "Error: configuration is invalid for this problem\n");
    return 1;
  }
  std::sort(res.begin(), res.end(), [](const Result &a, const Result &b) {
    // Whole cycles, so summation-order noise cannot break ties between equivalent configs.
    long long ca = std::llround(a.total_cycles), cb = std::llround(b.total_cycles);
    if (ca != cb) return ca < cb;
    if (a.cfg.K_layers != b.cfg.K_layers) return a.cfg.K_layers < b.cfg.K_layers;
    if (a.cfg.m_step * a.cfg.n_step != b.cfg.m_step * b.cfg.n_step)
      return a.cfg.m_step * a.cfg.n_step < b.cfg.m_step * b.cfg.n_step;
    return a.cfg.kbf < b.cfg.kbf;
  });
  const size_t n_evaluated = res.size();

  // Different (m_step, n_step) may yield the very same decomposition and time: report them once.
  std::vector<Result> uniq;
  std::vector<std::string> equiv;
  for (const Result &r : res) {
    char st[16];
    snprintf(st, sizeof st, "%ld,%ld", r.cfg.m_step, r.cfg.n_step);
    if (!uniq.empty()) {
      const Result &u = uniq.back();
      if (u.cfg.K_layers == r.cfg.K_layers && u.brcount == r.brcount && u.patches == r.patches &&
          u.a_sh_min == r.a_sh_min && u.a_sh_max == r.a_sh_max && u.b_sh_min == r.b_sh_min &&
          u.b_sh_max == r.b_sh_max && std::fabs(u.a_sh_avg - r.a_sh_avg) < 1e-9 &&
          std::fabs(u.b_sh_avg - r.b_sh_avg) < 1e-9 &&
          std::fabs(u.total_cycles - r.total_cycles) <= 1e-9 * u.total_cycles) {
        equiv.back() += std::string(" ") + st;
        continue;
      }
    }
    uniq.push_back(r);
    equiv.push_back("");
  }
  res.swap(uniq);

  // ---- Report ----
  printf("SFC-CA GEMM sharer-aware perf model\n");
  printf("Problem : M=%ld N=%ld K=%ld  bm=%ld bn=%ld bk=%ld  (Mb=%ld Nb=%ld Kb=%ld)  in=%gB out=%gB\n",
         p.M, p.N, p.K, p.bm, p.bn, p.bk, Mb, Nb, Kb, p.in_bytes, p.out_bytes);
  printf("Platform: T=%ld  MACs/cyc/core=%g  eff=%.1f%%  L2=%s  C-BW=%g B/cyc%s  sched=%s",
         hw.threads, hw.macs_per_cycle, 100.0 * hw.efficiency,
         hw.l2_bytes > 0 ? (std::to_string((long)(hw.l2_bytes / 1024)) + "KiB").c_str() : "inf",
         hw.cbw(), hw.c_bw > 0 ? " (--c-bw)" : " (=BW(1 sharer))",
         hw.sched == OmpSched::Balanced ? "balanced" : "greedy");
  if (hw.freq_ghz > 0) printf("  f=%gGHz", hw.freq_ghz);
  if (hw.overlap_c_writes) printf("  C-writes=overlapped");
  if (hw.pf_smooth) printf("  A/B-prefetch=ideal");
  printf("\nBW(sharers) [B/cyc/core]:");
  for (auto &kv : hw.bw_by_sharers) printf(" %ld:%g", kv.first, kv.second);
  if (!hw.eff_by_brcount.empty()) {
    printf("\nEff(brcount) [%%] (overrides eff):");
    for (auto &kv : hw.eff_by_brcount) printf(" %ld:%g", kv.first, 100.0 * kv.second);
  }
  printf("\nEvaluated %zu decompositions (K_layers<=%ld, m_step,n_step<=%ld, kbf<=%ld), %zu distinct\n\n",
         n_evaluated, sp.kl_max, sp.step_max, sp.kbf_max, res.size());

  if (res.empty()) { printf("No valid decomposition.\n"); return 1; }

  const double flops = 2.0 * (double)p.M * p.N * p.K;
  printf("%-4s %-3s %-5s %-8s %-6s %-13s %-11s %-13s %-13s %-13s %-12s %-12s %-10s %-7s %-17s\n",
         "rank", "KL", "ms,ns", "kbf(br)", "T'", "patch", "grid", "A-sh avg/rng", "B-sh avg/rng",
         "BW A/B B/cyc", "GEMM cyc", "Red cyc", "flop/cyc", "%peak",
         hw.freq_ghz > 0 ? "TFLOP/s" : "");
  for (long k = 0; k < std::min<long>(top, (long)res.size()); k++) {
    const Result &r = res[k];
    char ash[32], bsh[32], bws[32], kbf[32], stp[16];
    snprintf(stp, sizeof stp, "%ld,%ld", r.cfg.m_step, r.cfg.n_step);
    snprintf(kbf, sizeof kbf, "%ld(%ld)", r.cfg.kbf, r.brcount);
    snprintf(ash, sizeof ash, "%.1f/%ld-%ld", r.a_sh_avg, r.a_sh_min, r.a_sh_max);
    snprintf(bsh, sizeof bsh, "%.1f/%ld-%ld", r.b_sh_avg, r.b_sh_min, r.b_sh_max);
    snprintf(bws, sizeof bws, "%.2f/%.2f", r.crit_a_bw, r.crit_b_bw);
    printf("%-4ld %-3ld %-5s %-8s %-6.4g %-13s %-11s %-13s %-13s %-13s %-12.4g %-12.4g %-10.1f %-7.1f",
           k + 1, r.cfg.K_layers, stp, kbf, r.threads_per_layer, r.patches.c_str(), r.grid.c_str(), ash, bsh, bws,
           r.gemm_cycles, r.red_cycles, r.flops_per_cycle, r.pct_peak);
    if (hw.freq_ghz > 0) printf(" %-8.2f", flops / (r.total_cycles / (hw.freq_ghz * 1e9)) / 1e12);
    if (!equiv[k].empty()) printf("  (same as ms,ns:%s)", equiv[k].c_str());
    printf("\n");
  }

  printf("\nCritical-thread breakdown (cycles):\n");
  for (long k = 0; k < std::min<long>(top, (long)res.size()); k++) {
    const Result &r = res[k];
    double tot = r.total_cycles;
    printf("  #%ld thr %-4ld compute %.4g (%.0f%%)  mem-stall %.4g (%.0f%%)  C-io %.4g (%.0f%%)  reduction %.4g (%.0f%%)"
           "  A %.3g MB  B %.3g MB  tasks/thr %ld-%ld\n",
           k + 1, r.crit_thread, r.crit_compute, 100 * r.crit_compute / tot, r.crit_stall,
           100 * r.crit_stall / tot, r.crit_c, 100 * r.crit_c / tot, r.red_cycles,
           100 * r.red_cycles / tot, r.crit_a_bytes / 1e6, r.crit_b_bytes / 1e6, r.tasks_min, r.tasks_max);
  }

  printf("\nRun commands (sfc_ca_gemm M N K bm bn bk kbf K_layers n_layers n_iters check dtype m_step n_step):\n");
  for (long k = 0; k < std::min<long>(top, (long)res.size()); k++) {
    const Result &r = res[k];
    printf("  #%ld ./sfc_ca_gemm %ld %ld %ld %ld %ld %ld %ld %ld -1 10 0 BF16 %ld %ld\n", k + 1, p.M, p.N, p.K,
           p.bm, p.bn, p.bk, r.cfg.kbf, r.cfg.K_layers, r.cfg.m_step, r.cfg.n_step);
  }
  printf("\npatch: per-core C partition shapes (A panels x B panels), most common first; '*' = some core\n"
         "       does not own one contiguous rectangle in a single layer\n");
  printf("grid : K_layers x core-rows x core-cols if all patches are equal rectangles; A-sh/B-sh: cores\n"
         "       sharing an A/B panel (avg/min-max); BW A/B: effective B/cyc of the critical core\n");

  if (!dump_map.empty()) {
    const Config mc = eval_one ? one : res[0].cfg;
    Footprint fp = build_footprint(Mb, Nb, mc.K_layers, mc.m_step, mc.n_step, hw, maps.at({mc.m_step, mc.n_step}));
    const bool svg = dump_map.size() > 4 && dump_map.compare(dump_map.size() - 4, 4, ".svg") == 0;
    printf("\nCore-ownership map of K_layers=%ld m_step=%ld n_step=%ld", mc.K_layers, mc.m_step, mc.n_step);
    if (dump_map == "-") {
      printf(":\n");
      write_ascii_map(stdout, fp);
    } else if (svg ? write_svg_map(dump_map, fp, p, hw.threads, mc.kbf) : [&]() {
                 FILE *f = fopen(dump_map.c_str(), "w");
                 if (!f) return false;
                 write_ascii_map(f, fp);
                 return fclose(f) == 0;
               }()) {
      printf(" written to %s (%s)\n", dump_map.c_str(), svg ? "SVG" : "ASCII");
    } else {
      fprintf(stderr, "\nError: could not write %s\n", dump_map.c_str());
      return 1;
    }
  }
  return 0;
}
