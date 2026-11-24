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

/*
 * Sharer-aware analytical performance model for SFC-CA GEMM.
 *
 * Extends the BRGEMM-granular roofline of the paper (Sec. III-B,
 * t_i = max(G*gamma, beta*S_i)) and of roofline_predictor.c as follows:
 *  - The exact thread decomposition is obtained by replaying the production
 *    schedule: the (coarsened) Gilbert SFC map of sfc_ca_gemm.hpp and the
 *    OpenMP static block partition of the Mb*Nb*K_layers task space.
 *  - For every A/B panel the number of threads that read it (its sharers) is
 *    counted; the panel load uses the per-core bandwidth beta(sharers).
 *  - A/B/C reuse is tracked by a per-core LRU cache (or infinite cache).
 *  - Compute rate = MACs/cycle/core * efficiency.
 */

#ifndef SFC_CA_PERF_MODEL_HPP
#define SFC_CA_PERF_MODEL_HPP

#include <cstdint>
#include <cstdlib>
#include <cmath>
#include <algorithm>
#include <limits>
#include <list>
#include <map>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

#include "sfc_utils.h"

namespace sfc_model {

enum class OmpSched { Balanced /* libgomp */, Greedy /* libomp/icx */ };

struct Platform {
  long threads = 64;
  // Per-core B/cycle for loading a panel concurrently read by s cores.
  std::map<long, double> bw_by_sharers;
  double c_bw = 0.0;                 // per-core B/cycle for C traffic, <=0 -> bw(1)
  double macs_per_cycle = 1024.0;    // per core (BF16)
  double efficiency = 1.0;           // achieved fraction of MAC peak in BRGEMM
  double freq_ghz = 0.0;             // only for reporting
  double l2_bytes = 2.0 * 1024 * 1024; // <=0 -> infinite fast memory
  OmpSched sched = OmpSched::Balanced;

  // Log2-linear interpolation between given sharer counts, clamped at the ends.
  double bw(long s) const {
    if (s < 1) s = 1;
    auto hi = bw_by_sharers.lower_bound(s);
    if (hi == bw_by_sharers.end()) return std::prev(hi)->second;
    if (hi->first == s || hi == bw_by_sharers.begin()) return hi->second;
    auto lo = std::prev(hi);
    double f = (std::log2((double)s) - std::log2((double)lo->first)) /
               (std::log2((double)hi->first) - std::log2((double)lo->first));
    return lo->second + f * (hi->second - lo->second);
  }
  double cbw() const { return (c_bw > 0.0) ? c_bw : bw(1); }
  double peak_flops_per_cycle() const { return 2.0 * macs_per_cycle * (double)threads; }
};

struct Problem {
  long M, N, K, bm, bn, bk;
  double in_bytes = 2.0;   // BF16 A/B
  double out_bytes = 2.0;  // BF16 C
};

struct Config {
  long K_layers = 1, m_step = 1, n_step = 1, kbf = 1;
};

struct Result {
  Config cfg;
  bool valid = false;
  std::string reason;
  long brcount = 0, k_rounds = 0;
  double threads_per_layer = 0;

  double total_cycles = 0, gemm_cycles = 0, red_cycles = 0;
  double flops_per_cycle = 0, pct_peak = 0;

  // Topology of the decomposition (copied from the Footprint)
  std::string patches, grid;
  double a_sh_avg = 0, b_sh_avg = 0;
  long a_sh_min = 0, a_sh_max = 0, b_sh_min = 0, b_sh_max = 0;
  long tasks_min = 0, tasks_max = 0;

  // Critical (slowest GEMM-phase) thread breakdown
  long crit_thread = -1;
  double crit_compute = 0, crit_stall = 0, crit_c = 0;
  double crit_a_bytes = 0, crit_b_bytes = 0, crit_a_bw = 0, crit_b_bw = 0;
};

// Fine SFC index -> (im, in), identical to sfc_ca_gemm_fill_sfc_index_map().
inline std::vector<uint32_t> build_sfc_map(long Mb, long Nb, long m_step, long n_step) {
  long Mbc = Mb / m_step, Nbc = Nb / n_step;
  std::vector<uint32_t> map(2 * (size_t)Mb * (size_t)Nb, 0);
  for (long i = 0; i < Mbc * Nbc; i++) {
    int m_id, n_id;
    gilbert_d2xy(&m_id, &n_id, (int)i, (int)Mbc, (int)Nbc);
    for (long ms = 0; ms < m_step; ms++) {
      for (long ns = 0; ns < n_step; ns++) {
        size_t fi = (size_t)(i * m_step * n_step + ms * n_step + ns);
        map[2 * fi + 0] = (uint32_t)(m_id * m_step + ms);
        map[2 * fi + 1] = (uint32_t)(n_id * n_step + ns);
      }
    }
  }
  return map;
}

// Iteration range [lo, hi) of thread t under "#pragma omp for schedule(static)".
inline void static_chunk(long n, long T, long t, OmpSched s, long &lo, long &hi) {
  if (s == OmpSched::Balanced) {
    long q = n / T, r = n % T;
    lo = (t < r) ? t * (q + 1) : r * (q + 1) + (t - r) * q;
    hi = lo + ((t < r) ? q + 1 : q);
  } else if (n < T) {
    lo = std::min(t, n);
    hi = (t < n) ? t + 1 : n;
  } else {
    long ch = (n + T - 1) / T;
    lo = std::min(n, t * ch);
    hi = std::min(n, lo + ch);
  }
}

class LruCache {
 public:
  explicit LruCache(double cap) : cap_(cap > 0 ? cap : std::numeric_limits<double>::infinity()) {}
  void clear() { lst_.clear(); idx_.clear(); used_ = 0; }
  // Returns true on hit; on miss inserts the object as MRU and evicts LRU objects.
  bool access(uint64_t key, double sz) {
    auto it = idx_.find(key);
    if (it != idx_.end()) {
      lst_.splice(lst_.begin(), lst_, it->second);
      return true;
    }
    lst_.emplace_front(key, sz);
    idx_[key] = lst_.begin();
    used_ += sz;
    while (used_ > cap_ && lst_.size() > 1) {
      used_ -= lst_.back().second;
      idx_.erase(lst_.back().first);
      lst_.pop_back();
    }
    return false;
  }
 private:
  double cap_, used_ = 0;
  std::list<std::pair<uint64_t, double>> lst_;
  std::unordered_map<uint64_t, std::list<std::pair<uint64_t, double>>::iterator> idx_;
};

// Key layout: type(2) | round(10) | layer(12) | im(20) | in(20)
inline uint64_t cache_key(uint64_t type, uint64_t round, uint64_t layer, uint64_t im, uint64_t in) {
  return (type << 62) | (round << 52) | (layer << 40) | (im << 20) | in;
}

// One BRGEMM task of a core: C tile (layer, im, in) and the sharer counts of its A/B panels.
struct Task {
  uint32_t layer, im, in;
  uint32_t sh_a, sh_b;
};

// Exact A/B/C partition owned by one core.
struct CorePartition {
  long lo = 0, hi = 0;           // SFC task range [lo, hi) of the GEMM loop
  std::vector<Task> tasks;       // C partition, in execution order
  long n_a = 0, n_b = 0;         // distinct A panels (layer, im) / B panels (layer, in)
  long layer_lo = 0, layer_hi = -1;
  long im_lo = 0, im_hi = -1, in_lo = 0, in_hi = -1;  // bounding box of the C tiles
  bool rect = false;             // owns a full contiguous n_a x n_b rectangle within one layer
  long red_lo = 0, red_hi = 0;   // tile range [red_lo, red_hi) of the C reduction loop
};

// SFC footprint of a (K_layers, m_step, n_step) decomposition; independent of kbf.
struct Footprint {
  long K_layers = 1, m_step = 1, n_step = 1, Mb = 0, Nb = 0;
  std::vector<CorePartition> cores;
  std::vector<uint32_t> sh_a, sh_b;  // sharers of A panel (layer*Mb+im) / B panel (layer*Nb+in)
  std::map<std::pair<long, long>, long> shapes;  // (rows, cols) -> number of cores
  bool all_rect = true;
  long tasks_min = 0, tasks_max = 0;
  std::string patches, grid;
};

inline Footprint build_footprint(long Mb, long Nb, long K_layers, long m_step, long n_step,
                                 const Platform &hw, const std::vector<uint32_t> &sfc_map) {
  Footprint fp;
  fp.K_layers = K_layers; fp.m_step = m_step; fp.n_step = n_step; fp.Mb = Mb; fp.Nb = Nb;
  const long T = hw.threads, tiles = Mb * Nb, n_tasks = tiles * K_layers;
  fp.cores.resize(T);
  fp.sh_a.assign((size_t)K_layers * Mb, 0);
  fp.sh_b.assign((size_t)K_layers * Nb, 0);
  std::vector<int> lastA(fp.sh_a.size(), -1), lastB(fp.sh_b.size(), -1);
  fp.tasks_min = std::numeric_limits<long>::max();

  for (long t = 0; t < T; t++) {
    CorePartition &cp = fp.cores[t];
    static_chunk(n_tasks, T, t, hw.sched, cp.lo, cp.hi);
    static_chunk(tiles, T, t, hw.sched, cp.red_lo, cp.red_hi);
    const long nt = cp.hi - cp.lo;
    fp.tasks_min = std::min(fp.tasks_min, nt);
    fp.tasks_max = std::max(fp.tasks_max, nt);
    if (nt <= 0) continue;
    cp.tasks.resize(nt);
    cp.im_lo = cp.in_lo = std::numeric_limits<long>::max();
    cp.layer_lo = cp.lo / tiles;
    cp.layer_hi = (cp.hi - 1) / tiles;
    for (long i = cp.lo; i < cp.hi; i++) {
      long l = i / tiles, f = i % tiles;
      uint32_t im = sfc_map[2 * f], in = sfc_map[2 * f + 1];
      size_t ka = (size_t)(l * Mb + im), kb = (size_t)(l * Nb + in);
      if (lastA[ka] != t) { lastA[ka] = (int)t; fp.sh_a[ka]++; cp.n_a++; }
      if (lastB[kb] != t) { lastB[kb] = (int)t; fp.sh_b[kb]++; cp.n_b++; }
      cp.tasks[i - cp.lo] = Task{(uint32_t)l, im, in, 0, 0};
      cp.im_lo = std::min(cp.im_lo, (long)im); cp.im_hi = std::max(cp.im_hi, (long)im);
      cp.in_lo = std::min(cp.in_lo, (long)in); cp.in_hi = std::max(cp.in_hi, (long)in);
    }
    cp.rect = cp.layer_lo == cp.layer_hi && cp.n_a * cp.n_b == nt &&
              cp.im_hi - cp.im_lo + 1 == cp.n_a && cp.in_hi - cp.in_lo + 1 == cp.n_b;
    fp.all_rect = fp.all_rect && cp.rect;
    fp.shapes[{cp.n_a, cp.n_b}]++;
  }
  // Sharer counts are final only after all cores were visited.
  for (CorePartition &cp : fp.cores) {
    for (Task &tk : cp.tasks) {
      tk.sh_a = fp.sh_a[(size_t)tk.layer * Mb + tk.im];
      tk.sh_b = fp.sh_b[(size_t)tk.layer * Nb + tk.in];
    }
  }

  std::vector<std::pair<long, std::pair<long, long>>> by_count;
  for (auto &kv : fp.shapes) by_count.push_back({kv.second, kv.first});
  std::sort(by_count.rbegin(), by_count.rend());
  for (size_t i = 0; i < by_count.size() && i < 2; i++) {
    fp.patches += (i ? "/" : "") + std::to_string(by_count[i].second.first) + "x" +
                  std::to_string(by_count[i].second.second);
  }
  if (by_count.size() > 2) fp.patches += "/..";
  if (!fp.all_rect) fp.patches += "*";
  if (fp.all_rect && fp.shapes.size() == 1) {
    fp.grid = std::to_string(K_layers) + "x" + std::to_string(Mb / by_count[0].second.first) + "x" +
              std::to_string(Nb / by_count[0].second.second);
  } else {
    fp.grid = fp.all_rect ? "rect-mixed" : "irregular";
  }
  return fp;
}

inline Result evaluate(const Problem &p, const Platform &hw, const Footprint &fp, long kbf) {
  Result r;
  r.cfg = Config{fp.K_layers, fp.m_step, fp.n_step, kbf};
  const long Kb = p.K / p.bk, c = fp.K_layers;

  if (c < 1 || c > Kb) { r.reason = "invalid K_layers"; return r; }
  const long Kb_per_layer = (Kb + c - 1) / c;
  const long Kb_last_layer = Kb - (c - 1) * Kb_per_layer;
  if (Kb_last_layer <= 0) { r.reason = "empty last K-layer"; return r; }
  // kbf > Kb_per_layer is legal in the code and behaves like kbf == Kb_per_layer (brcount = 1).
  if (kbf < 1) { r.reason = "invalid kbf"; return r; }

  const long brcount = (Kb_per_layer + kbf - 1) / kbf;
  const long rounds = (Kb_per_layer + brcount - 1) / brcount;
  if (rounds >= (1L << 10)) { r.reason = "too many K rounds for cache keys"; return r; }
  r.brcount = brcount;
  r.k_rounds = rounds;
  r.threads_per_layer = (double)hw.threads / (double)c;

  r.patches = fp.patches;
  r.grid = fp.grid;
  r.tasks_min = fp.tasks_min;
  r.tasks_max = fp.tasks_max;
  auto sh_stats = [](const std::vector<uint32_t> &v, double &avg, long &mn, long &mx) {
    double s = 0; long n = 0; mn = std::numeric_limits<long>::max(); mx = 0;
    for (uint32_t x : v) if (x > 0) { s += x; n++; mn = std::min(mn, (long)x); mx = std::max(mx, (long)x); }
    avg = n ? s / n : 0;
    if (!n) mn = 0;
  };
  sh_stats(fp.sh_a, r.a_sh_avg, r.a_sh_min, r.a_sh_max);
  sh_stats(fp.sh_b, r.b_sh_avg, r.b_sh_min, r.b_sh_max);

  const double a_unit_bytes = (double)p.bm * p.bk * p.in_bytes;   // per K-block of an A panel
  const double b_unit_bytes = (double)p.bn * p.bk * p.in_bytes;
  const double c_tile_bytes = (double)p.bm * p.bn * p.out_bytes;
  const double unit_compute = (double)p.bm * p.bn * p.bk / (hw.macs_per_cycle * hw.efficiency);
  const double cbw = hw.cbw();

  // Per-core bandwidth lookup by sharer count, cached for the sharer values that occur.
  long max_sh = 1;
  for (uint32_t x : fp.sh_a) max_sh = std::max(max_sh, (long)x);
  for (uint32_t x : fp.sh_b) max_sh = std::max(max_sh, (long)x);
  std::vector<double> bw_of(max_sh + 1);
  for (long s = 0; s <= max_sh; s++) bw_of[s] = hw.bw(s);

  LruCache cache(hw.l2_bytes);
  double gemm_max = -1;
  for (long t = 0; t < (long)fp.cores.size(); t++) {
    const CorePartition &cp = fp.cores[t];
    const long nt = (long)cp.tasks.size();
    if (nt == 0) continue;

    // Replay every K round of the core's own A/B/C partition through its private cache.
    cache.clear();
    double comp = 0, stall = 0, cyc = 0, a_bytes = 0, b_bytes = 0, a_cyc = 0, b_cyc = 0;
    long c_rereads = 0;
    for (long rd = 0; rd < rounds; rd++) {
      const long i_k = rd * brcount;
      for (const Task &tk : cp.tasks) {
        long kb_l = ((long)tk.layer == c - 1) ? Kb_last_layer : Kb_per_layer;
        long brc = std::max(0L, std::min(brcount, kb_l - i_k));
        if (brc == 0) continue;
        bool missA = !cache.access(cache_key(0, rd, tk.layer, tk.im, 0), a_unit_bytes * brc);
        bool missB = !cache.access(cache_key(1, rd, tk.layer, 0, tk.in), b_unit_bytes * brc);
        bool missC = !cache.access(cache_key(2, 0, tk.layer, tk.im, tk.in), c_tile_bytes);
        double ua = missA ? a_unit_bytes / bw_of[tk.sh_a] : 0.0;
        double ub = missB ? b_unit_bytes / bw_of[tk.sh_b] : 0.0;
        double mem = ua + ub;
        cyc += brc * std::max(unit_compute, mem);
        comp += brc * unit_compute;
        stall += brc * std::max(0.0, mem - unit_compute);
        if (missA) { a_bytes += brc * a_unit_bytes; a_cyc += brc * ua; }
        if (missB) { b_bytes += brc * b_unit_bytes; b_cyc += brc * ub; }
        // Round 0 allocates C (zero/BETA_0); a later miss means the partial C was evicted.
        if (rd > 0 && missC) c_rereads++;
      }
    }
    // Final C write-back plus read+write for every C tile evicted between K rounds.
    double c_cyc = ((double)nt + 2.0 * c_rereads) * c_tile_bytes / cbw;
    cyc += c_cyc;
    if (cyc > gemm_max) {
      gemm_max = cyc;
      r.crit_thread = t;
      r.crit_compute = comp;
      r.crit_stall = stall;
      r.crit_c = c_cyc;
      r.crit_a_bytes = a_bytes;
      r.crit_b_bytes = b_bytes;
      r.crit_a_bw = a_cyc > 0 ? a_bytes / a_cyc : 0;
      r.crit_b_bw = b_cyc > 0 ? b_bytes / b_cyc : 0;
    }
  }

  // Reduction of the K_layers C copies: c reads + 1 write per tile.
  double red_max = 0;
  if (c > 1) {
    for (const CorePartition &cp : fp.cores)
      red_max = std::max(red_max, (double)(cp.red_hi - cp.red_lo) * (c + 1) * c_tile_bytes / cbw);
  }

  r.gemm_cycles = gemm_max;
  r.red_cycles = red_max;
  r.total_cycles = gemm_max + red_max;
  r.flops_per_cycle = 2.0 * (double)p.M * p.N * p.K / r.total_cycles;
  r.pct_peak = 100.0 * r.flops_per_cycle / hw.peak_flops_per_cycle();
  r.valid = true;
  return r;
}

struct SearchSpace {
  long kl_max = 16;
  long step_max = 4;
  long kbf_max = 8;
};

}  // namespace sfc_model

#endif  // SFC_CA_PERF_MODEL_HPP
