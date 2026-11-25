/* Per-core read bandwidth vs. number of cores concurrently reading the same panel.
 * Produces the --bw "s:B/cycle,..." list for sfc_ca_perf_model.
 * Groups of s consecutive threads read the same panel; panels rotate through a buffer much
 * larger than the LLC, so a panel is re-read only after the whole buffer was swept.
 * Build: icx -O3 -qopenmp -xHost bw_sharers.c -o bw_sharers
 * Run:   OMP_NUM_THREADS=<cores> OMP_PROC_BIND=close OMP_PLACES=cores ./bw_sharers freq_GHz [buf_GiB] [panel_KiB] [reps]
 *        freq_GHz: core frequency of the workload the B/cycle values are meant for
 */
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/mman.h>

#define ALIGN (2UL * 1024 * 1024)
#define MAX_BYTES_PER_CORE (4UL * 1024 * 1024 * 1024)

static __m512 read_panel(const float *p, size_t n, __m512 acc) {
  __m512 a0 = acc, a1 = _mm512_setzero_ps(), a2 = a1, a3 = a1, a4 = a1, a5 = a1, a6 = a1, a7 = a1;
  for (size_t i = 0; i < n; i += 128) {
    a0 = _mm512_add_ps(a0, _mm512_load_ps(p + i));
    a1 = _mm512_add_ps(a1, _mm512_load_ps(p + i + 16));
    a2 = _mm512_add_ps(a2, _mm512_load_ps(p + i + 32));
    a3 = _mm512_add_ps(a3, _mm512_load_ps(p + i + 48));
    a4 = _mm512_add_ps(a4, _mm512_load_ps(p + i + 64));
    a5 = _mm512_add_ps(a5, _mm512_load_ps(p + i + 80));
    a6 = _mm512_add_ps(a6, _mm512_load_ps(p + i + 96));
    a7 = _mm512_add_ps(a7, _mm512_load_ps(p + i + 112));
  }
  return _mm512_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)),
                       _mm512_add_ps(_mm512_add_ps(a4, a5), _mm512_add_ps(a6, a7)));
}

int main(int argc, char **argv) {
  if (argc < 2 || atof(argv[1]) <= 0) {
    fprintf(stderr, "Usage: %s freq_GHz [buf_GiB] [panel_KiB] [reps]\n", argv[0]);
    return 1;
  }
  double freq = atof(argv[1]);
  double gib = argc > 2 ? atof(argv[2]) : 32.0;
  size_t panel_bytes = (size_t)(argc > 3 ? atol(argv[3]) : 128) * 1024;
  int reps = argc > 4 ? atoi(argv[4]) : 3;
  const int T = omp_get_max_threads();

  if (panel_bytes == 0 || panel_bytes % 512) { fprintf(stderr, "panel_KiB must be > 0\n"); return 1; }
  size_t npanels = (size_t)(gib * 1024 * 1024 * 1024) / panel_bytes;
  size_t bytes = (npanels * panel_bytes + ALIGN - 1) / ALIGN * ALIGN;
  const size_t pn = panel_bytes / sizeof(float);

  float *buf = (float *)aligned_alloc(ALIGN, bytes);
  float *sink = (float *)aligned_alloc(64, (size_t)T * 16 * sizeof(float));
  double *tcore = (double *)malloc((size_t)T * sizeof(double));
  if (!buf || !sink || !tcore) { fprintf(stderr, "allocation failed\n"); return 1; }
  madvise(buf, bytes, MADV_HUGEPAGE);
#pragma omp parallel for schedule(static)
  for (size_t i = 0; i < bytes / sizeof(float); i++) buf[i] = 1.0f;

  printf("Threads: %d   Buffer: %.1f GiB (%zu panels)   Panel: %zu KiB   Reps: %d   f=%.3g GHz\n\n",
         T, (double)npanels * panel_bytes / (1024.0 * 1024 * 1024), npanels, panel_bytes / 1024, reps, freq);
  printf("%-8s %-8s %-12s %-14s %-14s %-14s %-12s\n", "sharers", "panels", "per-core GB/s",
         "per-core B/cyc", "min-core B/cyc", "max-core B/cyc", "DRAM GB/s");

  char bw_list[1024];
  int len = 0;
  size_t cursor = 0;  // keeps rotating across reps and sharer counts so every panel starts cold
  for (int s = 1; s <= T; s *= 2) {
    if (T % s) continue;
    const size_t ngroups = (size_t)(T / s);
    size_t iters = npanels / ngroups;
    if (iters * panel_bytes > MAX_BYTES_PER_CORE) iters = MAX_BYTES_PER_CORE / panel_bytes;
    if (iters == 0) { fprintf(stderr, "buffer too small for %zu groups\n", ngroups); return 1; }
    const double core_bytes = (double)iters * panel_bytes;

    double best_agg = 0, best_min = 0, best_max = 0;
    for (int r = 0; r < reps; r++) {
#pragma omp parallel
      {
        const int tid = omp_get_thread_num();
        const size_t g = (size_t)(tid / s);
        __m512 acc = _mm512_setzero_ps();
#pragma omp barrier
        double t0 = omp_get_wtime();
        for (size_t i = 0; i < iters; i++) {
          size_t panel = (cursor + i * ngroups + g) % npanels;
          acc = read_panel(buf + panel * pn, pn, acc);
        }
        tcore[tid] = omp_get_wtime() - t0;
        sink[tid * 16] = _mm512_reduce_add_ps(acc);
      }
      cursor = (cursor + iters * ngroups) % npanels;

      double tmax = 0, bmin = 1e30, bmax = 0;
      for (int t = 0; t < T; t++) {
        double b = core_bytes / tcore[t];
        if (tcore[t] > tmax) tmax = tcore[t];
        if (b < bmin) bmin = b;
        if (b > bmax) bmax = b;
      }
      double agg = core_bytes / tmax;  // per-core rate if all cores finish together
      if (agg > best_agg) { best_agg = agg; best_min = bmin; best_max = bmax; }
    }
    const double bpc = best_agg / (freq * 1e9);
    printf("%-8d %-8zu %-12.2f %-14.2f %-14.2f %-14.2f %-12.1f\n", s, ngroups, best_agg / 1e9, bpc,
           best_min / (freq * 1e9), best_max / (freq * 1e9), best_agg * T / s / 1e9);
    len += snprintf(bw_list + len, sizeof bw_list - len, "%s%d:%.2f", len ? "," : "", s, bpc);
  }
  printf("\nper-core B/cyc = bytes read by a core / (slowest core time * f)\n");
  printf("DRAM GB/s      = unique bytes fetched from memory (each panel counted once)\n");
  printf("\n--bw \"%s\"\n", bw_list);

  float s = 0;
  for (int t = 0; t < T; t++) s += sink[t * 16];
  if (s == 12345.0f) printf("%f\n", s);
  free(buf); free(sink); free(tcore);
  return 0;
}
