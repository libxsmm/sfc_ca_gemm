/* Simple AVX-512 memory bandwidth test (read / NT-write / regular-write).
 * Build: icx -O3 -qopenmp -xHost bw_test.c -o bw_test
 * Run:   OMP_NUM_THREADS=<cores> OMP_PROC_BIND=close OMP_PLACES=cores ./bw_test [GiB] [reps]
 */
#include <immintrin.h>
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>

#define ALIGN (2UL * 1024 * 1024)

static double run_read(const float *buf, size_t n, float *sink) {
  double t0, t1;
#pragma omp parallel
  {
    __m512 a0 = _mm512_setzero_ps(), a1 = a0, a2 = a0, a3 = a0, a4 = a0, a5 = a0, a6 = a0, a7 = a0;
#pragma omp barrier
#pragma omp master
    t0 = omp_get_wtime();
#pragma omp for schedule(static) nowait
    for (size_t i = 0; i < n; i += 128) {
      a0 = _mm512_add_ps(a0, _mm512_load_ps(buf + i));
      a1 = _mm512_add_ps(a1, _mm512_load_ps(buf + i + 16));
      a2 = _mm512_add_ps(a2, _mm512_load_ps(buf + i + 32));
      a3 = _mm512_add_ps(a3, _mm512_load_ps(buf + i + 48));
      a4 = _mm512_add_ps(a4, _mm512_load_ps(buf + i + 64));
      a5 = _mm512_add_ps(a5, _mm512_load_ps(buf + i + 80));
      a6 = _mm512_add_ps(a6, _mm512_load_ps(buf + i + 96));
      a7 = _mm512_add_ps(a7, _mm512_load_ps(buf + i + 112));
    }
#pragma omp barrier
#pragma omp master
    t1 = omp_get_wtime();
    a0 = _mm512_add_ps(_mm512_add_ps(_mm512_add_ps(a0, a1), _mm512_add_ps(a2, a3)),
                       _mm512_add_ps(_mm512_add_ps(a4, a5), _mm512_add_ps(a6, a7)));
    sink[omp_get_thread_num() * 16] = _mm512_reduce_add_ps(a0);
  }
  return t1 - t0;
}

static double run_write(float *buf, size_t n, int nt, float val) {
  double t0, t1;
#pragma omp parallel
  {
    const __m512 v = _mm512_set1_ps(val);
#pragma omp barrier
#pragma omp master
    t0 = omp_get_wtime();
    if (nt) {
#pragma omp for schedule(static) nowait
      for (size_t i = 0; i < n; i += 128) {
        for (int j = 0; j < 128; j += 16) _mm512_stream_ps(buf + i + j, v);
      }
      _mm_sfence();
    } else {
#pragma omp for schedule(static) nowait
      for (size_t i = 0; i < n; i += 128) {
        for (int j = 0; j < 128; j += 16) _mm512_store_ps(buf + i + j, v);
      }
    }
#pragma omp barrier
#pragma omp master
    t1 = omp_get_wtime();
  }
  return t1 - t0;
}

static void report(const char *name, double bytes, double *t, int reps) {
  double best = t[0], sum = 0.0;
  for (int r = 0; r < reps; r++) { if (t[r] < best) best = t[r]; sum += t[r]; }
  printf("%-28s best %8.1f GB/s   avg %8.1f GB/s\n", name, bytes / best / 1e9, bytes / (sum / reps) / 1e9);
}

int main(int argc, char **argv) {
  double gib = argc > 1 ? atof(argv[1]) : 32.0;
  int reps = argc > 2 ? atoi(argv[2]) : 10;
  size_t bytes = (size_t)(gib * 1024 * 1024 * 1024) / ALIGN * ALIGN;
  size_t n = bytes / sizeof(float);
  int nthr = omp_get_max_threads();

  float *buf = (float *)aligned_alloc(ALIGN, bytes);
  float *sink = (float *)aligned_alloc(64, (size_t)nthr * 16 * sizeof(float));
  double *t = (double *)malloc(reps * sizeof(double));
  if (!buf || !sink || !t) { fprintf(stderr, "allocation failed\n"); return 1; }
  madvise(buf, bytes, MADV_HUGEPAGE);

  /* First touch with the same static schedule as the kernels */
  run_write(buf, n, 0, 1.0f);

  printf("Threads: %d   Buffer: %.2f GiB   Reps: %d\n", nthr, bytes / (1024.0 * 1024 * 1024), reps);

  for (int r = 0; r < reps; r++) t[r] = run_read(buf, n, sink);
  report("Read  (AVX-512 loads)", (double)bytes, t, reps);

  for (int r = 0; r < reps; r++) t[r] = run_write(buf, n, 1, (float)r);
  report("Write (AVX-512 NT stores)", (double)bytes, t, reps);

  for (int r = 0; r < reps; r++) t[r] = run_write(buf, n, 0, (float)r);
  report("Write (AVX-512 stores)", (double)bytes, t, reps);
  printf("  (regular stores also incur RFO reads: DRAM traffic ~2x the reported value)\n");

  double s = 0.0;
  for (int i = 0; i < nthr; i++) s += sink[i * 16];
  if (s == 12345.0f) printf("%f\n", s);

  free(buf); free(sink); free(t);
  return 0;
}
