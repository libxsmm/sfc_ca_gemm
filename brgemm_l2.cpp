/* L2-resident BRGEMM throughput: every thread runs the same LIBXSMM BF16 BRGEMM as sfc_ca_gemm
 * (bm x bn x bk blocks, stride batch-reduce over K) in a hot loop on private A/B/C buffers.
 * Core cycles are counted per thread with perf events, so FLOPs/cycle is frequency-independent.
 * Build: icpx -qopenmp -xHost -std=c++17 -O2 -I./libxsmm/include brgemm_l2.cpp \
 *          -L./libxsmm/lib -Wl,-rpath,./libxsmm/lib -lxsmm -o brgemm_l2
 * Run:   OMP_NUM_THREADS=<cores> KMP_AFFINITY=granularity=fine,compact,1,0 ./brgemm_l2 [K] [brcount] [iters] [warmup] [peak_flops_per_cycle] [freq_GHz]
 * One iteration computes a full 32x32 C tile over K like sfc_ca_gemm: a single BETA_0 call if
 * brcount covers K, else zero C + ceil(K/bk/brcount) beta=1 calls.
 */
#include <libxsmm.h>
#include <linux/perf_event.h>
#include <omp.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#include <algorithm>
#include <vector>

static int open_cycles_counter() {
  struct perf_event_attr pe = {};
  pe.type = PERF_TYPE_HARDWARE;
  pe.size = sizeof(pe);
  pe.config = PERF_COUNT_HW_CPU_CYCLES;
  pe.disabled = 1;
  pe.exclude_kernel = 1;
  pe.exclude_hv = 1;
  return (int)syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
}

static libxsmm_bfloat16 to_bf16(float f) {
  uint32_t u;
  memcpy(&u, &f, sizeof(u));
  return (libxsmm_bfloat16)((u + 0x7fffu + ((u >> 16) & 1u)) >> 16);
}

int main(int argc, char **argv) {
  const long bm = 32, bn = 32, bk = 32;
  const long K = argc > 1 ? atol(argv[1]) : 2048;
  const long Kb = K / bk;
  const long brcount = (argc > 2 && atol(argv[2]) > 0) ? std::min(atol(argv[2]), Kb) : Kb;
  const long iters = argc > 3 ? atol(argv[3]) : 2000000;
  const long warmup = argc > 4 ? atol(argv[4]) : 500000;
  const double peak_fpc = argc > 5 ? atof(argv[5]) : 1024.0;
  const double freq_ghz = argc > 6 ? atof(argv[6]) : 0.0;  // optional nominal frequency
  const long rounds = (Kb + brcount - 1) / brcount;
  const int T = omp_get_max_threads();

  libxsmm_init();
  auto dt = LIBXSMM_DATATYPE_BF16;
  auto l_flags = LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG |
                 LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG;
  if (rounds == 1) l_flags |= LIBXSMM_GEMM_FLAG_BETA_0;
  auto l_tc_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N');
  auto l_tr_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N');
  auto l_shape = libxsmm_create_gemm_shape(bm, bn, bk, bm, bk, bm, dt, dt, dt, LIBXSMM_DATATYPE_F32);
  auto l_brconfig = libxsmm_create_gemm_batch_reduce_config(LIBXSMM_GEMM_BATCH_REDUCE_STRIDE,
                                                            bm * bk * sizeof(libxsmm_bfloat16),
                                                            bk * bn * sizeof(libxsmm_bfloat16), brcount);
  libxsmm_gemmfunction brgemm = libxsmm_dispatch_brgemm(l_shape, l_flags, LIBXSMM_GEMM_PREFETCH_NONE, l_brconfig);
  libxsmm_tilecfgfunction tilecfg = libxsmm_dispatch_tilecfg_gemm(l_shape, l_tc_flags);
  libxsmm_tilecfgfunction tilerel = libxsmm_dispatch_tilecfg_gemm(l_shape, l_tr_flags);
  if (!brgemm) { fprintf(stderr, "BRGEMM dispatch failed\n"); return 1; }

  std::vector<double> t_core(T), cyc_core(T);
  bool perf_ok = true;

#pragma omp parallel
  {
    const int tid = omp_get_thread_num();
    // Private, first-touched buffers so A/B/C stay in this core's L2.
    libxsmm_bfloat16 *A = (libxsmm_bfloat16 *)libxsmm_aligned_malloc(bm * K * sizeof(libxsmm_bfloat16), 2097152);
    libxsmm_bfloat16 *B = (libxsmm_bfloat16 *)libxsmm_aligned_malloc(bn * K * sizeof(libxsmm_bfloat16), 2097152);
    libxsmm_bfloat16 *C = (libxsmm_bfloat16 *)libxsmm_aligned_malloc(bm * bn * sizeof(libxsmm_bfloat16), 4096);
    unsigned int seed = 1234u + (unsigned int)tid;
    for (long i = 0; i < bm * K; i++) A[i] = to_bf16((float)rand_r(&seed) / (float)RAND_MAX - 0.5f);
    for (long i = 0; i < bn * K; i++) B[i] = to_bf16((float)rand_r(&seed) / (float)RAND_MAX - 0.5f);

    libxsmm_gemm_param p;
    long brc = brcount;
    p.op.tertiary = &brc;
    p.c.primary = C;
    auto tile = [&]() {
      if (rounds == 1) {
        brc = brcount;
        p.a.primary = A;
        p.b.primary = B;
        brgemm(&p);
        return;
      }
      memset(C, 0, bm * bn * sizeof(libxsmm_bfloat16));
      for (long i_k = 0; i_k < Kb; i_k += brcount) {
        brc = std::min(brcount, Kb - i_k);
        p.a.primary = A + i_k * bk * bm;
        p.b.primary = B + i_k * bk * bn;
        brgemm(&p);
      }
    };

    int fd = open_cycles_counter();
    if (fd < 0) {
#pragma omp atomic write
      perf_ok = false;
    }
    if (tilecfg) tilecfg(NULL);
    for (long it = 0; it < warmup; it++) tile();
#pragma omp barrier
    if (fd >= 0) { ioctl(fd, PERF_EVENT_IOC_RESET, 0); ioctl(fd, PERF_EVENT_IOC_ENABLE, 0); }
    double t0 = omp_get_wtime();
    for (long it = 0; it < iters; it++) tile();
    double t1 = omp_get_wtime();
    if (fd >= 0) ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
    if (tilerel) tilerel(NULL);

    uint64_t cyc = 0;
    if (fd >= 0 && read(fd, &cyc, sizeof(cyc)) != sizeof(cyc)) cyc = 0;
    if (fd >= 0) close(fd);
    t_core[tid] = t1 - t0;
    cyc_core[tid] = (double)cyc;
    libxsmm_free(A); libxsmm_free(B); libxsmm_free(C);
  }

  const double flops_core = 2.0 * bm * bn * (double)K * (double)iters;
  double tmax = 0, fpc_min = 1e30, fpc_max = 0, fpc_sum = 0, ghz_min = 1e30, ghz_max = 0, ghz_sum = 0;
  for (int t = 0; t < T; t++) {
    if (t_core[t] > tmax) tmax = t_core[t];
    double ghz = cyc_core[t] / t_core[t] / 1e9;
    ghz_sum += ghz; if (ghz < ghz_min) ghz_min = ghz; if (ghz > ghz_max) ghz_max = ghz;
    double fpc = cyc_core[t] > 0 ? flops_core / cyc_core[t] : 0;
    fpc_sum += fpc; if (fpc < fpc_min) fpc_min = fpc; if (fpc > fpc_max) fpc_max = fpc;
  }
  const double gflops = flops_core * T / tmax / 1e9;
  const double peak_nominal = (double)T * freq_ghz * peak_fpc;

  printf("BRGEMM %ldx%ldx%ld (bk=%ld, brcount=%ld, %ld round%s%s) BF16, %d threads, %ld tiles/thread (+%ld warmup)\n",
         bm, bn, K, bk, brcount, rounds, rounds > 1 ? "s" : "", rounds > 1 ? ", zero+beta=1" : ", beta=0",
         T, iters, warmup);
  printf("Wall time            : %.3f s\n", tmax);
  printf("Throughput           : %.1f GFLOPS (%.2f GFLOPS/core)\n", gflops, gflops / T);
  if (freq_ghz > 0)
    printf("vs nominal peak      : %.2f%% of %.1f GFLOPS (%d x %.3g GHz x %.0f FLOPs/cycle)\n",
           100.0 * gflops / peak_nominal, peak_nominal, T, freq_ghz, peak_fpc);
  if (perf_ok) {
    printf("Measured core freq   : avg %.3f GHz  (min %.3f, max %.3f)\n", ghz_sum / T, ghz_min, ghz_max);
    printf("FLOPs/cycle/core     : avg %.1f  (min %.1f, max %.1f)  -> %.2f%% of %.0f FLOPs/cycle\n",
           fpc_sum / T, fpc_min, fpc_max, 100.0 * fpc_sum / T / peak_fpc, peak_fpc);
  } else {
    printf("perf_event_open failed: no cycle counts (check /proc/sys/kernel/perf_event_paranoid)\n");
  }
  return 0;
}
