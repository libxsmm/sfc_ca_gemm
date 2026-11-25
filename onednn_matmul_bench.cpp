// Full oneDNN matmul baseline for SFC-CA GEMM, using the same timing harness as sfc_ca_gemm.
// Usage: ./onednn_matmul_bench M N K [n_layers=-1] [n_iters=10] [layout=any|plain]
// M, N, K follow sfc_ca_gemm's column-major semantics: C(MxN) = A(MxK) * B(KxN).
// oneDNN is row-major, so it computes C^T(NxM) = B^T(NxK) * A^T(KxM): src = B, weights = A.
#include "oneapi/dnnl/dnnl.hpp"
#include <omp.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <string>
#include <vector>
#include <unistd.h>
#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#endif

using namespace dnnl;
using tag = memory::format_tag;
using dt = memory::data_type;

static double ifreq;

static inline unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}

static double getFreq() {
  long long s = rdtsc();
  sleep(1);
  return (double)(rdtsc() - s);
}

static inline double getTime() { return rdtsc() * ifreq; }

static int open_thread_cycles_counter() {
#ifdef __linux__
  struct perf_event_attr pe;
  memset(&pe, 0, sizeof(pe));
  pe.type = PERF_TYPE_HARDWARE;
  pe.size = sizeof(pe);
  pe.config = PERF_COUNT_HW_CPU_CYCLES;
  pe.disabled = 1;
  pe.exclude_kernel = 1;
  pe.exclude_hv = 1;
  return (int)syscall(__NR_perf_event_open, &pe, 0, -1, -1, 0);
#else
  return -1;
#endif
}

static void fill_bf16(memory &m, long n, unsigned seed) {
  uint16_t *p = (uint16_t *)m.get_data_handle();
#pragma omp parallel for
  for (long i = 0; i < n; i++) {
    unsigned h = (unsigned)(i * 2654435761u) ^ seed;
    float f = (float)((h >> 8) % 1000) / 1000.0f - 0.5f;
    uint32_t u;
    memcpy(&u, &f, 4);
    p[i] = (uint16_t)((u + 0x7fff + ((u >> 16) & 1)) >> 16);
  }
}

int main(int argc, char **argv) {
  if (argc < 4) {
    printf("Usage: %s M N K [n_layers=-1] [n_iters=10] [layout=any|plain]\n", argv[0]);
    return 1;
  }
  const long M = atol(argv[1]), N = atol(argv[2]), K = atol(argv[3]);
  long n_layers = argc > 4 ? atol(argv[4]) : -1;
  const long n_iters = argc > 5 ? atol(argv[5]) : 10;
  const bool plain = argc > 6 && std::string(argv[6]) == "plain";
  ifreq = 1.0 / getFreq();

  // Same rotating-buffer sizing rule as sfc_ca_gemm (BF16 in, BF16 out, >= 5 GB in total).
  const double set_gb = 2.0 * ((double)M * K + (double)M * N + (double)K * N) / (1024.0 * 1024.0 * 1024.0);
  if (n_layers == -1) {
    n_layers = 1;
    while (n_layers * set_gb < 5.0) n_layers++;
    printf("Autocalculated %ld layers with total size %.2g GB\n", n_layers, n_layers * set_gb);
  }

  engine eng(engine::kind::cpu, 0);
  stream strm(eng);

  // oneDNN dims: src = B^T (N x K), weights = A^T (K x M), dst = C^T (N x M).
  memory::desc src_user({N, K}, dt::bf16, tag::ab);
  memory::desc wei_user({K, M}, dt::bf16, tag::ab);
  memory::desc dst_user({N, M}, dt::bf16, tag::ab);
  const tag t = plain ? tag::ab : tag::any;
  memory::desc src_md({N, K}, dt::bf16, t);
  memory::desc wei_md({K, M}, dt::bf16, t);
  memory::desc dst_md({N, M}, dt::bf16, t);

  matmul::primitive_desc pd(eng, src_md, wei_md, dst_md);
  matmul prim(pd);
  printf("oneDNN %d.%d.%d matmul impl: %s (layout %s)\n", dnnl_version()->major, dnnl_version()->minor,
         dnnl_version()->patch, pd.impl_info_str(), plain ? "plain" : "any");
  printf("src %s, weights %s, dst %s\n",
         pd.src_desc() == src_user ? "plain" : "blocked",
         pd.weights_desc() == wei_user ? "plain" : "blocked",
         pd.dst_desc() == dst_user ? "plain" : "blocked");

  memory src_plain(src_user, eng), wei_plain(wei_user, eng);
  fill_bf16(src_plain, N * K, 0x1234u);
  fill_bf16(wei_plain, K * M, 0x9876u);

  // Inputs are converted into the implementation's preferred layouts outside the timed region.
  std::vector<memory> src(n_layers), wei(n_layers), dst(n_layers);
  for (long l = 0; l < n_layers; l++) {
    src[l] = memory(pd.src_desc(), eng);
    wei[l] = memory(pd.weights_desc(), eng);
    dst[l] = memory(pd.dst_desc(), eng);
    reorder(src_plain, src[l]).execute(strm, src_plain, src[l]);
    reorder(wei_plain, wei[l]).execute(strm, wei_plain, wei[l]);
    memset(dst[l].get_data_handle(), 0, pd.dst_desc().get_size());
  }
  memory scratch(pd.scratchpad_desc(), eng);
  strm.wait();

  auto run_layers = [&]() {
    for (long l = 0; l < n_layers; l++)
      prim.execute(strm, {{DNNL_ARG_SRC, src[l]}, {DNNL_ARG_WEIGHTS, wei[l]}, {DNNL_ARG_DST, dst[l]},
                          {DNNL_ARG_SCRATCHPAD, scratch}});
    strm.wait();
  };

  run_layers();  // warmup

  printf("##############################################################\n");
  printf("    %ld Sets of GEMMS with sizes  %ld x %ld x %ld  (M x N x K)  \n", n_layers, M, N, K);
  printf("##############################################################\n");

  const int n_threads = omp_get_max_threads();
  std::vector<int> cyc_fd(n_threads, -1);
#pragma omp parallel
  cyc_fd[omp_get_thread_num()] = open_thread_cycles_counter();
#ifdef __linux__
  for (int fd : cyc_fd) if (fd >= 0) { ioctl(fd, PERF_EVENT_IOC_RESET, 0); ioctl(fd, PERF_EVENT_IOC_ENABLE, 0); }
#endif
  double t_start = getTime();
  for (long it = 0; it < n_iters; it++) run_layers();
  double t_end = getTime();
  double cyc_sum = 0.0;
  int n_cyc = 0;
#ifdef __linux__
  for (int fd : cyc_fd) {
    if (fd < 0) continue;
    ioctl(fd, PERF_EVENT_IOC_DISABLE, 0);
    unsigned long long c = 0;
    if (read(fd, &c, sizeof(c)) == sizeof(c)) { cyc_sum += (double)c; n_cyc++; }
    close(fd);
  }
#endif

  const double gflop = 2.0 * (double)n_layers * M * N * K / 1e9;
  const double t_it = (t_end - t_start) / (double)n_iters;
  printf("Time is %.5g ms (%.7g GFLOPS)\n", 1000.0 * t_it, gflop / t_it);
  printf("Effective total GEMM sizes: %.5g GB\n", n_layers * set_gb);
  printf("MEASURE %.7g ONEDNN_MATMUL_%ld_%ld_%ld_layout_%s_threads%d\n", gflop / t_it, M, N, K,
         plain ? "plain" : "any", n_threads);

  double flops_per_cycle = 1024.0;
  if (getenv("SFC_CA_GEMM_FLOPS_PER_CYCLE")) flops_per_cycle = atof(getenv("SFC_CA_GEMM_FLOPS_PER_CYCLE"));
  if (getenv("SFC_CA_GEMM_FREQ_GHZ")) {
    double freq_ghz = atof(getenv("SFC_CA_GEMM_FREQ_GHZ"));
    double peak_gflops = (double)n_threads * freq_ghz * flops_per_cycle;
    printf("Peak is %.7g GFLOPS (%d cores x %.3g GHz x %.0f FLOPs/cycle) -> %.2f%% of peak\n", peak_gflops,
           n_threads, freq_ghz, flops_per_cycle, 100.0 * (gflop / t_it) / peak_gflops);
  }
  if (n_cyc == n_threads) {
    double cyc_avg = cyc_sum / n_cyc;
    double meas_ghz = cyc_avg / (t_end - t_start) / 1e9;
    double fpc = gflop * 1e9 * (double)n_iters / ((double)n_threads * cyc_avg);
    printf("Measured freq %.3f GHz -> %.1f FLOPs/cycle/core -> %.2f%% of peak at measured freq\n", meas_ghz, fpc,
           100.0 * fpc / flops_per_cycle);
  } else {
    printf("Measured freq unavailable (perf_event_open failed on %d/%d threads)\n", n_threads - n_cyc, n_threads);
  }
  return 0;
}
