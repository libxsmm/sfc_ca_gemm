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

#ifndef SFC_CA_GEMM_HPP
#define SFC_CA_GEMM_HPP

#define USE_ONEDNN
// oneDNN ukernel API
#ifndef DNNL_EXPERIMENTAL_UKERNEL
#define DNNL_EXPERIMENTAL_UKERNEL
#endif
#include "oneapi/dnnl/dnnl.hpp"
#include "oneapi/dnnl/dnnl_ukernel.hpp"
#include <vector>

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <cmath>
#include <omp.h>
#include <immintrin.h>
#ifdef __linux__
#include <linux/perf_event.h>
#include <sys/ioctl.h>
#include <sys/syscall.h>
#include <unistd.h>
#endif
#include <libxsmm.h>
#include <libxsmm_utils.h>
#include <list>
#include <stdint.h>
#include <unordered_map>
#include "sfc_utils.h"
#include "sfc_ca_gemm_heuristics.h"
#include "sfc_ca_gemm_heuristics.h"
// ALIGNEMNET SIZE is 2 MB

#define ALIGNMENT_SIZE (2 * 1024 * 1024)

// Forward declarations
template<typename DType> libxsmm_datatype sfc_ca_gemm_get_libxsmm_dtype();
unsigned int sfc_ca_gemm_fill_sfc_index_map(unsigned char **sfc_index_map, unsigned int Mb, unsigned int Nb, unsigned int m_step = 1, unsigned int n_step = 1);
void sfc_ca_gemm_extract_indices_from_sfc(int *i_m, int *i_n, unsigned char *sfc_index_map, int sfc_index, unsigned int index_tsize);

// Type trait to determine output type (INT32 for INT8 input, same as input otherwise)
template<typename DType> struct output_type { using type = DType; };
template<> struct output_type<char> { using type = int; };

// Timing utilities
double ifreq;

#ifdef __x86_64__
static __inline__ unsigned long long rdtsc(void) {
  unsigned hi, lo;
  __asm__ __volatile__("rdtsc" : "=a"(lo), "=d"(hi));
  return ((unsigned long long)lo) | (((unsigned long long)hi) << 32);
}
#else
static __inline__ unsigned long long rdtsc(void) {
  unsigned long long virtual_timer_value;
  asm volatile("mrs %0, cntvct_el0" : "=r"(virtual_timer_value));
  return virtual_timer_value;
}
#endif

inline double getFreq() {
  long long int s = rdtsc();
  sleep(1);
  long long int e = rdtsc();
  return (e - s) * 1.0;
}

inline double getTime() {
  return rdtsc() * ifreq;
}

// Per-thread core-cycle counter (user mode) of the calling thread; -1 if unavailable.
inline int open_thread_cycles_counter() {
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

// Buffer utilities
void zero_buf(float* buf, long size) {
  for (long i = 0; i < size; i++) {
    buf[i] = 0.0f;
  }
}

// Buffer initialization
void init_buf(float* buf, long size, long initPos, long initOne)
{
  int i;
  zero_buf(buf, size);
  for (i = 0; i < size; ++i) {
    buf[i] = (float)((initOne != 0) ? 1.0 : ((initPos != 0) ? drand48() : (0.05 - drand48() / 10.0)));
  }
}

// Per-C-tile spinlock and contribution counter for the lock-based reduction, one cache line each.
struct alignas(64) c_tile_sync_t {
  int lock;
  int count;
};

static inline void c_tile_lock(c_tile_sync_t *s) {
  while (__atomic_exchange_n(&s->lock, 1, __ATOMIC_ACQUIRE)) {
    while (__atomic_load_n(&s->lock, __ATOMIC_RELAXED)) _mm_pause();
  }
}

static inline void c_tile_unlock(c_tile_sync_t *s) {
  __atomic_store_n(&s->lock, 0, __ATOMIC_RELEASE);
}

// Per-thread plan of the L2-aware prefetching variant (SFC_CA_GEMM_PF=1): which BRGEMMs find both
// A and B in L2 (predicted with an LRU model of the thread's own task sequence), and the A/B K-slices
// that will miss, in task order.
struct pf_item_t {
  long task;   // flat task index (round * n + q) that loads this slice
  long off;    // element offset into A (which == 0) or B (which == 1)
  long bytes;
  int which;
};

struct pf_plan_t {
  long lo = 0, n = 0;
  std::vector<uint8_t> l2_hit;   // per flat task: A and B both predicted in L2
  std::vector<pf_item_t> items;
  long budget = 0;               // prefetch bytes issued per L2-hit BRGEMM
};

// Define struct to store GEMM configuration
typedef struct
{
  long M, N, K;
  long Mb, Nb, Kb;
  long bm, bn, bk;
  long K_layers;
  long brcount;
  void *gemm_scratch;
  unsigned char *sfc_index_map;
  unsigned int index_tsize;
  libxsmm_gemmfunction brgemm_kernel;
  libxsmm_meltwfunction_unary zero_kernel;
  libxsmm_tilecfgfunction tileconfig_kernel;
  libxsmm_tilecfgfunction tilerelease_kernel;
  libxsmm_meltwfunction_binary l_add_kernel;
  libxsmm_meltwfunction_unary l_reduce_kernel;
  long m_step, n_step;  // SFC coarsening step sizes (default 1)
  long unblocked_bc;     // 0 = blocked activations (NCNC), 1 = flat activations (NC), 2 = flat with upfront packing of B
  void *scratch_B;       // scratch buffer for upfront packing of B (unblocked_bc == 2)
  libxsmm_meltwfunction_unary b_xform_kernel;  // identity kernel for packing B
  // oneDNN-specific fields
  void *onednn_brgemm_kernel;  // dnnl::ukernel::brgemm*
  size_t onednn_scratchpad_size;
  void *onednn_A_B_offsets_ptrs;  // thread-local offset vectors
  // I8 -> F32 dequantization fields (I8F32 variant)
  long dequant_f32;                    // 0 = disabled, 1 = dequant I32 accumulator to F32 at block output
  void *a_scales;                      // per-M scales (size M), applied to A
  void *b_scales;                      // per-N scales (size N), applied to B
  libxsmm_meqn_function dequant_kernel; // TPP equation: out_f32 = (a_scale x b_scale) * i32_acc
  // Lock-based C reduction (SFC_CA_GEMM_REDUCTION=locks, K_layers > 1): every partial result is
  // added into the single C under a per-tile lock, so no K_layers C copies and no final reduction.
  long locked_reduction;
  long skip_c_zero;                             // experiment: no C writes at all (LIBXSMM_X86_AMX_GEMM_SKIP_C_STORE)
  long pf_enable;                               // SFC_CA_GEMM_PF=1: L2-aware T1 prefetching (run_gemm_pf)
  long pf_window;                               // only prefetch slices needed within this many tasks
  long pf_split;                                // L2-hit BRGEMMs are split into this many calls
  void *pf_plans;                               // std::vector<pf_plan_t>, one per thread
  libxsmm_gemmfunction pf_f32_beta0_kernel;     // split pieces: BRGEMM into a stack FP32 tile, first piece
  libxsmm_gemmfunction pf_f32_beta1_kernel;     // split pieces: later pieces accumulate into the FP32 tile
  libxsmm_meltwfunction_unary pf_copy_kernel;   // FP32 tile -> C (first K round)
  libxsmm_meltwfunction_binary pf_acc_kernel;   // C += FP32 tile (later K rounds)
  long lock_stagger;                            // rotate each K-layer's start within a thread's chunk
  c_tile_sync_t *c_tile_sync;                   // Mb*Nb locks + contribution counters
  long c_contributions;                         // partial results per C tile and GEMM call
  void *c_partial_scratch;                      // per-thread bm*bn tile in the compute type
  size_t c_comp_size;                           // bytes per compute-type element
  libxsmm_gemmfunction brgemm_partial_kernel;   // BRGEMM (BETA_0) into the per-thread scratch tile
  libxsmm_meltwfunction_unary c_copy_kernel;    // first contribution: C = partial
  libxsmm_meltwfunction_binary c_acc_kernel;    // later contributions: C += partial
} gemm_config_t;

// Builds the per-thread prefetch plans of run_gemm_pf (K_layers = 1, blocked B/C). Each thread replays
// its own task sequence (same split and order as run_gemm) through an LRU model of its L2.
inline void sfc_ca_gemm_build_pf_plans(gemm_config_t *cfg, size_t in_bytes, size_t out_bytes) {
  const long Mb = cfg->Mb, Nb = cfg->Nb, Kb = cfg->Kb, K = cfg->K, bm = cfg->bm, bn = cfg->bn, bk = cfg->bk;
  const long brcount = cfg->brcount, total = Mb * Nb, rounds = (Kb + brcount - 1) / brcount;
  const double l2_bytes = 1024.0 * (getenv("SFC_CA_GEMM_PF_L2_KB") ? atof(getenv("SFC_CA_GEMM_PF_L2_KB")) : 2048.0);
  const double scale = getenv("SFC_CA_GEMM_PF_SCALE") ? atof(getenv("SFC_CA_GEMM_PF_SCALE")) : 1.0;
  cfg->pf_window = getenv("SFC_CA_GEMM_PF_WINDOW") ? atol(getenv("SFC_CA_GEMM_PF_WINDOW")) : 16;
  cfg->pf_split = getenv("SFC_CA_GEMM_PF_SPLIT") ? std::max(1L, atol(getenv("SFC_CA_GEMM_PF_SPLIT"))) : 1;
  std::vector<pf_plan_t> *plans = new std::vector<pf_plan_t>(omp_get_max_threads());

#pragma omp parallel
  {
    pf_plan_t &p = (*plans)[omp_get_thread_num()];
    const long nthr = omp_get_num_threads(), chunk = (total + nthr - 1) / nthr;
    p.lo = std::min(total, omp_get_thread_num() * chunk);
    p.n = std::min(total, p.lo + chunk) - p.lo;
    std::list<std::pair<uint64_t, long>> lru;
    std::unordered_map<uint64_t, std::list<std::pair<uint64_t, long>>::iterator> idx;
    double used = 0;
    auto access = [&](uint64_t key, long sz) {
      auto it = idx.find(key);
      if (it != idx.end()) { lru.splice(lru.begin(), lru, it->second); return true; }
      lru.emplace_front(key, sz);
      idx[key] = lru.begin();
      used += sz;
      while (used > l2_bytes && lru.size() > 1) { used -= lru.back().second; idx.erase(lru.back().first); lru.pop_back(); }
      return false;
    };
    p.l2_hit.assign((size_t)(rounds * p.n), 0);
    long miss_bytes = 0, hits = 0;
    for (long t = 0; t < rounds * p.n; t++) {
      const long rd = t / p.n, i_k = rd * brcount, brc = std::min(brcount, Kb - i_k);
      int i_m, i_n;
      sfc_ca_gemm_extract_indices_from_sfc(&i_m, &i_n, cfg->sfc_index_map, (int)(p.lo + t % p.n), cfg->index_tsize);
      const long a_bytes = brc * bk * bm * (long)in_bytes, b_bytes = brc * bk * bn * (long)in_bytes;
      const bool hit_a = access(((uint64_t)0 << 60) | ((uint64_t)rd << 32) | (uint64_t)i_m, a_bytes);
      const bool hit_b = access(((uint64_t)1 << 60) | ((uint64_t)rd << 32) | (uint64_t)i_n, b_bytes);
      access(((uint64_t)2 << 60) | (uint64_t)(i_n * Mb + i_m), bm * bn * (long)out_bytes);
      if (!hit_a) { p.items.push_back({t, i_m * K * bm + i_k * bk * bm, a_bytes, 0}); miss_bytes += a_bytes; }
      if (!hit_b) { p.items.push_back({t, i_n * K * bn + i_k * bk * bn, b_bytes, 1}); miss_bytes += b_bytes; }
      p.l2_hit[t] = (hit_a && hit_b) ? 1 : 0;
      hits += p.l2_hit[t];
    }
    p.budget = hits ? ((long)(scale * (double)miss_bytes / (double)hits) + 63) / 64 * 64 : 0;
  }
  const pf_plan_t &p0 = (*plans)[0];
  long hits0 = 0, miss0 = 0;
  for (uint8_t h : p0.l2_hit) hits0 += h;
  for (const pf_item_t &it : p0.items) miss0 += it.bytes;
  printf("L2-aware prefetch (thread 0): %ld BRGEMMs, %ld with A+B in L2, %zu missing slices (%.1f MB), %.1f KB T1 per L2-hit BRGEMM, window %ld, split %ld, L2 model %.0f KB\n",
         (long)p0.l2_hit.size(), hits0, p0.items.size(), miss0 / 1e6, p0.budget / 1024.0, cfg->pf_window, cfg->pf_split, l2_bytes / 1024);
  cfg->pf_plans = (void *)plans;
}

template <typename DType>
gemm_config_t *setup_gemm_config(
    long M, long N, long K,
    long bm, long bn, long bk,
    long &kbf, long &K_layers,
    long m_step = 1, long n_step = 1,
    long unblocked_bc = 0,
    long use_nts = 0,
    long dequant_f32 = 0)
{
  gemm_config_t *config = new gemm_config_t();

  // Calculate derived parameters
  long Mb = M / bm, Nb = N / bn, Kb = K / bk;

#if 1
  long Kb_per_layer = (Kb + K_layers - 1) / K_layers;
  long brcount = (Kb_per_layer + kbf - 1) / kbf;
  long K_rounds_per_layer = (Kb_per_layer + brcount - 1) / brcount;
#else
  long brcount = (Kb / K_layers) / kbf;
  while (Kb % K_layers != 0)
  {
    K_layers--;
  }
  while ((Kb / K_layers) % kbf != 0)
  {
    kbf--;
  }
  brcount = (Kb / K_layers) / kbf;
#endif

  // Store basic parameters
  config->M = M;
  config->N = N;
  config->K = K;
  config->Mb = Mb;
  config->Nb = Nb;
  config->Kb = Kb;
  config->bm = bm;
  config->bn = bn;
  config->bk = bk;
  config->K_layers = K_layers;
  config->brcount = brcount;
  config->m_step = m_step;
  config->n_step = n_step;
  config->unblocked_bc = unblocked_bc;

  // Allocate output_partial scratch buffers for K_layers > 1
  using CType = typename output_type<DType>::type;
  const char *red_env = getenv("SFC_CA_GEMM_REDUCTION");
  const char *skip_store_env = getenv("LIBXSMM_X86_AMX_GEMM_SKIP_C_STORE");
  config->skip_c_zero = (skip_store_env != NULL && atoi(skip_store_env) != 0) ? 1 : 0;
  if (config->skip_c_zero) printf("Experiment: BRGEMM C stores and C zeroing disabled (results are invalid)\n");
  config->locked_reduction = (K_layers > 1 && red_env != NULL && strcmp(red_env, "locks") == 0) ? 1 : 0;
  if (config->locked_reduction && (unblocked_bc > 0 || dequant_f32)) {
    printf("SFC_CA_GEMM_REDUCTION=locks needs blocked B/C and no dequant; using the end-of-GEMM reduction\n");
    config->locked_reduction = 0;
  }
  const char *pf_env = getenv("SFC_CA_GEMM_PF");
  config->pf_enable = (pf_env != NULL && atoi(pf_env) > 0) ? 1 : 0;
  if (config->pf_enable && (K_layers > 1 || unblocked_bc > 0 || dequant_f32)) {
    printf("SFC_CA_GEMM_PF needs K_layers=1, blocked B/C and no dequant; prefetching disabled\n");
    config->pf_enable = 0;
  }
  int n_out_copies = LIBXSMM_MAX(1, K_layers - 1);
  CType *global_scratch = NULL;
  CType **output_partial_array = (CType**)libxsmm_aligned_malloc(sizeof(CType*) * n_out_copies, ALIGNMENT_SIZE);
  output_partial_array[0] = NULL;
  if (K_layers > 1 && !config->locked_reduction)
  {
    size_t scratch_size = (size_t)M * (size_t)N * sizeof(CType) * (size_t)(K_layers - 1);
    global_scratch = (CType *)libxsmm_aligned_malloc(scratch_size, ALIGNMENT_SIZE);
    for (int i = 1; i < K_layers; i++)
    {
      size_t scratch_offset = (size_t)(i - 1) * (size_t)M * (size_t)N;
      output_partial_array[i - 1] = (CType *)global_scratch + scratch_offset;
    }
  }
  config->gemm_scratch = (void *)output_partial_array;

  // Create SFC index map (coarsened SFC ordering baked into full Mb x Nb map)
  unsigned char *sfc_index_map = NULL;
  unsigned int index_tsize = sfc_ca_gemm_fill_sfc_index_map(&sfc_index_map, Mb, Nb, m_step, n_step);
  config->sfc_index_map = sfc_index_map;
  config->index_tsize = index_tsize;

  // Setup TPP kernels
  auto dtype = sfc_ca_gemm_get_libxsmm_dtype<DType>();
  auto dtype_out = sfc_ca_gemm_get_libxsmm_dtype<CType>();
  // Computation type: I32 for I8 inputs, F64 for F64 inputs, F32 for BF16/FP32
  auto dtype_comp = (dtype == LIBXSMM_DATATYPE_I8) ? LIBXSMM_DATATYPE_I32 : 
                    (dtype == LIBXSMM_DATATYPE_F64) ? LIBXSMM_DATATYPE_F64 : LIBXSMM_DATATYPE_F32;
  auto l_flags = LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG;
  auto l_tc_flags = LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N');
  auto l_tr_flags = LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N');
  auto l_shape = libxsmm_create_gemm_shape(bm, bn, bk, bm, (unblocked_bc == 1) ? K : bk, (unblocked_bc > 0) ? M : bm, dtype, dtype, dtype_out, dtype_comp);
  auto l_prefetch_flags = LIBXSMM_GEMM_PREFETCH_NONE;
  // BR unroll hint is an unsigned char in libxsmm: brcount > 255 wraps (256 -> 0 = rolled loop).
  const long br_unroll = getenv("SFC_CA_GEMM_BR_UNROLL") ? atol(getenv("SFC_CA_GEMM_BR_UNROLL")) : brcount;
  auto l_brconfig = libxsmm_create_gemm_batch_reduce_config(LIBXSMM_GEMM_BATCH_REDUCE_STRIDE, bm * bk * sizeof(DType), (unblocked_bc == 1) ? bk * sizeof(DType) : bk * bn * sizeof(DType), (unsigned char)br_unroll);
  auto l_unary_shape = libxsmm_create_meltw_unary_shape((unblocked_bc > 0) ? bm : bm * bn, (unblocked_bc > 0) ? bn : 1, (unblocked_bc > 0) ? M : bm * bn, (unblocked_bc > 0) ? M : bm * bn, dtype_out, dtype_out, dtype_comp);
  if (K_rounds_per_layer == 1) l_flags |= LIBXSMM_GEMM_FLAG_BETA_0;
  if (use_nts > 0) l_flags |= LIBXSMM_GEMM_FLAG_ALIGN_C_NTS_HINT;
  config->zero_kernel = libxsmm_dispatch_meltw_unary(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);
  config->tileconfig_kernel = libxsmm_dispatch_tilecfg_gemm(l_shape, l_tc_flags);
  config->tilerelease_kernel = libxsmm_dispatch_tilecfg_gemm(l_shape, l_tr_flags);
  config->brgemm_kernel = libxsmm_dispatch_brgemm(l_shape, l_flags, l_prefetch_flags, l_brconfig);
  if (config->pf_enable) {
    sfc_ca_gemm_build_pf_plans(config, sizeof(DType), sizeof(CType));
    if (config->pf_split > 1 && (bm != 32 || bn != 32)) {
      printf("SFC_CA_GEMM_PF_SPLIT needs bm = bn = 32; BRGEMMs are not split\n");
      config->pf_split = 1;
    }
    if (config->pf_split > 1) {
      auto f_flags = LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG | LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG;
      auto f_shape = libxsmm_create_gemm_shape(bm, bn, bk, bm, bk, bm, dtype, dtype, dtype_comp, dtype_comp);
      config->pf_f32_beta0_kernel = libxsmm_dispatch_brgemm(f_shape, f_flags | LIBXSMM_GEMM_FLAG_BETA_0, l_prefetch_flags, l_brconfig);
      config->pf_f32_beta1_kernel = libxsmm_dispatch_brgemm(f_shape, f_flags, l_prefetch_flags, l_brconfig);
      auto f_copy_shape = libxsmm_create_meltw_unary_shape(bm * bn, 1, bm * bn, bm * bn, dtype_comp, dtype_out, dtype_comp);
      config->pf_copy_kernel = libxsmm_dispatch_meltw_unary(LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, f_copy_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);
      auto f_acc_shape = libxsmm_create_meltw_binary_shape(bm * bn, 1, bm * bn, bm * bn, bm * bn, dtype_comp, dtype_out, dtype_out, dtype_comp);
      config->pf_acc_kernel = libxsmm_dispatch_meltw_binary(LIBXSMM_MELTW_TYPE_BINARY_ADD, f_acc_shape, LIBXSMM_MELTW_FLAG_BINARY_NONE);
      if (!config->pf_f32_beta0_kernel || !config->pf_f32_beta1_kernel || !config->pf_copy_kernel || !config->pf_acc_kernel) {
        printf("Error: could not dispatch the split-BRGEMM kernels\n");
        exit(1);
      }
    }
  }
  auto l_binary_shape = libxsmm_create_meltw_binary_shape(bm, (unblocked_bc == 0) ? bn : ((K_layers > 2) ? 1 : bn), (unblocked_bc > 0) ? M : bm, (unblocked_bc > 0) ? M : bm, (unblocked_bc > 0) ? M : bm, dtype_out, dtype_out, dtype_out, dtype_comp);
  config->l_add_kernel = libxsmm_dispatch_meltw_binary(LIBXSMM_MELTW_TYPE_BINARY_ADD, l_binary_shape, LIBXSMM_MELTW_FLAG_BINARY_NONE);
  auto l_reduce_shape = libxsmm_create_meltw_unary_shape((unblocked_bc > 0) ? bm : bm * bn, n_out_copies, M * N, (unblocked_bc > 0) ? bm : bm * bn, dtype_out, dtype_out, dtype_comp);
  config->l_reduce_kernel = libxsmm_dispatch_meltw_unary(LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_reduce_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS);

  // Setup dequantization equation for I8 -> F32 output (I8F32 variant).
  // At the block output level we compute: out_f32 = (a_scale[m] * b_scale[n]) * i32_acc[m][n]
  config->dequant_f32 = dequant_f32;
  config->a_scales = NULL;
  config->b_scales = NULL;
  config->dequant_kernel = NULL;
  if (dequant_f32 && dtype == LIBXSMM_DATATYPE_I8) {
    long c_ld = (unblocked_bc > 0) ? M : bm;
    libxsmm_matrix_arg_attributes arg_singular_attr = libxsmm_create_matrix_arg_attributes(LIBXSMM_MATRIX_ARG_TYPE_SINGULAR, LIBXSMM_MATRIX_ARG_SET_TYPE_NONE, 0, 0);
    libxsmm_blasint my_eqn = libxsmm_meqn_create();
    libxsmm_meqn_op_metadata  op_metadata = libxsmm_create_meqn_op_metadata(my_eqn, -1);
    libxsmm_meqn_arg_metadata arg_metadata;
    libxsmm_meqn_arg_shape    arg_shape_in, arg_shape_out;
    // Top op: out = scale_matrix * i32_acc
    libxsmm_meqn_push_back_binary_op(op_metadata, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_NONE);
    // Inner op: scale_matrix = a_scales(bm x 1) * b_scales(1 x bn)
    libxsmm_meqn_push_back_binary_op(op_metadata, LIBXSMM_MELTW_TYPE_BINARY_MUL, LIBXSMM_DATATYPE_F32, LIBXSMM_MELTW_FLAG_BINARY_BCAST_COL_IN_0 | LIBXSMM_MELTW_FLAG_BINARY_BCAST_ROW_IN_1);
    // arg 0: a_scales (bm, 1)
    arg_shape_in = libxsmm_create_meqn_arg_shape(bm, 1, bm, LIBXSMM_DATATYPE_F32);
    arg_metadata = libxsmm_create_meqn_arg_metadata(my_eqn, 0);
    libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
    // arg 1: b_scales (1, bn)
    arg_shape_in = libxsmm_create_meqn_arg_shape(1, bn, 1, LIBXSMM_DATATYPE_F32);
    arg_metadata = libxsmm_create_meqn_arg_metadata(my_eqn, 1);
    libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
    // arg 2: I32 accumulator block (bm, bn, ld=c_ld)
    arg_shape_in = libxsmm_create_meqn_arg_shape(bm, bn, c_ld, LIBXSMM_DATATYPE_I32);
    arg_metadata = libxsmm_create_meqn_arg_metadata(my_eqn, 2);
    libxsmm_meqn_push_back_arg(arg_metadata, arg_shape_in, arg_singular_attr);
    // output: F32 block (bm, bn, ld=c_ld)
    arg_shape_out = libxsmm_create_meqn_arg_shape(bm, bn, c_ld, LIBXSMM_DATATYPE_F32);
    config->dequant_kernel = libxsmm_dispatch_meqn(my_eqn, arg_shape_out);
  }

  // Setup upfront packing for B (unblocked_bc == 2)
  config->scratch_B = NULL;
  config->b_xform_kernel = NULL;
  if (unblocked_bc == 2) {
    config->scratch_B = (void *)libxsmm_aligned_malloc((size_t)K * (size_t)N * sizeof(DType), ALIGNMENT_SIZE);
    // Use BF8 instead of I8 for the copy kernel since libxsmm doesn't support I8 meltw identity
    auto xform_dtype = (dtype == LIBXSMM_DATATYPE_I8) ? LIBXSMM_DATATYPE_BF8 : dtype;
    auto xform_unary_shape = libxsmm_create_meltw_unary_shape(bk, bn, K, bk, xform_dtype, xform_dtype, xform_dtype);
    config->b_xform_kernel = libxsmm_dispatch_meltw_unary(LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, xform_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);
  }

  if (config->locked_reduction) {
    auto p_flags = LIBXSMM_GEMM_VNNI_FLAGS('N', 'N', 'V', 'N') | LIBXSMM_GEMM_FLAG_NO_RESET_TILECONFIG |
                   LIBXSMM_GEMM_FLAG_NO_SETUP_TILECONFIG | LIBXSMM_GEMM_FLAG_BETA_0;
    auto p_shape = libxsmm_create_gemm_shape(bm, bn, bk, bm, bk, bm, dtype, dtype, dtype_comp, dtype_comp);
    config->brgemm_partial_kernel = libxsmm_dispatch_brgemm(p_shape, p_flags, LIBXSMM_GEMM_PREFETCH_NONE, l_brconfig);
    config->tileconfig_kernel = libxsmm_dispatch_tilecfg_gemm(p_shape, l_tc_flags);
    config->tilerelease_kernel = libxsmm_dispatch_tilecfg_gemm(p_shape, l_tr_flags);
    auto c_copy_shape = libxsmm_create_meltw_unary_shape(bm * bn, 1, bm * bn, bm * bn, dtype_comp, dtype_out, dtype_comp);
    config->c_copy_kernel = libxsmm_dispatch_meltw_unary(LIBXSMM_MELTW_TYPE_UNARY_IDENTITY, c_copy_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);
    auto c_acc_shape = libxsmm_create_meltw_binary_shape(bm * bn, 1, bm * bn, bm * bn, bm * bn, dtype_comp, dtype_out, dtype_out, dtype_comp);
    config->c_acc_kernel = libxsmm_dispatch_meltw_binary(LIBXSMM_MELTW_TYPE_BINARY_ADD, c_acc_shape, LIBXSMM_MELTW_FLAG_BINARY_NONE);
    const long Kb_last_layer = Kb - (K_layers - 1) * Kb_per_layer;
    config->c_contributions = (K_layers - 1) * K_rounds_per_layer + (Kb_last_layer + brcount - 1) / brcount;
    config->c_tile_sync = (c_tile_sync_t *)libxsmm_aligned_malloc((size_t)Mb * Nb * sizeof(c_tile_sync_t), 64);
    memset((void *)config->c_tile_sync, 0, (size_t)Mb * Nb * sizeof(c_tile_sync_t));
    config->c_comp_size = (size_t)LIBXSMM_TYPESIZE(dtype_comp);
    config->c_partial_scratch = libxsmm_aligned_malloc((size_t)omp_get_max_threads() * bm * bn * config->c_comp_size, 64);
    const char *stagger_env = getenv("SFC_CA_GEMM_LOCK_STAGGER");
    config->lock_stagger = (stagger_env == NULL || atoi(stagger_env) != 0) ? 1 : 0;
    if (!config->brgemm_partial_kernel || !config->c_copy_kernel || !config->c_acc_kernel) {
      printf("Error: could not dispatch the lock-based reduction kernels\n");
      exit(1);
    }
    printf("C reduction: per-tile locks into a single C (%ld partial results per tile, K-layer stagger %s)\n",
           config->c_contributions, config->lock_stagger ? "on" : "off");
  }

  return config;
}

#ifdef USE_ONEDNN
// oneDNN-specific setup function
template <typename DType>
gemm_config_t *setup_gemm_config_onednn(
    long M, long N, long K,
    long bm, long bn, long bk,
    long &kbf, long &K_layers,
    long m_step = 1, long n_step = 1)
{
  gemm_config_t *config = new gemm_config_t();
  // Calculate derived parameters
  long Mb = M / bm, Nb = N / bn, Kb = K / bk;

#if 1
  long Kb_per_layer = (Kb + K_layers - 1) / K_layers;
  long brcount = (Kb_per_layer + kbf - 1) / kbf;
  long K_rounds_per_layer = (Kb_per_layer + brcount - 1) / brcount;
#else
  long brcount = (Kb / K_layers) / kbf;
  while (Kb % K_layers != 0)
  {
    K_layers--;
  }
  while ((Kb / K_layers) % kbf != 0)
  {
    kbf--;
  }
  brcount = (Kb / K_layers) / kbf;
#endif

  // Store basic parameters
  config->M = M;
  config->N = N;
  config->K = K;
  config->Mb = Mb;
  config->Nb = Nb;
  config->Kb = Kb;
  config->bm = bm;
  config->bn = bn;
  config->bk = bk;
  config->K_layers = K_layers;
  config->brcount = brcount;
  config->m_step = m_step;
  config->n_step = n_step;

  // Allocate output_partial scratch buffers for K_layers > 1
  int n_out_copies = LIBXSMM_MAX(1, K_layers - 1);
  DType *global_scratch = NULL;
  DType **output_partial_array = (DType **)libxsmm_aligned_malloc(sizeof(DType *) * n_out_copies, ALIGNMENT_SIZE);
  output_partial_array[0] = NULL;
  if (K_layers > 1)
  {
    size_t scratch_size = (size_t)M * (size_t)N * sizeof(DType) * (size_t)(K_layers - 1);
    global_scratch = (DType *)libxsmm_aligned_malloc(scratch_size, ALIGNMENT_SIZE);
    for (int i = 1; i < K_layers; i++)
    {
      size_t scratch_offset = (size_t)(i - 1) * (size_t)M * (size_t)N;
      output_partial_array[i - 1] = (DType *)global_scratch + scratch_offset;
    }
  }
  config->gemm_scratch = (void *)output_partial_array;

  // Create SFC index map (coarsened SFC ordering baked into full Mb x Nb map)
  unsigned char *sfc_index_map = NULL;
  unsigned int index_tsize = sfc_ca_gemm_fill_sfc_index_map(&sfc_index_map, Mb, Nb, m_step, n_step);
  config->sfc_index_map = sfc_index_map;
  config->index_tsize = index_tsize;

  // Setup libxsmm TPP kernels for auxiliary operations (zero, add, reduce)
  auto dtype = sfc_ca_gemm_get_libxsmm_dtype<DType>();
  auto l_unary_shape = libxsmm_create_meltw_unary_shape(bm * bn, 1, bm * bn, bm * bn, dtype, dtype, dtype);
  config->zero_kernel = libxsmm_dispatch_meltw_unary(LIBXSMM_MELTW_TYPE_UNARY_XOR, l_unary_shape, LIBXSMM_MELTW_FLAG_UNARY_NONE);

  // Tile config/release not needed for oneDNN\n  config->tileconfig_kernel = NULL;
  config->tilerelease_kernel = NULL;

  // LIBXSMM brgemm kernel not used (oneDNN replaces it)
  config->brgemm_kernel = NULL;

  auto l_binary_shape = libxsmm_create_meltw_binary_shape(bm, bn, bm, bm, bm, dtype, dtype, dtype, LIBXSMM_DATATYPE_F32);
  config->l_add_kernel = libxsmm_dispatch_meltw_binary(LIBXSMM_MELTW_TYPE_BINARY_ADD, l_binary_shape, LIBXSMM_MELTW_FLAG_BINARY_NONE);
  auto l_reduce_shape = libxsmm_create_meltw_unary_shape(bm * bn, n_out_copies, M * N, bm * bn, dtype, dtype, LIBXSMM_DATATYPE_F32);
  config->l_reduce_kernel = libxsmm_dispatch_meltw_unary(LIBXSMM_MELTW_TYPE_UNARY_REDUCE_X_OP_ADD, l_reduce_shape, LIBXSMM_MELTW_FLAG_UNARY_REDUCE_COLS);

  // Create oneDNN brgemm kernel
  dnnl::memory::data_type a_dt = dnnl::memory::data_type::bf16;
  dnnl::memory::data_type b_dt = dnnl::memory::data_type::bf16;
  dnnl::memory::data_type c_dt = dnnl::memory::data_type::bf16;

  const dnnl::memory::dim lda = bk; // A (activations) leading dim
  const dnnl::memory::dim ldb = bm; // VNNI - packed B(weights) leading dim
  const dnnl::memory::dim ldc = bm; // Local C buffer leading dimension
  const dnnl::memory::dim ldd = bm; // D (BF16 output) leading dimension

  // Allocate oneDNN brgemm kernel on heap (must persist beyond setup)
  dnnl::ukernel::brgemm *brgemm_onednn = new dnnl::ukernel::brgemm(bn, bm, bk, brcount, lda, ldb, ldc, a_dt, b_dt, c_dt, true);

  if (!(*brgemm_onednn))
  {
    printf("Error: oneDNN brgemm object was not constructed.\n");
    delete brgemm_onednn;
    config->onednn_brgemm_kernel = NULL;
    return config;
  }

  // Create post-ops: convert F32->BF16, then binary add with destination
  dnnl::memory::dims binary_add_dims = {bn, bm};
  auto binary_add_md = dnnl::memory::desc(binary_add_dims, dnnl::memory::data_type::bf16, {ldd, 1});

  dnnl::post_ops brgemm_po;
  brgemm_po.append_binary(dnnl::algorithm::binary_add, binary_add_md);

  brgemm_onednn->set_post_ops(ldd, dnnl::memory::data_type::bf16, brgemm_po);

  if (!brgemm_onednn->finalize())
  {
    printf("oneDNN brgemm kernel is not supported on this platform.\n");
    delete brgemm_onednn;
    config->onednn_brgemm_kernel = NULL;
    return config;
  }
  brgemm_onednn->generate();

  config->onednn_brgemm_kernel = (void *)brgemm_onednn;
  config->onednn_scratchpad_size = brgemm_onednn->get_scratchpad_size();

  // Create thread-local A_B_offsets buffers (one per thread)
  int num_threads = omp_get_max_threads();
  std::vector<std::pair<dnnl::memory::dim, dnnl::memory::dim>> ***tl_offsets_array =
      new std::vector<std::pair<dnnl::memory::dim, dnnl::memory::dim>> **[1];
  tl_offsets_array[0] = new std::vector<std::pair<dnnl::memory::dim, dnnl::memory::dim>> *[num_threads];

  for (int t = 0; t < num_threads; t++)
  {
    tl_offsets_array[0][t] = new std::vector<std::pair<dnnl::memory::dim, dnnl::memory::dim>>(brcount);
    const size_t a_dt_size = sizeof(DType);
    const size_t b_dt_size = sizeof(DType);
    for (dnnl::memory::dim br = 0; br < brcount; br++)
    {
      const dnnl::memory::dim A_offset_br = br * bk * bn * a_dt_size;
      const dnnl::memory::dim B_offset_br = br * bk * bm * b_dt_size;
      (*tl_offsets_array[0][t])[br] = std::make_pair(A_offset_br, B_offset_br);
    }
  }
  config->onednn_A_B_offsets_ptrs = (void *)tl_offsets_array;

  return config;
}
#endif // USE_ONEDNN

// Null pointer check
void check_null_ptr(void* ptr, const char* ptr_name) {
  if (ptr == NULL) {
    printf("Pointer for %s is NULL. Exiting...\n", ptr_name);
    exit(0);
  } 
}

// Datatype helpers
template <typename DType> libxsmm_datatype sfc_ca_gemm_get_libxsmm_dtype()
{
  return LIBXSMM_DATATYPE_BF16;
}

template<> libxsmm_datatype sfc_ca_gemm_get_libxsmm_dtype<float>() {
  return LIBXSMM_DATATYPE_F32;
}

template<> libxsmm_datatype sfc_ca_gemm_get_libxsmm_dtype<libxsmm_bfloat16>() {
  return LIBXSMM_DATATYPE_BF16;
}

template<> libxsmm_datatype sfc_ca_gemm_get_libxsmm_dtype<libxsmm_bfloat8>() {
  return LIBXSMM_DATATYPE_BF8;
}

template<> libxsmm_datatype sfc_ca_gemm_get_libxsmm_dtype<char>() {
  return LIBXSMM_DATATYPE_I8;
}

template<> libxsmm_datatype sfc_ca_gemm_get_libxsmm_dtype<int>() {
  return LIBXSMM_DATATYPE_I32;
}

template<> libxsmm_datatype sfc_ca_gemm_get_libxsmm_dtype<double>() {
  return LIBXSMM_DATATYPE_F64;
}

// Forward declaration of naive_fullyconnected struct
typedef struct {
  long N;
  long K;
  long M;
  long fuse_type;
} naive_gemm_t;

// Naive fullyconnected forward pass
LIBXSMM_INLINE void naive_gemm_fp(naive_gemm_t* param, const float* b_ptr, float* c_ptr, const float* a_ptr)
{
  const int N = param->N;
  const int K = param->K;
  const int M = param->M;

  int n, k, m;

  LIBXSMM_VLA_DECL(2, const float, b, b_ptr, K);
  LIBXSMM_VLA_DECL(2, const float, a, a_ptr, K);
  LIBXSMM_VLA_DECL(2,       float, c, c_ptr, M);

#if defined(_OPENMP)
  LIBXSMM_OMP_VAR(n); LIBXSMM_OMP_VAR(k); LIBXSMM_OMP_VAR(m);
# pragma omp parallel for private(n, m, k)
#endif
  for (m = 0; m < M; ++m) {
    for (n = 0; n < N; ++n) {
      float accum = 0.f;
      for (k = 0; k < K; ++k) {
        accum += LIBXSMM_VLA_ACCESS(2, a, m, k, K) * LIBXSMM_VLA_ACCESS(2, b, n, k, K);
      }
      LIBXSMM_VLA_ACCESS(2, c, n, m, M) = accum;
    }
  }
}

// Matrix copy NCNC to NC - default template
template<typename DType> void sfc_ca_gemm_matrix_copy_NCNC_to_NC(void *in, void *out, long N, long M, long bn, long bm) {
  DType *src_ptr = (DType*)in;
  DType *dst_ptr = (DType*)out;
  long nBlocks = N / bn;
  long mBlocks = M / bm;
  for (long n1 = 0; n1 < nBlocks; n1++) {
    for (long m1 = 0; m1 < mBlocks; m1++) {
      for (long n2 = 0; n2 < bn; n2++) {
        for (long m2 = 0; m2 < bm; m2++) {
          dst_ptr[(n1*bn+n2)*M + m1*bm+m2] = 
            src_ptr[n1*mBlocks*bn*bm + m1*bn*bm + n2*bm + m2];
        }
      }
    }
  }
}

template<> void sfc_ca_gemm_matrix_copy_NCNC_to_NC<float>(void *in, void *out, long N, long M, long bn, long bm) {
  float *src_ptr = (float*)in;
  float *dst_ptr = (float*)out;
  long nBlocks = N / bn;
  long mBlocks = M / bm;
  for (long n1 = 0; n1 < nBlocks; n1++) {
    for (long m1 = 0; m1 < mBlocks; m1++) {
      for (long n2 = 0; n2 < bn; n2++) {
        for (long m2 = 0; m2 < bm; m2++) {
          dst_ptr[(n1*bn+n2)*M + m1*bm+m2] = 
            src_ptr[n1*mBlocks*bn*bm + m1*bn*bm + n2*bm + m2];
        }
      }
    }
  }
}

template<> void sfc_ca_gemm_matrix_copy_NCNC_to_NC<libxsmm_bfloat16>(void *in, void *out, long N, long M, long bn, long bm) {
  libxsmm_bfloat16 *src_ptr = (libxsmm_bfloat16*)in;
  libxsmm_bfloat16 *dst_ptr = (libxsmm_bfloat16*)out;
  long nBlocks = N / bn;
  long mBlocks = M / bm;
  for (long n1 = 0; n1 < nBlocks; n1++) {
    for (long m1 = 0; m1 < mBlocks; m1++) {
      for (long n2 = 0; n2 < bn; n2++) {
        for (long m2 = 0; m2 < bm; m2++) {
          dst_ptr[(n1*bn+n2)*M + m1*bm+m2] = 
            src_ptr[n1*mBlocks*bn*bm + m1*bn*bm + n2*bm + m2];
        }
      }
    }
  }
}

template<> void sfc_ca_gemm_matrix_copy_NCNC_to_NC<libxsmm_bfloat8>(void *in, void *out, long N, long M, long bn, long bm) {
  libxsmm_bfloat8 *src_ptr = (libxsmm_bfloat8*)in;
  libxsmm_bfloat8 *dst_ptr = (libxsmm_bfloat8*)out;
  long nBlocks = N / bn;
  long mBlocks = M / bm;
  for (long n1 = 0; n1 < nBlocks; n1++) {
    for (long m1 = 0; m1 < mBlocks; m1++) {
      for (long n2 = 0; n2 < bn; n2++) {
        for (long m2 = 0; m2 < bm; m2++) {
          dst_ptr[(n1*bn+n2)*M + m1*bm+m2] = 
            src_ptr[n1*mBlocks*bn*bm + m1*bn*bm + n2*bm + m2];
        }
      }
    }
  }
}

template<> void sfc_ca_gemm_matrix_copy_NCNC_to_NC<double>(void *in, void *out, long N, long M, long bn, long bm) {
  double *src_ptr = (double*)in;
  double *dst_ptr = (double*)out;
  long nBlocks = N / bn;
  long mBlocks = M / bm;
  for (long n1 = 0; n1 < nBlocks; n1++) {
    for (long m1 = 0; m1 < mBlocks; m1++) {
      for (long n2 = 0; n2 < bn; n2++) {
        for (long m2 = 0; m2 < bm; m2++) {
          dst_ptr[(n1*bn+n2)*M + m1*bm+m2] = 
            src_ptr[n1*mBlocks*bn*bm + m1*bn*bm + n2*bm + m2];
        }
      }
    }
  }
}

template<typename DType> void sfc_ca_gemm_rne_convert_fp32_lp(float *in, void *out, long size) {
  libxsmm_rne_convert_fp32_bf16(in, (libxsmm_bfloat16*)out, size);
  return;
}

template<> void sfc_ca_gemm_rne_convert_fp32_lp<float>(float *in, void *out, long size) {
  memcpy(out, in, size * sizeof(float));
  return;
}

template<> void sfc_ca_gemm_rne_convert_fp32_lp<libxsmm_bfloat16>(float *in, void *out, long size) {
  libxsmm_rne_convert_fp32_bf16(in, (libxsmm_bfloat16*)out, size);
  return;
}

template<> void sfc_ca_gemm_rne_convert_fp32_lp<libxsmm_bfloat8>(float *in, void *out, long size) {
  libxsmm_rne_convert_fp32_bf8(in, (libxsmm_bfloat8*)out, size);
  return;
}

template<> void sfc_ca_gemm_rne_convert_fp32_lp<char>(float *in, void *out, long size) {
  char *out_i8 = (char*)out;
  for (long i = 0; i < size; i++) {
    // Clamp to INT8 range [-128, 127]
    float val = in[i];
    if (val > 127.0f) val = 127.0f;
    if (val < -128.0f) val = -128.0f;
    out_i8[i] = (char)(val);
  }
  return;
}

template<> void sfc_ca_gemm_rne_convert_fp32_lp<int>(float *in, void *out, long size) {
  int *out_i32 = (int*)out;
  for (long i = 0; i < size; i++) {
    out_i32[i] = (int)(in[i]);
  }
  return;
}

template<> void sfc_ca_gemm_rne_convert_fp32_lp<double>(float *in, void *out, long size) {
  double *out_f64 = (double*)out;
  for (long i = 0; i < size; i++) {
    out_f64[i] = (double)(in[i]);
  }
  return;
}

template<typename DType> void sfc_ca_gemm_convert_lp_f32(void *in, float *out, long size) {
  libxsmm_convert_bf16_f32((libxsmm_bfloat16*)in, out, size);
  return;
}

template<> void sfc_ca_gemm_convert_lp_f32<float>(void *in, float *out, long size) {
  memcpy(out, in, size * sizeof(float));
  return;
}

template<> void sfc_ca_gemm_convert_lp_f32<libxsmm_bfloat16>(void *in, float *out, long size) {
  libxsmm_convert_bf16_f32((libxsmm_bfloat16*)in, out, size);
  return;
}

template<> void sfc_ca_gemm_convert_lp_f32<libxsmm_bfloat8>(void *in, float *out, long size) {
  libxsmm_convert_bf8_f32((libxsmm_bfloat8*)in, out, size);
  return;
}

template<> void sfc_ca_gemm_convert_lp_f32<char>(void *in, float *out, long size) {
  char *in_i8 = (char*)in;
  for (long i = 0; i < size; i++) {
    out[i] = (float)in_i8[i];
  }
  return;
}

// Direct random initialization for I8 and I32 (no float conversion)
template<typename T>
void init_buf_int(T *buf, long size, int range_min, int range_max) {
  for (long i = 0; i < size; i++) {
    buf[i] = (T)(rand() % (range_max - range_min + 1) + range_min);
  }
}

template<> void sfc_ca_gemm_convert_lp_f32<int>(void *in, float *out, long size) {
  int *in_i32 = (int*)in;
  for (long i = 0; i < size; i++) {
    out[i] = (float)in_i32[i];
  }
  return;
}

template<> void sfc_ca_gemm_convert_lp_f32<double>(void *in, float *out, long size) {
  double *in_f64 = (double*)in;
  for (long i = 0; i < size; i++) {
    out[i] = (float)in_f64[i];
  }
  return;
}

template<typename DType> void sfc_ca_gemm_matrix_copy_KC_to_KCCK(void *src, void *dst, int C, int K, int bc, int bk)
{
  int kBlocks = K/bk;
  int cBlocks = C/bc;
  int vnni_block = libxsmm_cpuid_dot_pack_factor(sfc_ca_gemm_get_libxsmm_dtype<DType>());
  LIBXSMM_VLA_DECL(2, DType, real_src, (DType*)src, C);
  LIBXSMM_VLA_DECL(5, DType, real_dst, (DType*)dst, cBlocks, bc/vnni_block, bk, vnni_block);
  # pragma omp parallel for
  for (int k1 = 0; k1 < kBlocks; k1++) {
    for (int c1 = 0; c1 < cBlocks; c1++) {
      for (int c2 = 0; c2 < bc; c2++) {
        for (int k2 = 0; k2 < bk; k2++) {
          LIBXSMM_VLA_ACCESS(5, real_dst, k1, c1, c2/vnni_block, k2, c2%vnni_block, cBlocks, bc/vnni_block, bk, vnni_block) =
            LIBXSMM_VLA_ACCESS(2, real_src, k1*bk+k2, c1*bc+c2, C);
        }
      }
    }
  }
}

template<typename DType> void sfc_ca_gemm_matrix_copy_NC_to_NCNC(void *src, void *dst, int N, int C, int bn, int bc)
{
  int nBlocks = N/bn;
  int cBlocks = C/bc;
  LIBXSMM_VLA_DECL(3, DType, real_src, (DType*)src, N, C);
  LIBXSMM_VLA_DECL(5, DType, real_dst, (DType*)dst, nBlocks, cBlocks, bn, bc);
  # pragma omp parallel for 
  for (int n1 = 0; n1 < nBlocks; n1++) {
    for (int c1 = 0; c1 < cBlocks; c1++) {
      for (int n2 = 0; n2 < bn; n2++) {
        for (int c2 = 0; c2 < bc; c2++) {
          LIBXSMM_VLA_ACCESS(5, real_dst, 0, n1, c1, n2, c2, nBlocks, cBlocks, bn, bc) =
            LIBXSMM_VLA_ACCESS(3, real_src, 0, n1*bn+n2, c1*bc+c2, N, C);
        }
      }
    }
  }
}

unsigned int sfc_ca_gemm_fill_sfc_index_map(unsigned char **sfc_index_map, unsigned int Mb, unsigned int Nb, unsigned int m_step, unsigned int n_step) {
  unsigned int Mb_coarse = Mb / m_step, Nb_coarse = Nb / n_step;
  long long n_coarse_tasks = Mb_coarse * Nb_coarse;
  long long n_fine_tasks = (long long)Mb * Nb;
  int m_id, n_id;
  if (Mb < 256 && Nb < 256) {
    unsigned char *map;
    *sfc_index_map = (unsigned char*) libxsmm_aligned_malloc( 2*n_fine_tasks*sizeof(unsigned char), 2097152);
    map = (unsigned char*) *sfc_index_map;
    for (long long i = 0; i < n_coarse_tasks; i++) {
      gilbert_d2xy( &m_id, &n_id, i, Mb_coarse, Nb_coarse );
      for (unsigned int ms = 0; ms < m_step; ms++) {
        for (unsigned int ns = 0; ns < n_step; ns++) {
          long long fi = i * m_step * n_step + ms * n_step + ns;
          map[2*fi+0] = (unsigned char)(m_id * m_step + ms);
          map[2*fi+1] = (unsigned char)(n_id * n_step + ns);
        }
      }
    }
    return 1;
  } else if (Mb < 65536 && Nb < 65536) {
    unsigned short *map;
    *sfc_index_map = (unsigned char*) libxsmm_aligned_malloc( 2*n_fine_tasks*sizeof(unsigned short), 2097152);
    map = (unsigned short*) *sfc_index_map;
    for (long long i = 0; i < n_coarse_tasks; i++) {
      gilbert_d2xy( &m_id, &n_id, i, Mb_coarse, Nb_coarse );
      for (unsigned int ms = 0; ms < m_step; ms++) {
        for (unsigned int ns = 0; ns < n_step; ns++) {
          long long fi = i * m_step * n_step + ms * n_step + ns;
          map[2*fi+0] = (unsigned short)(m_id * m_step + ms);
          map[2*fi+1] = (unsigned short)(n_id * n_step + ns);
        }
      }
    }
    return 2;
  } else {
    unsigned int *map;
    *sfc_index_map = (unsigned char*) libxsmm_aligned_malloc( 2*n_fine_tasks*sizeof(unsigned int), 2097152);
    map = (unsigned int*) *sfc_index_map;
    for (long long i = 0; i < n_coarse_tasks; i++) {
      gilbert_d2xy( &m_id, &n_id, i, Mb_coarse, Nb_coarse );
      for (unsigned int ms = 0; ms < m_step; ms++) {
        for (unsigned int ns = 0; ns < n_step; ns++) {
          long long fi = i * m_step * n_step + ms * n_step + ns;
          map[2*fi+0] = (unsigned int)(m_id * m_step + ms);
          map[2*fi+1] = (unsigned int)(n_id * n_step + ns);
        }
      }
    }
    return 4;
  }
}

void sfc_ca_gemm_extract_indices_from_sfc(int *i_m, int *i_n, unsigned char *sfc_index_map, int sfc_index, unsigned int index_tsize) {
  if (index_tsize == 1) {
    unsigned char *map = (unsigned char*) sfc_index_map;
    *i_m = (int) map[2*sfc_index + 0];
    *i_n = (int) map[2*sfc_index + 1];
  } else if (index_tsize == 2) {
    unsigned short *map = (unsigned short*) sfc_index_map;
    *i_m = (int) map[2*sfc_index + 0];
    *i_n = (int) map[2*sfc_index + 1];
  } else {
    unsigned int *map = (unsigned int*) sfc_index_map;
    *i_m = (int) map[2*sfc_index + 0];
    *i_n = (int) map[2*sfc_index + 1];
  }
  return;
}

#endif // SFC_CA_GEMM_HPP