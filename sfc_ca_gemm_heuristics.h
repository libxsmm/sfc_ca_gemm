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

#ifndef SFC_CA_GEMM_HEURISTICS_H
#define SFC_CA_GEMM_HEURISTICS_H

#include <math.h>

/*
 * Heuristics for selecting SFC CA GEMM tuning parameters:
 *   kbf, K_layers, m_step, n_step, flat
 *
 * given problem dimensions M, N, K and tile sizes bm, bn, bk.
 *
 * Heuristic summary (GeoMean % of oracle, GeoMean speedup vs oneDNN):
 *
 *   H1: Constant baseline            60.7%  0.89x
 *   H2: K-only scaling               80.2%  1.17x
 *   H3: Area-aware formula           83.9%  1.23x
 *   H4: Piecewise threshold          85.0%  1.24x
 *   H5: Global mode (static)         62.1%  0.91x
 *   H6: K/MN ratio + step by M≷N    81.5%  1.19x
 *   H7: Step by M≷N, KL from logK   81.8%  1.20x
 *   H8: Hand-crafted rules           84.6%  1.24x
 *   H9: Size-bin lookup table        95.3%  1.39x  <-- best automatic
 *  H10: Per-param mode by size-bin   93.7%  1.37x
 *  Oracle                           100.0%  1.46x
 */

/* ── Step Heuristic (shared by H1–H4) ──────────────────────────────── */

static inline void sfc_heuristic_steps(long M, long N,
                                       long *m_step, long *n_step)
{
    if (M < N) {
        *m_step = 2;
        *n_step = 1;
    } else {
        *m_step = 1;
        *n_step = 2;
    }
}

/* ── H1: Constant Baseline (0.89x vs oneDNN) ────────────────────────── */
/*
 * kbf=1, K_layers=1, flat=2, steps from M≷N
 */
static inline void sfc_heuristic_h1(long M, long N, long K,
                                    long bm, long bn, long bk,
                                    long *kbf, long *K_layers,
                                    long *m_step, long *n_step,
                                    long *flat)
{
    (void)K; (void)bm; (void)bn; (void)bk;
    sfc_heuristic_steps(M, N, m_step, n_step);
    *kbf      = 1;
    *K_layers = 1;
    *flat     = 2;
}

/* ── H2: K-only Scaling (1.17x vs oneDNN) ───────────────────────────── */
/*
 * kbf=1, K_layers = clamp(round(K/1024), 1, 8), flat=2, steps from M≷N
 */
static inline void sfc_heuristic_h2(long M, long N, long K,
                                    long bm, long bn, long bk,
                                    long *kbf, long *K_layers,
                                    long *m_step, long *n_step,
                                    long *flat)
{
    long Kl;
    (void)bm; (void)bn; (void)bk;
    sfc_heuristic_steps(M, N, m_step, n_step);

    Kl = (K + 512) / 1024;   /* round(K / 1024) */
    if (Kl < 1) Kl = 1;
    if (Kl > 8) Kl = 8;

    *kbf      = 1;
    *K_layers = Kl;
    *flat     = 2;
}

/* ── H3: Area-aware Formula (1.23x vs oneDNN) ───────────────────────── */
/*
 * kbf=1, K_layers = clamp(round(K_tiles / (MN_tiles^0.25 * 3.5)), 1, 8)
 * flat=2, steps from M≷N
 */
static inline void sfc_heuristic_h3(long M, long N, long K,
                                    long bm, long bn, long bk,
                                    long *kbf, long *K_layers,
                                    long *m_step, long *n_step,
                                    long *flat)
{
    long Kl;
    double K_tiles  = (double)K / (double)bk;
    double MN_tiles = ((double)M / (double)bm) * ((double)N / (double)bn);
    double val;

    sfc_heuristic_steps(M, N, m_step, n_step);

    val = K_tiles / (pow(MN_tiles, 0.25) * 3.5);
    Kl  = (long)(val + 0.5);   /* round */
    if (Kl < 1) Kl = 1;
    if (Kl > 8) Kl = 8;

    *kbf      = 1;
    *K_layers = Kl;
    *flat     = 2;
}

/* ── H4: Piecewise Threshold (1.24x vs oneDNN) ──────────────────────── */
/*
 * kbf=1, flat=2, steps from M≷N
 * K_layers from piecewise rules on K_tiles and MN_tiles.
 */
static inline void sfc_heuristic_h4(long M, long N, long K,
                                    long bm, long bn, long bk,
                                    long *kbf, long *K_layers,
                                    long *m_step, long *n_step,
                                    long *flat)
{
    long Kl;
    long K_tiles  = K / bk;
    long MN_tiles = (M / bm) * (N / bn);

    sfc_heuristic_steps(M, N, m_step, n_step);

    if (K_tiles <= 16) {
        Kl = 1;
    } else if (MN_tiles <= 1024) {
        Kl = (K_tiles + 3) / 4;
    } else if (MN_tiles <= 8192) {
        Kl = (K_tiles + 31) / 32;
    } else {
        Kl = (K_tiles + 63) / 64;
    }
    if (Kl < 1) Kl = 1;
    if (Kl > 8) Kl = 8;

    *kbf      = 1;
    *K_layers = Kl;
    *flat     = 2;
}

/* ── H5: Global Mode / static (0.91x vs oneDNN) ─────────────────────── */
/*
 * Most frequently optimal single config across all 125 M/N/K combos.
 * kbf=1, K_layers=1, stepM=2, stepN=1, flat=2
 */
static inline void sfc_heuristic_h5(long M, long N, long K,
                                    long bm, long bn, long bk,
                                    long *kbf, long *K_layers,
                                    long *m_step, long *n_step,
                                    long *flat)
{
    (void)M; (void)N; (void)K; (void)bm; (void)bn; (void)bk;
    *kbf      = 1;
    *K_layers = 1;
    *m_step   = 2;
    *n_step   = 1;
    *flat     = 2;
}

/* ── H6: K/MN ratio + step by M≷N (1.19x vs oneDNN) ─────────────────── */
/*
 * flat=2 if N>=M else 1; steps from M≷N; K_layers from K/max(M,N).
 */
static inline void sfc_heuristic_h6(long M, long N, long K,
                                    long bm, long bn, long bk,
                                    long *kbf, long *K_layers,
                                    long *m_step, long *n_step,
                                    long *flat)
{
    long Kl;
    double ratio;
    (void)bm; (void)bn; (void)bk;

    *flat = (N >= M) ? 2 : 1;
    if (M < N) {
        *m_step = 2; *n_step = 1;
    } else {
        *m_step = 1; *n_step = 2;
    }

    ratio = (double)K / (double)(M > N ? M : N);
    if (ratio >= 4.0)      Kl = 8;
    else if (ratio >= 2.0) Kl = 6;
    else if (ratio >= 1.0) Kl = 3;
    else                   Kl = 1;

    *kbf      = 1;
    *K_layers = Kl;
}

/* ── H7: Step by M≷N, KL from log₂K (1.20x vs oneDNN) ───────────────── */
/*
 * M>=N: stepM=1, stepN=1, flat=1
 * M<N:  stepM=2, stepN=1, flat=2
 * K_layers = clamp(round(log2(K/512) + 1), 1, 9) for K>512, else 1.
 */
static inline void sfc_heuristic_h7(long M, long N, long K,
                                    long bm, long bn, long bk,
                                    long *kbf, long *K_layers,
                                    long *m_step, long *n_step,
                                    long *flat)
{
    long Kl;
    (void)bm; (void)bn; (void)bk;

    if (M >= N) {
        *m_step = 1; *n_step = 1; *flat = 1;
    } else {
        *m_step = 2; *n_step = 1; *flat = 2;
    }

    if (K > 512) {
        double v = log2((double)K / 512.0) + 1.0;
        Kl = (long)(v + 0.5);
        if (Kl < 1) Kl = 1;
        if (Kl > 9) Kl = 9;
    } else {
        Kl = 1;
    }

    *kbf      = 1;
    *K_layers = Kl;
}

/* ── H8: Hand-crafted rules (1.24x vs oneDNN) ───────────────────────── */
/*
 * kbf=1; K_layers from conditional rules on M and K;
 * steps and flat depend on M vs N aspect ratio.
 */
static inline void sfc_heuristic_h8(long M, long N, long K,
                                    long bm, long bn, long bk,
                                    long *kbf, long *K_layers,
                                    long *m_step, long *n_step,
                                    long *flat)
{
    long Kl;
    (void)bm; (void)bn; (void)bk;

    /* K_layers from M and K */
    if (K >= 4096) {
        Kl = (M >= 4096) ? 3 : 8;
    } else if (K >= 2048) {
        Kl = (M <= 1024) ? 6 : 3;
    } else if (K >= 1024) {
        Kl = (M <= 1024) ? 4 : 1;
    } else {
        Kl = 1;
    }

    /* steps & flat from M vs N */
    if (M >= N) {
        *m_step = 1;
        *n_step = (N < M) ? 2 : 1;
        *flat   = 1;
    } else {
        *m_step = 2;
        *n_step = 1;
        *flat   = 2;
    }

    /* override for very large N >> M */
    if (N >= 4 * M) {
        *m_step = 2;
        *n_step = 1;
        *flat   = (N > M) ? 2 : 1;
    }

    *kbf      = 1;
    *K_layers = Kl;
}

/* ── H9: Size-bin Lookup Table (1.39x vs oneDNN) — BEST HEURISTIC ─── */
/*
 * Classifies M, N, K into size bins (S/M/L) and looks up the best
 * configuration from a 27-entry table trained on exhaustive tuning data.
 *
 * Bins: S = {512, 1024}, M = {2048}, L = {4096, 8192}
 * For sizes outside these, uses nearest-match binning.
 *
 * Achieves 95.3% geomean of oracle-optimal GFLOPS.
 * 83/125 configs within 5% of optimal.
 */

/* Size-bin classification: 0=S, 1=M, 2=L */
static inline int sfc_size_bin(long dim)
{
    if (dim <= 1024) return 0;  /* S */
    if (dim <= 2048) return 1;  /* M */
    return 2;                   /* L */
}

static inline void sfc_heuristic_h9(long M, long N, long K,
                                    long bm, long bn, long bk,
                                    long *kbf, long *K_layers,
                                    long *m_step, long *n_step,
                                    long *flat)
{
    /*
     * Lookup table: [bin_M][bin_N][bin_K] -> {kbf, K_layers, stepM, stepN, flat}
     * Indices: 0=S, 1=M, 2=L
     * Trained on exhaustive sweep: 3 kbf × 9 K_layers × 3 step combos × 2 flat
     * across 125 M/N/K combinations (512..8192).
     */
    static const int LUT[3][3][3][5] = {
        /* bin_M = S (512, 1024) */
        {
            /* bin_N = S */
            { /* bin_K=S */ {1, 3, 2, 1, 2}, /* bin_K=M */ {1, 8, 2, 1, 2}, /* bin_K=L */ {1, 8, 2, 1, 2} },
            /* bin_N = M */
            { /* bin_K=S */ {1, 4, 2, 1, 2}, /* bin_K=M */ {1, 8, 2, 1, 2}, /* bin_K=L */ {1, 8, 2, 1, 2} },
            /* bin_N = L */
            { /* bin_K=S */ {1, 1, 2, 1, 2}, /* bin_K=M */ {1, 4, 2, 1, 2}, /* bin_K=L */ {1, 4, 2, 1, 1} },
        },
        /* bin_M = M (2048) */
        {
            /* bin_N = S */
            { /* bin_K=S */ {1, 8, 1, 2, 1}, /* bin_K=M */ {1, 6, 1, 2, 2}, /* bin_K=L */ {1, 6, 1, 2, 2} },
            /* bin_N = M */
            { /* bin_K=S */ {2, 1, 1, 1, 1}, /* bin_K=M */ {1, 5, 2, 1, 2}, /* bin_K=L */ {1, 3, 2, 1, 2} },
            /* bin_N = L */
            { /* bin_K=S */ {1, 1, 2, 1, 2}, /* bin_K=M */ {1, 2, 2, 1, 2}, /* bin_K=L */ {1, 6, 2, 1, 2} },
        },
        /* bin_M = L (4096, 8192) */
        {
            /* bin_N = S */
            { /* bin_K=S */ {1, 1, 1, 2, 1}, /* bin_K=M */ {1, 3, 1, 2, 1}, /* bin_K=L */ {1, 6, 1, 1, 2} },
            /* bin_N = M */
            { /* bin_K=S */ {2, 1, 1, 2, 1}, /* bin_K=M */ {1, 3, 2, 1, 2}, /* bin_K=L */ {1, 3, 1, 1, 2} },
            /* bin_N = L */
            { /* bin_K=S */ {1, 1, 1, 1, 1}, /* bin_K=M */ {1, 1, 2, 1, 2}, /* bin_K=L */ {1, 3, 2, 1, 2} },
        },
    };

    int bM, bN, bK;
    (void)bm; (void)bn; (void)bk;

    bM = sfc_size_bin(M);
    bN = sfc_size_bin(N);
    bK = sfc_size_bin(K);

    *kbf      = LUT[bM][bN][bK][0];
    *K_layers = LUT[bM][bN][bK][1];
    *m_step   = LUT[bM][bN][bK][2];
    *n_step   = LUT[bM][bN][bK][3];
    *flat     = LUT[bM][bN][bK][4];
}

/* ── H10: Per-param mode by size-bin (1.37x vs oneDNN) ───────────────── */
/*
 * For each parameter independently, picks the value that most often
 * wins within the size bin. Less coupled than H9 but slightly weaker.
 */
static inline void sfc_heuristic_h10(long M, long N, long K,
                                     long bm, long bn, long bk,
                                     long *kbf, long *K_layers,
                                     long *m_step, long *n_step,
                                     long *flat)
{
    /* Table: [bin_M][bin_N][bin_K] -> {kbf, K_layers, stepM, stepN, flat}
     * Each param chosen independently as the mode of winners in that bin. */
    static const int LUT[3][3][3][5] = {
        /* bin_M = S */
        {
            /* bin_N = S */
            { {1, 8, 2, 1, 2}, {1, 8, 2, 1, 2}, {1, 8, 2, 1, 2} },
            /* bin_N = M */
            { {1, 4, 2, 1, 2}, {1, 8, 2, 1, 2}, {1, 8, 2, 1, 2} },
            /* bin_N = L */
            { {1, 1, 2, 1, 2}, {1, 1, 2, 1, 2}, {1, 7, 2, 1, 1} },
        },
        /* bin_M = M */
        {
            /* bin_N = S */
            { {1, 1, 1, 2, 2}, {1, 6, 1, 2, 2}, {1, 9, 1, 2, 2} },
            /* bin_N = M */
            { {2, 1, 1, 1, 1}, {1, 3, 2, 1, 2}, {1, 8, 2, 1, 2} },
            /* bin_N = L */
            { {1, 1, 2, 1, 2}, {1, 1, 2, 1, 2}, {1, 6, 2, 1, 2} },
        },
        /* bin_M = L */
        {
            /* bin_N = S */
            { {1, 1, 1, 2, 1}, {1, 3, 1, 2, 1}, {1, 6, 1, 1, 2} },
            /* bin_N = M */
            { {2, 1, 1, 2, 1}, {1, 3, 1, 1, 2}, {1, 3, 1, 1, 2} },
            /* bin_N = L */
            { {1, 1, 1, 1, 1}, {1, 1, 2, 1, 2}, {1, 3, 2, 1, 2} },
        },
    };

    int bM, bN, bK;
    (void)bm; (void)bn; (void)bk;

    bM = sfc_size_bin(M);
    bN = sfc_size_bin(N);
    bK = sfc_size_bin(K);

    *kbf      = LUT[bM][bN][bK][0];
    *K_layers = LUT[bM][bN][bK][1];
    *m_step   = LUT[bM][bN][bK][2];
    *n_step   = LUT[bM][bN][bK][3];
    *flat     = LUT[bM][bN][bK][4];
}

/* ── Ensemble: Best of H2/H3/H4 ────────────────────────────────────── */
static inline void sfc_heuristic_ensemble_candidates(
    long M, long N, long K,
    long bm, long bn, long bk,
    long *kbf, long candidates[3], int *n_candidates,
    long *m_step, long *n_step, long *flat)
{
    long kbf_tmp, Kl_h2, Kl_h3, Kl_h4, flat_tmp;
    long ms_tmp, ns_tmp;
    int n = 0;

    sfc_heuristic_h2(M, N, K, bm, bn, bk, &kbf_tmp, &Kl_h2, &ms_tmp, &ns_tmp, &flat_tmp);
    sfc_heuristic_h3(M, N, K, bm, bn, bk, &kbf_tmp, &Kl_h3, &ms_tmp, &ns_tmp, &flat_tmp);
    sfc_heuristic_h4(M, N, K, bm, bn, bk, &kbf_tmp, &Kl_h4, &ms_tmp, &ns_tmp, &flat_tmp);

    sfc_heuristic_steps(M, N, m_step, n_step);
    *kbf  = 1;
    *flat = 2;

    candidates[n++] = Kl_h2;
    if (Kl_h3 != Kl_h2) {
        candidates[n++] = Kl_h3;
    }
    if (Kl_h4 != Kl_h2 && Kl_h4 != Kl_h3) {
        candidates[n++] = Kl_h4;
    }
    *n_candidates = n;
}

#endif /* SFC_CA_GEMM_HEURISTICS_H */
