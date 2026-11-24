# SFC-CA GEMM sharer-aware performance model

`sfc_ca_perf_model` is a standalone analytical model (no LIBXSMM/OpenMP needed) that
predicts the runtime of SFC-CA GEMM for a given problem and platform, searches all
decompositions, and reports the best ones. It extends the BRGEMM-granular roofline of the
paper (Sec. III-B, `t_i = max(G*gamma, beta*S_i)`) and of `roofline_predictor.c`.

## What it models

1. **Exact decomposition.** For every `(K_layers, m_step, n_step)` it replays the production
   schedule of `run_gemm`: the coarsened Gilbert SFC map (`sfc_ca_gemm_fill_sfc_index_map`)
   and the OpenMP static block partition of the `Mb*Nb*K_layers` task loop (libgomp
   "balanced" or libomp "greedy"). This yields, per core, its exact C partition (tasks in
   execution order) and the A/B panels it reads. Threads may straddle K-layers, exactly as
   in the kernel. The footprint is computed once and reused for every `kbf`.
2. **Panel sharers.** For every A panel `(layer, im)` and B panel `(layer, in)` it counts the
   distinct cores that read it. A panel load uses the per-core bandwidth `beta(sharers)`
   from the user-provided table (log2-interpolated between points, clamped at the ends).
3. **Per-core cache.** Each core replays all K rounds of its tasks through a private,
   fully associative, byte-granular LRU cache (`--l2-kb`, 0 = infinite). This classifies
   every BRGEMM as BRGEMM0..3 (A/B from memory or fast memory) and detects C tiles evicted
   between K rounds.
4. **Cost.**
   - BRGEMM: `brc * max(bm*bn*bk / (MACs/cycle * eff), missA*S_A/beta(sA) + missB*S_B/beta(sB))`
   - C: final write-back per tile, plus write-back + re-read per C tile evicted between rounds
   - Reduction (`K_layers > 1`): `(K_layers + 1)` C tile transfers per reduced tile, split
     over cores as in the kernel's reduction loop
   - Total = slowest core's GEMM phase + slowest core's reduction phase
5. **Search.** `K_layers <= --kl-max`, `m_step, n_step <= --step-max` (dividing Mb/Nb),
   `kbf <= --kbf-max` (kbf values yielding the same brcount evaluated once). Reports the
   top-N by predicted time, with topology, sharer statistics and a run command for
   `sfc_ca_gemm`.

## Build

```
make sfc_ca_perf_model            # or: make sfc_ca_perf_model CXX=clang++
```

## Usage

```
./sfc_ca_perf_model M N K [options]
```

| Option | Meaning |
|---|---|
| `--threads T` | number of cores (default 64) |
| `--bw "s:B,..."` | per-core Bytes/cycle when a panel has `s` sharers, e.g. `"1:4.8,2:8.8,4:9.6,8:13.2,16:15.9,32:16.7"` |
| `--bw-flat B` | per-core Bytes/cycle regardless of sharers (instead of `--bw`) |
| `--c-bw B` | per-core Bytes/cycle for C writes/reduction (default `beta(1)`) |
| `--macs X` | BF16 MACs/cycle/core (1 MAC = 2 flops) |
| `--eff P` | achieved % of MAC peak inside BRGEMM |
| `--freq GHz` | only to report TFLOP/s |
| `--l2-kb KiB` | per-core fast memory, 0 = infinite (default 2048) |
| `--bm/--bn/--bk` | tile sizes (default 32) |
| `--kl-max`, `--step-max`, `--kbf-max` | search limits (default 16, 4, 8); `--step-max 1` forces `m_step = n_step = 1` |
| `--sched balanced\|greedy` | OpenMP static partition of libgomp (gcc) / libomp (clang, icx) |
| `--top N` | number of decompositions to report (default 5) |
| `--eval KL,ms,ns,kbf` | evaluate a single configuration |
| `--dump-cores` | with `--eval`: per-core A/B/C partitions |
| `--dump-map FILE` | map of the `--eval` (else rank-1) configuration: `*.svg` colored SVG, `-` ASCII to stdout, otherwise ASCII file |

Bandwidths are **per core in Bytes/cycle**; e.g. 700 GB/s at 3.2 GHz over 64 cores is
`700e9 / 3.2e9 / 64 = 3.42`.

### Examples

```
# 64 cores, 13 B/cycle flat, 512 flops/cycle/core
./sfc_ca_perf_model 4096 4096 4096 --threads 64 --bw-flat 13 --macs 256 --eff 100

# Sharer-dependent bandwidth, C bandwidth override, SVG of the best decomposition
./sfc_ca_perf_model 8192 8192 2048 --threads 64 \
  --bw "1:4.8,2:8.8,4:9.6,8:13.2,16:15.9,32:16.7" \
  --macs 256 --eff 95 --c-bw 3 --dump-map best.svg

# Single configuration with per-core partitions and ASCII map
./sfc_ca_perf_model 2048 2048 8192 --threads 64 --bw-flat 6 --macs 1024 \
  --eval 4,1,1,2 --dump-cores --dump-map -
```

## Output

- `patch`: per-core C partition shapes (A panels x B panels), most common first; `*` means
  some core does not own one contiguous rectangle in a single layer.
- `grid`: `K_layers x core-rows x core-cols` when all patches are equal rectangles
  (e.g. `4x4x4`), else `rect-mixed` / `irregular`.
- `A-sh` / `B-sh`: cores sharing an A/B panel (avg / min-max); `BW A/B`: effective
  Bytes/cycle of the critical core.
- Critical-core breakdown: compute, memory stall, C I/O and reduction cycles.
- SVG map: C ownership per core; A (left) and B (top) panels colored by the set of cores
  reading them and labeled with `#sharers {cores}`; dashed lines mark kbf K-rounds.

## Known simplifications

- Fully associative LRU L2, byte-granular objects; no shared LLC level, no prefetching.
- No overlap of memory and compute across BRGEMMs; C traffic is additive.
- Panel reloads after L2 eviction use the same `beta(sharers)` as first loads.
- No write-allocate (RFO) on C stores, no NT-store, B-packing (`unblocked_bc=2`) or
  dequantization modeling; flat and blocked layouts are treated alike.
- No platform topology (dies/CCX/NUMA): bandwidth depends on the sharer count only.
