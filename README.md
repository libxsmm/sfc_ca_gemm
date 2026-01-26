# SFC Communication-Avoiding GEMM

A minimal, standalone implementation of space-filling curve (SFC) based communication-avoiding (CA) GEMM. For more details, refer to the associated [paper](https://arxiv.org/abs/2601.16294).

## Dependencies

- **LIBXSMM** (https://github.com/libxsmm/libxsmm)
- OpenMP (for parallel execution)
- C++14 compatible compiler (gcc, clang, icc, or icx)

## Quick Start (Recommended)

Use the provided script to automatically download and build LIBXSMM:

```bash
# Download and build LIBXSMM (one-time setup)
./prepare_libxsmm.sh

# Build the benchmark
make

# Run a test
./sfc_ca_gemm
```

## Building with Different Compilers

The Makefile supports multiple compilers via the `SFC_CA_GEMM_COMPILER` environment variable:

### GCC (default)
```bash
make
# or explicitly:
SFC_CA_GEMM_COMPILER=gcc make
```

### Clang
```bash
SFC_CA_GEMM_COMPILER=clang make
```

### Intel oneAPI Compiler (icx)
```bash
SFC_CA_GEMM_COMPILER=icx make
```

## Manual Build (if LIBXSMM is already installed)

If you have LIBXSMM installed elsewhere:

```bash
# Set LIBXSMM_ROOT to your installation
export LIBXSMM_ROOT=/path/to/libxsmm
make
```

## Usage

```bash
./sfc_ca_gemm <M> <N> <K> <bm> <bn> <bk> [kbf] [K_layers] [n_layers] [n_iters] [check] [dtype]
```

### Parameters

- `M, N, K`: Matrix dimensions (C = A × B, where A is M×K, B is K×N, C is M×N)
- `bm, bn, bk`: Block sizes for A/B/C tiling
- `kbf`: (optional) K-blocking factor
- `K_layers`: (optional) Replication factor for 2.5D and 3D GEMM configs (default: 1)
- `n_layers`: (optional) Number of GEMM layers to execute (default: 1, use -1 for auto-sizing to ~5GB)
- `n_iters`: (optional) Number of benchmark iterations (default: 1)
- `check`: (optional) Enable correctness checking (0=off, 1=on, default: 0)
- `dtype`: (optional) Data type - "BF16" (default), "BF8", or "FP32"

## Output
The benchmark reports:
- **GFLOPS achieved** - Floating point operations per second
- **Execution time** - Time per iteration in milliseconds
- **Memory footprint** - Effective model size and total GEMM size in GB
- **Bandwidth utilization** - Effective memory bandwidth for matrix A in GB/s
- **MEASURE line** - Summary with all configuration parameters for easy parsing

## License

BSD-3-Clause (see file headers)
