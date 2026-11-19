# Sparse matrix-vector (SpMV) accelerator with a format advisor

[![CI](https://github.com/Ana2601/sparse-spmv-accelerator/actions/workflows/ci.yml/badge.svg)](https://github.com/Ana2601/sparse-spmv-accelerator/actions/workflows/ci.yml)

A CSR sparse matrix × dense vector engine with P parallel MAC lanes, plus a sparsity-aware cost model whose predictions are validated **cycle-exactly** against the RTL. This is the hardware follow-up to the MatAnalyze sparse-format research (CHIPS Lab).

## Design (`rtl/spmv_csr.sv`)

- y = A·x with INT16 values and vector, **exact** 48-bit accumulation
- P lanes consume P non-zeros of the current row per clock, reduced by an adder tree
- A row costs max(1, ⌈nnz/P⌉) clocks (empty rows cost 1)
- On-chip row_ptr / col_idx / values / x buffers (up to 64k non-zeros), y read port
- Performance counters: busy cycles and MACs, giving lane utilisation per matrix

## Verification and advisor (`scripts/run_checks.py`)

For each sparsity pattern:

- y from the RTL must equal numpy int64 `A @ x` **exactly**
- the RTL cycle count must equal the analytical model Σ max(1, ⌈nnz_r/P⌉) **exactly**
- the advisor compares that model with a dense GEMV on the same P MACs

Patterns: uniform 1 % / 5 % / 20 % / 60 %, power-law rows (graph-like), banded (stencil), block-diagonal, 50 % empty rows, and full-scale extreme values.

## Results (measured, 256×256, P = 4)

| Pattern | nnz | Cycles (RTL = model) | MAC/clk | Lane util | vs. dense GEMV |
|---|---|---|---|---|---|
| uniform 1 % | 668 | 283 | 2.36 | 59 % | 57.9× faster |
| uniform 5 % | 3,185 | 887 | 3.59 | 90 % | 18.5× |
| uniform 20 % | 13,174 | 3,386 | 3.89 | 97 % | 4.8× |
| uniform 60 % | 39,462 | 9,949 | 3.97 | 99 % | 1.6× |
| power-law | 1,975 | 625 | 3.16 | 79 % | 26.2× |
| banded | 1,780 | 510 | 3.49 | 87 % | 32.1× |
| block-diag | 2,507 | 714 | 3.51 | 88 % | 22.9× |
| empty rows | 1,652 | 590 | 2.80 | 70 % | 27.8× |
| extremes | 32,990 | 8,340 | 3.96 | 99 % | 2.0× |

All 9 patterns pass at P = 4 and at P = 8. Lane utilisation drops for very sparse matrices and power-law rows (short rows waste lanes). That is exactly the effect a format choice (ELL, merged-row CSR) addresses.

## Run

```bash
make test         # P=4
make test P=8
make lint
```

## Limitations / next steps

Short rows waste lanes: a merge-based (row-splitting) scheduler or an ELL/SELL-C-σ mode would raise utilisation on power-law matrices. The model then extends to choose between formats, not just CSR vs. dense. x is held on chip (no streaming x for very wide matrices).

## Setup

- [OSS CAD Suite](https://github.com/YosysHQ/oss-cad-suite-build) (Icarus Verilog, Verilator, Yosys)
- Python packages: `pip install -r requirements.txt`

```bash
make test
make lint
```

The same commands run in CI on every push (`.github/workflows/ci.yml`).
