#!/usr/bin/env python3
"""SpMV accelerator regression + sparsity-aware format advisor.

For a set of sparsity patterns (uniform at several densities, power-law row
lengths, banded, block-diagonal, many empty rows, extreme values):
  * y from the RTL must equal numpy int64 A @ x exactly
  * the RTL cycle count must equal the analytical CSR cost model exactly
  * the advisor compares that model with a dense GEMV engine of the same MAC
    count and picks the faster representation for the matrix

Usage: run_checks.py <build dir> <compiled tb.vvp> <P lanes>
"""
import math
import subprocess
import sys

import numpy as np

ROWS, COLS = 256, 256


def csr_cycles(row_nnz, p):
    """Cost model of the RTL: each row takes max(1, ceil(nnz/P)) clocks."""
    return int(sum(max(1, math.ceil(n / p)) for n in row_nnz))


def dense_cycles(rows, cols, p):
    """Dense GEMV on the same P MACs: every row streams all columns."""
    return rows * math.ceil(cols / p)


def make(kind, rng):
    a = np.zeros((ROWS, COLS), dtype=np.int64)
    if kind.startswith("uniform"):
        d = float(kind.split("_")[1]) / 100
        mask = rng.random((ROWS, COLS)) < d
    elif kind == "powerlaw":                       # a few very long rows, many short ones (graph-like)
        mask = np.zeros((ROWS, COLS), bool)
        for r in range(ROWS):
            n = min(COLS, int(rng.pareto(1.2) * 2) + 1)
            mask[r, rng.choice(COLS, n, replace=False)] = True
    elif kind == "banded":                          # e.g. a finite-difference stencil
        mask = np.zeros((ROWS, COLS), bool)
        for r in range(ROWS):
            mask[r, max(0, r - 3):min(COLS, r + 4)] = True
    elif kind == "blockdiag":                       # e.g. a batched / grouped layer
        mask = np.zeros((ROWS, COLS), bool)
        for b in range(0, ROWS, 16):
            mask[b:b + 16, b:b + 16] = rng.random((16, 16)) < 0.6
    elif kind == "empty_rows":
        mask = rng.random((ROWS, COLS)) < 0.05
        mask[rng.random(ROWS) < 0.5] = False
    else:                                           # "extremes": dense-ish, full-scale values
        mask = rng.random((ROWS, COLS)) < 0.5
    vals = rng.integers(-32768, 32768, (ROWS, COLS)) if kind == "extremes" else rng.integers(-2000, 2000, (ROWS, COLS))
    a[mask] = vals[mask]
    a[mask & (a == 0)] = 1                          # keep the pattern exact
    x = rng.integers(-32768, 32768, COLS) if kind == "extremes" else rng.integers(-1000, 1000, COLS)
    return a, x


def to_csr(a):
    row_ptr, cols, vals = [0], [], []
    for r in range(a.shape[0]):
        nz = np.nonzero(a[r])[0]
        cols += nz.tolist()
        vals += a[r, nz].tolist()
        row_ptr.append(len(cols))
    return row_ptr, cols, vals


def main(build, vvp, p):
    rng = np.random.default_rng(11)
    kinds = ["uniform_1", "uniform_5", "uniform_20", "uniform_60", "powerlaw", "banded", "blockdiag",
             "empty_rows", "extremes"]
    fails = 0
    print(f"SpMV CSR accelerator, P={p} lanes, {ROWS}x{COLS} matrices")
    print(f"  {'pattern':11s} {'nnz':>6s} {'density':>8s} {'cycles':>7s} {'model':>7s} {'MAC/clk':>8s} "
          f"{'lane util':>9s} {'dense GEMV':>10s} {'advisor':>8s} {'speed-up':>8s}  result")
    for k in kinds:
        a, x = make(k, rng)
        row_ptr, cols, vals = to_csr(a)
        nnz = len(cols)
        with open(f"{build}/m.txt", "w") as f:
            f.write(f"{ROWS} {nnz} {COLS}\n")
            for seq in (row_ptr, cols, vals, x.tolist()):
                f.write(" ".join(map(str, seq)) + "\n")
        subprocess.run(["vvp", "-n", vvp, f"+in={build}/m.txt", f"+out={build}/y.txt"], check=True,
                       stdout=subprocess.DEVNULL)
        lines = open(f"{build}/y.txt").read().split()
        cyc, macs = int(lines[0]), int(lines[1])
        y = np.array([int(v) for v in lines[2:]], dtype=np.int64)
        exp = a @ x
        row_nnz = np.diff(row_ptr)
        model = csr_cycles(row_nnz, p)
        dense = dense_cycles(ROWS, COLS, p)
        ok = np.array_equal(y, exp) and cyc == model and macs == nnz
        fails += not ok
        pick = "CSR" if model < dense else "dense"
        print(f"  {k:11s} {nnz:6d} {nnz / (ROWS * COLS):8.3f} {cyc:7d} {model:7d} {nnz / cyc:8.2f} "
              f"{nnz / (cyc * p):9.1%} {dense:10d} {pick:>8s} {dense / model:7.1f}x  {'ok' if ok else 'FAIL'}")
        if not np.array_equal(y, exp):
            bad = np.nonzero(y != exp)[0]
            print(f"    y mismatch in {len(bad)} rows, first row {bad[0]}: got {y[bad[0]]} expected {exp[bad[0]]}")
    print("TEST PASSED" if fails == 0 else f"TEST FAILED ({fails} patterns)")
    sys.exit(1 if fails else 0)


if __name__ == "__main__":
    main(sys.argv[1], sys.argv[2], int(sys.argv[3]))
