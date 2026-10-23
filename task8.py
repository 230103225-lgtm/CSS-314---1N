import os
import time
import subprocess

import numpy as np
from numba import njit, prange, set_num_threads, get_num_threads

STUDENT_ID = "230103225"
N = 1024
BLOCKS = [16, 64]
REPEATS = 3


@njit(parallel=True, fastmath=False)
def matmul_naive(A, B, C, n):
    for i in prange(n):
        for j in range(n):
            s = 0.0
            for k in range(n):
                s += A[i, k] * B[k, j]
            C[i, j] = s


@njit(parallel=True, fastmath=False)
def matmul_tiled(A, B, C, n, bs):
    n_tiles = (n + bs - 1) // bs
    for t in prange(n_tiles):
        ii = t * bs
        i_end = min(ii + bs, n)
        for i in range(ii, i_end):
            for j in range(n):
                C[i, j] = 0.0
        for kk in range(0, n, bs):
            k_end = min(kk + bs, n)
            for jj in range(0, n, bs):
                j_end = min(jj + bs, n)
                for i in range(ii, i_end):
                    for k in range(kk, k_end):
                        a = A[i, k]
                        for j in range(jj, j_end):
                            C[i, j] += a * B[k, j]


def sysctl(key):
    try:
        return int(subprocess.check_output(["sysctl", "-n", key], stderr=subprocess.DEVNULL).decode().strip())
    except Exception:
        return None


def cache_info():
    info = {}
    for label, keys in [
        ("L1 data cache", ["hw.perflevel0.l1dcachesize", "hw.l1dcachesize"]),
        ("L2 cache", ["hw.perflevel0.l2cachesize", "hw.l2cachesize"]),
        ("Cache line", ["hw.cachelinesize"]),
    ]:
        for k in keys:
            v = sysctl(k)
            if v:
                info[label] = v
                break
    return info


def bench(fn, *args):
    best = float("inf")
    for _ in range(REPEATS):
        t0 = time.perf_counter()
        fn(*args)
        best = min(best, (time.perf_counter() - t0) * 1000)
    return best


def run_all(A, B, ref, threads):
    set_num_threads(threads)
    rows = []

    C = np.zeros((N, N))
    t = bench(matmul_naive, A, B, C, N)
    rows.append(("Non-tiled (i-j-k)", t, np.allclose(C, ref)))

    for bs in BLOCKS:
        C = np.zeros((N, N))
        t = bench(matmul_tiled, A, B, C, N, bs)
        rows.append((f"Tiled {bs}x{bs}", t, np.allclose(C, ref)))
    return rows


def print_table(rows, threads):
    base = rows[0][1]
    best = min(r[1] for r in rows)
    print(f"{'Variant':<20} | {'Runtime (ms)':>12} | {'GFLOP/s':>8} | {'vs non-tiled':>12} | {'vs best':>8} | OK")
    print("-" * 78)
    flops = 2.0 * N ** 3
    for name, t, ok in rows:
        print(f"{name:<20} | {t:>12.2f} | {flops / (t / 1000) / 1e9:>8.2f} | "
              f"{base / t:>11.2f}x | {t / best:>7.2f}x | {'OK' if ok else 'FAIL'}")


def main():
    max_threads = os.cpu_count()
    print(f"Student ID: {STUDENT_ID}")
    print(f"Matrix size: {N}x{N} float64 ({N * N * 8 / 1e6:.1f} MB per matrix)")
    print(f"Logical cores (threads used): {max_threads}")
    print(f"Timing: best of {REPEATS} runs, JIT compilation excluded")
    ci = cache_info()
    for k, v in ci.items():
        print(f"{k}: {v:,} bytes")
    for bs in BLOCKS:
        print(f"Tile {bs}x{bs}: working set of 3 tiles = {3 * bs * bs * 8 / 1024:.0f} KB")
    print()

    rng = np.random.default_rng(int(STUDENT_ID))
    A = rng.random((N, N))
    B = rng.random((N, N))
    ref = A @ B

    a_s, b_s, c_s = A[:64, :64].copy(), B[:64, :64].copy(), np.zeros((64, 64))
    matmul_naive(a_s, b_s, c_s, 64)
    matmul_tiled(a_s, b_s, c_s, 64, 16)

    thread_list = [1, max_threads] if max_threads > 1 else [1]
    all_rows = {}
    for th in thread_list:
        print(f"=== {th} thread(s) ===")
        rows = run_all(A, B, ref, th)
        print_table(rows, th)
        all_rows[th] = rows
        print()

    print("=== CONCLUSION ===")
    rows = all_rows[max_threads]
    naive_t = rows[0][1]
    tile_ts = {r[0]: r[1] for r in rows[1:]}
    best_name = min(tile_ts, key=tile_ts.get)
    print(f"At {get_num_threads()} threads: non-tiled = {naive_t:.2f} ms, "
          f"best tiled ({best_name}) = {tile_ts[best_name]:.2f} ms "
          f"-> {naive_t / tile_ts[best_name]:.2f}x faster.")
    print("Non-tiled version walks column B[k, j] with a 8 KB stride, so almost every "
          "access loads a new cache line (high cache miss rate).")
    print("Tiled versions reuse a small block of A, B, C while it is still in L1/L2 cache, "
          "so the runtime difference is a direct proxy for the cache-miss latency saved.")
    print("Note: macOS does not expose hardware cache-miss counters without Instruments, "
          "so cache-miss cost is reported as runtime difference.")


if __name__ == "__main__":
    main()