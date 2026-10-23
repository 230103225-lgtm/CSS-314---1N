import os
import time
import multiprocessing as mp

import numpy as np

STUDENT_ID = "230103225"
N = 30_000_000
REPEATS = 3
BLOCK = 262_144

_A = None
_B = None


def init_worker(raw_a, raw_b):
    global _A, _B
    _A = np.frombuffer(raw_a, dtype=np.float64)
    _B = np.frombuffer(raw_b, dtype=np.float64)


def noop(_):
    return 0


def kernel_copy(bounds):
    s, e = bounds
    np.copyto(_B[s:e], _A[s:e])
    return 0


def kernel_math(bounds):
    s, e = bounds
    for i in range(s, e, BLOCK):
        j = min(i + BLOCK, e)
        a = _A[i:j]
        _B[i:j] = np.sin(a) * np.cos(a) + np.tan(a)
    return 0


def make_chunks(n, p):
    size = n // p
    return [(i * size, n if i == p - 1 else (i + 1) * size) for i in range(p)]


def bench(kernel, p, raw_a, raw_b):
    chunks = make_chunks(N, p)
    with mp.Pool(p, initializer=init_worker, initargs=(raw_a, raw_b)) as pool:
        pool.map(noop, range(p))
        pool.map(kernel, chunks, chunksize=1)
        best = float("inf")
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            pool.map(kernel, chunks, chunksize=1)
            best = min(best, (time.perf_counter() - t0) * 1000)
    return best


def run_kernel(name, kernel, core_list, raw_a, raw_b):
    print(f"--- {name} ---")
    print(f"{'Cores':>6} | {'Wall Time (ms)':>14} | {'Speedup':>8} | {'Efficiency':>10}")
    print("-" * 49)
    t1 = None
    result = {}
    for p in core_list:
        t = bench(kernel, p, raw_a, raw_b)
        if t1 is None:
            t1 = t
        speedup = t1 / t
        eff = speedup / p
        result[p] = (t, speedup, eff)
        print(f"{p:>6} | {t:>14.2f} | {speedup:>8.2f} | {eff:>10.2f}")
    print()
    return result


def main():
    max_cores = os.cpu_count()
    core_list = sorted(set([1, 2, max_cores]))

    print(f"Student ID: {STUDENT_ID}")
    print(f"Array size: {N:,} float64 ({N * 8 / 1e6:.0f} MB per array)")
    print(f"Logical cores: {max_cores}")
    print(f"Timing: best of {REPEATS} runs, pool startup excluded\n")

    raw_a = mp.RawArray("d", N)
    raw_b = mp.RawArray("d", N)
    a = np.frombuffer(raw_a, dtype=np.float64)
    np.random.default_rng(int(STUDENT_ID)).random(N, out=a)

    res_copy = run_kernel("Kernel A (memory-bound): B[i] = A[i]", kernel_copy, core_list, raw_a, raw_b)
    res_math = run_kernel("Kernel B (compute-bound): sin*cos+tan", kernel_math, core_list, raw_a, raw_b)

    b = np.frombuffer(raw_b, dtype=np.float64)
    idx = np.random.default_rng(1).integers(0, N, 1000)
    exp = np.sin(a[idx]) * np.cos(a[idx]) + np.tan(a[idx])
    print(f"Correctness check (Kernel B sample): {'OK' if np.allclose(b[idx], exp) else 'FAILED'}\n")

    eff_copy = res_copy[max_cores][2]
    eff_math = res_math[max_cores][2]
    sp_copy = res_copy[max_cores][1]
    sp_math = res_math[max_cores][1]

    print("=== CONCLUSION ===")
    print(f"At {max_cores} cores: Kernel A (copy) efficiency = {eff_copy:.2f} (speedup {sp_copy:.2f}x)")
    print(f"At {max_cores} cores: Kernel B (math) efficiency = {eff_math:.2f} (speedup {sp_math:.2f}x)")

    if eff_copy < eff_math - 0.10:
        print("RESULT: Kernel A (vector copy, memory-bound) STALLED in scaling due to "
              "memory channel / RAM bandwidth saturation.")
        print("Kernel B (sin*cos+tan, compute-bound) kept scaling because each element "
              "needs many CPU operations per byte moved.")
    elif eff_math < eff_copy - 0.10:
        print("RESULT: Kernel B scaled worse than Kernel A on this machine. "
              "Memory saturation is NOT the dominant limit here; likely causes: "
              "efficiency cores / thermal limits / OS scheduling.")
    else:
        print("RESULT: Both kernels show similar efficiency. No clear memory channel "
              "saturation was detected on this run.")


if __name__ == "__main__":
    main()