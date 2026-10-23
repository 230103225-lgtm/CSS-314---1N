import os
import time
import tempfile
import multiprocessing as mp

import numpy as np

STUDENT_ID = 230103225
N = 20_000_000
P_PARALLEL = 0.95
REPEATS = 3
BLOCK = 1_000_000

DATA_FILE = os.path.join(tempfile.gettempdir(), f"task2_data_{STUDENT_ID}.npy")


def noop(_):
    return 0


def partial_sum(bounds):
    start, end = bounds
    arr = np.load(DATA_FILE, mmap_mode="r")
    total = 0.0
    for i in range(start, end, BLOCK):
        block = arr[i:min(i + BLOCK, end)].tolist()
        for x in block:
            total += x
    return total


def make_chunks(n, p):
    size = n // p
    chunks = []
    for i in range(p):
        s = i * size
        e = n if i == p - 1 else s + size
        chunks.append((s, e))
    return chunks


def benchmark(p):
    chunks = make_chunks(N, p)
    with mp.Pool(processes=p) as pool:
        pool.map(noop, range(p))
        best = None
        result = None
        for _ in range(REPEATS):
            t0 = time.perf_counter()
            result = sum(pool.map(partial_sum, chunks, chunksize=1))
            t = (time.perf_counter() - t0) * 1000
            best = t if best is None else min(best, t)
    return best, result


def amdahl(p, P=P_PARALLEL):
    return 1.0 / ((1.0 - P) + P / p)


def main():
    max_cores = os.cpu_count()
    worker_counts = sorted(set([1, 2, 4, max_cores, max_cores * 2]))

    print(f"Student ID: {STUDENT_ID}")
    print(f"Logical cores: {max_cores}")
    print(f"Array size: {N:,}")
    print(f"Parallel fraction P = {P_PARALLEL}")
    print(f"Timing: best of {REPEATS} runs, pool startup excluded\n")

    rng = np.random.default_rng(STUDENT_ID)
    np.save(DATA_FILE, rng.random(N))

    print(f"{'Cores':>6} | {'Wall Time (ms)':>15} | {'Speedup':>8} | {'Amdahl Upper Bound':>18}")
    print("-" * 58)

    t1 = None
    for p in worker_counts:
        t, total = benchmark(p)
        if p == 1:
            t1 = t
        speedup = t1 / t
        print(f"{p:>6} | {t:>15.2f} | {speedup:>8.2f} | {amdahl(p):>18.2f}")

    print(f"\nChecksum (sum of last run): {total:.6f}")
    print(f"Amdahl limit for p -> infinity: {1 / (1 - P_PARALLEL):.2f}")

    os.remove(DATA_FILE)


if __name__ == "__main__":
    main()