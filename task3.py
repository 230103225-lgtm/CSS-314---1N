#task3
import os
import time
import subprocess
import multiprocessing as mp
from multiprocessing import RawArray

STUDENT_ID = "230103225"
ITERS = 50_000_000
PAD_ELEMS = 16


def physical_cores():
    try:
        out = subprocess.check_output(["sysctl", "-n", "hw.physicalcpu"]).decode().strip()
        return int(out)
    except Exception:
        return os.cpu_count() or 1


def worker(arr, idx, stride, mode, barrier):
    pos = idx * stride
    barrier.wait()
    if mode == "local":
        local = 0
        for _ in range(ITERS):
            local += 1
        arr[pos] = local
    else:
        for _ in range(ITERS):
            arr[pos] += 1


def run(mode, n, stride):
    arr = RawArray("q", n * stride)
    barrier = mp.Barrier(n + 1)
    procs = [mp.Process(target=worker, args=(arr, i, stride, mode, barrier)) for i in range(n)]
    for p in procs:
        p.start()
    barrier.wait()
    t0 = time.perf_counter()
    for p in procs:
        p.join()
    elapsed = (time.perf_counter() - t0) * 1000
    ok = all(arr[i * stride] == ITERS for i in range(n))
    return elapsed, ok


def main():
    n = physical_cores()
    print(f"Student ID: {STUDENT_ID}")
    print(f"Physical cores (workers used): {n}")
    print(f"Increments per worker: {ITERS:,}")
    print(f"Padding: {PAD_ELEMS * 8} bytes\n")

    t_unpadded, ok1 = run("shared", n, 1)
    t_padded, ok2 = run("shared", n, PAD_ELEMS)
    t_local, ok3 = run("local", n, 1)

    print(f"{'Variant':<22} | {'Wall Time (ms)':>14}")
    print("-" * 23 + "+" + "-" * 16)
    print(f"{'Unpadded Shared':<22} | {t_unpadded:>14.2f}")
    print(f"{'Padded (128B)':<22} | {t_padded:>14.2f}")
    print(f"{'Local variable':<22} | {t_local:>14.2f}")

    print(f"\nRatio Unpadded / Padded: {t_unpadded / t_padded:.2f}x")
    print(f"Ratio Unpadded / Local:  {t_unpadded / t_local:.2f}x")
    print(f"Correctness check: {'OK' if ok1 and ok2 and ok3 else 'FAILED'}")


if __name__ == "__main__":
    main()
