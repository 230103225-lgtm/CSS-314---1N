import os
import math
import time
import multiprocessing as mp

import numpy as np

STUDENT_ID = "230103225"
A = 2.5
B = 1.0
SIZES = [100, 1_000, 10_000, 100_000, 1_000_000, 10_000_000]

_X = None
_Y = None


def init_worker(raw_x, raw_y):
    global _X, _Y
    _X = np.frombuffer(raw_x, dtype=np.float64)
    _Y = np.frombuffer(raw_y, dtype=np.float64)


def noop(_):
    return 0


def work(bounds):
    s, e = bounds
    np.multiply(_X[s:e], A, out=_Y[s:e])
    np.add(_Y[s:e], B, out=_Y[s:e])
    return 0


def make_chunks(n, p):
    size = n // p
    return [(i * size, n if i == p - 1 else (i + 1) * size) for i in range(p)]


def best_time_ms(fn, repeats):
    best = float("inf")
    for _ in range(repeats):
        t0 = time.perf_counter()
        fn()
        best = min(best, (time.perf_counter() - t0) * 1000)
    return best


def benchmark(n, cores):
    raw_x = mp.RawArray("d", n)
    raw_y = mp.RawArray("d", n)
    x = np.frombuffer(raw_x, dtype=np.float64)
    y = np.frombuffer(raw_y, dtype=np.float64)
    x[:] = np.random.default_rng(int(STUDENT_ID)).random(n)

    repeats = 30 if n <= 100_000 else 10

    # Serial
    def serial():
        np.multiply(x, A, out=y)
        np.add(y, B, out=y)

    serial()  # прогрев
    t_serial = best_time_ms(serial, repeats)
    chunks = make_chunks(n, cores)
    with mp.Pool(cores, initializer=init_worker, initargs=(raw_x, raw_y)) as pool:
        pool.map(noop, range(cores))
        pool.map(work, chunks, chunksize=1)  # прогрев
        t_par = best_time_ms(lambda: pool.map(work, chunks, chunksize=1), repeats)

    expected = A * x + B
    ok = np.allclose(y, expected)
    return t_serial, t_par, ok


def find_crossover(rows):
    first_fast = None
    for i in range(len(rows)):
        if all(s > 1.0 for _, s in rows[i:]):
            first_fast = i
            break

    if first_fast is None:
        return "Параллельный вариант НЕ стал быстрее последовательного ни на одном из протестированных N."
    if first_fast == 0:
        return f"Параллельный вариант быстрее уже при N = {rows[0][0]:,} (кроссовер ниже протестированного диапазона)."

    n1, s1 = rows[first_fast - 1]
    n2, s2 = rows[first_fast]
    ln = math.log(n1) + (0 - math.log(s1)) * (math.log(n2) - math.log(n1)) / (math.log(s2) - math.log(s1))
    est = math.exp(ln)
    return (f"Кроссовер: параллельный вариант МЕДЛЕННЕЕ при N = {n1:,} (speedup {s1:.2f}x) "
            f"и БЫСТРЕЕ при N = {n2:,} (speedup {s2:.2f}x).\n"
            f"Диапазон точки перехода: N in ({n1:,}, {n2:,}], оценка по интерполяции: N ~ {est:,.0f}")


def main():
    cores = os.cpu_count()
    print(f"Student ID: {STUDENT_ID}")
    print(f"Operation: y[i] = {A} * x[i] + {B}")
    print(f"Workers (all logical cores): {cores}")
    print("Timing: best of several runs, pool startup excluded, dispatch overhead included\n")

    print(f"{'N':>12} | {'Serial (ms)':>12} | {'Parallel (ms)':>14} | {'Speedup':>8} | {'Winner':>8}")
    print("-" * 65)

    rows = []
    all_ok = True
    for n in SIZES:
        t_s, t_p, ok = benchmark(n, cores)
        all_ok = all_ok and ok
        speedup = t_s / t_p
        rows.append((n, speedup))
        winner = "parallel" if speedup > 1 else "serial"
        print(f"{n:>12,} | {t_s:>12.4f} | {t_p:>14.4f} | {speedup:>8.3f} | {winner:>8}")

    print(f"\nCorrectness check: {'OK' if all_ok else 'FAILED'}")
    print("\n=== CROSSOVER RESULT ===")
    print(find_crossover(rows))


if __name__ == "__main__":
    main()