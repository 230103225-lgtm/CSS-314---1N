import sys
import time
import threading

STUDENT_ID = "230103225"
THREADS = 4
INCREMENTS = 2_000_000
EXPECTED = THREADS * INCREMENTS


class Counter:
    value = 0


def add_one(v):
    return v + 1


def unsafe_worker(c, barrier):
    barrier.wait()
    for _ in range(INCREMENTS):
        v = c.value
        c.value = add_one(v)


def safe_worker(c, lock, barrier):
    barrier.wait()
    for _ in range(INCREMENTS):
        with lock:
            v = c.value
            c.value = add_one(v)


def run(target, make_args):
    c = Counter()
    barrier = threading.Barrier(THREADS + 1)
    threads = [threading.Thread(target=target, args=make_args(c, barrier)) for _ in range(THREADS)]
    for t in threads:
        t.start()
    barrier.wait()
    t0 = time.perf_counter()
    for t in threads:
        t.join()
    elapsed = (time.perf_counter() - t0) * 1000
    return c.value, elapsed


def main():
    sys.setswitchinterval(1e-6)

    print(f"Student ID: {STUDENT_ID}")
    print(f"Python: {sys.version.split()[0]}")
    print(f"Threads: {THREADS}, increments per thread: {INCREMENTS:,}")
    print(f"Thread switch interval: {sys.getswitchinterval()} s\n")

    actual, t_unsafe = run(unsafe_worker, lambda c, b: (c, b))
    lost = EXPECTED - actual
    pct = lost / EXPECTED * 100

    print("=== WITHOUT LOCK ===")
    print(f"Expected final value: {EXPECTED:,}")
    print(f"Actual final value:   {actual:,}")
    print(f"Lost updates:         {lost:,}")
    print(f"Percentage lost:      {pct:.4f}%")
    print(f"Wall time:            {t_unsafe:.2f} ms\n")

    lock = threading.Lock()
    actual_s, t_safe = run(safe_worker, lambda c, b: (c, lock, b))

    print("=== WITH LOCK (threading.Lock) ===")
    print(f"Expected final value: {EXPECTED:,}")
    print(f"Actual final value:   {actual_s:,}")
    print(f"Correctness check:    {'OK' if actual_s == EXPECTED else 'FAILED'}")
    print(f"Wall time:            {t_safe:.2f} ms\n")

    print("=== SYNCHRONIZATION PENALTY ===")
    print(f"Time without lock: {t_unsafe:.2f} ms")
    print(f"Time with lock:    {t_safe:.2f} ms")
    print(f"Penalty:           {t_safe - t_unsafe:+.2f} ms ({t_safe / t_unsafe:.2f}x)")


if __name__ == "__main__":
    main()