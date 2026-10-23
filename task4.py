import time
import queue
import threading

STUDENT_ID = "230103225"
N_ITEMS = 1000
WORKERS = 4


def cost(i):
    time.sleep(0.0001 * (i % 20))


def run_strategy(name, make_worker_fn):
    finish = [0.0] * WORKERS
    barrier = threading.Barrier(WORKERS + 1)
    start_holder = {}

    def runner(wid):
        work = make_worker_fn(wid)
        barrier.wait()
        work()
        finish[wid] = time.perf_counter()

    threads = [threading.Thread(target=runner, args=(w,)) for w in range(WORKERS)]
    for t in threads:
        t.start()
    barrier.wait()
    t0 = time.perf_counter()
    for t in threads:
        t.join()

    rel = [(f - t0) * 1000 for f in finish]
    total = max(rel)
    first = min(rel)
    delta = total - first
    return name, total, first, delta, rel


def static_strategy():
    size = N_ITEMS // WORKERS
    ranges = [(w * size, N_ITEMS if w == WORKERS - 1 else (w + 1) * size) for w in range(WORKERS)]

    def make(wid):
        s, e = ranges[wid]

        def work():
            for i in range(s, e):
                cost(i)
        return work
    return make


def dynamic_strategy(chunk):
    q = queue.Queue()
    for s in range(0, N_ITEMS, chunk):
        q.put((s, min(s + chunk, N_ITEMS)))

    def make(wid):
        def work():
            while True:
                try:
                    s, e = q.get_nowait()
                except queue.Empty:
                    return
                for i in range(s, e):
                    cost(i)
        return work
    return make


def main():
    ideal_serial = sum(0.0001 * (i % 20) for i in range(N_ITEMS)) * 1000
    print(f"Student ID: {STUDENT_ID}")
    print(f"Items: {N_ITEMS}, workers: {WORKERS}")
    print(f"Cost of item i: sleep(0.0001 * (i % 20))")
    print(f"Ideal serial work: {ideal_serial:.2f} ms, ideal per worker: {ideal_serial / WORKERS:.2f} ms\n")

    results = [
        run_strategy("A: Static chunks (250)", static_strategy()),
        run_strategy("B1: Dynamic (chunk=1)", dynamic_strategy(1)),
        run_strategy("B2: Dynamic (chunk=5)", dynamic_strategy(5)),
    ]

    print(f"{'Strategy':<24} | {'Total (ms)':>10} | {'First done':>10} | {'Last done':>10} | {'Delta (ms)':>10}")
    print("-" * 75)
    for name, total, first, delta, rel in results:
        print(f"{name:<24} | {total:>10.2f} | {first:>10.2f} | {total:>10.2f} | {delta:>10.2f}")

    print("\nPer-worker finish times (ms):")
    for name, total, first, delta, rel in results:
        print(f"  {name:<24} " + "  ".join(f"W{i}={t:.2f}" for i, t in enumerate(rel)))

    base = results[0][1]
    print()
    for name, total, first, delta, rel in results[1:]:
        print(f"Speedup {name} vs Static: {base / total:.3f}x")


if __name__ == "__main__":
    main()