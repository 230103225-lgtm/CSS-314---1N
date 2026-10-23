import os
import platform
import random
import time
import math

STUDENT_ID = 230103225
N = 20_000_000


def get_cpu_model():
    # Пытаемся получить название процессора на разных ОС
    try:
        if platform.system() == "Windows":
            import subprocess
            out = subprocess.check_output("wmic cpu get name", shell=True).decode()
            lines = [l.strip() for l in out.splitlines() if l.strip()]
            if len(lines) > 1:
                return lines[1]
        elif platform.system() == "Linux":
            with open("/proc/cpuinfo") as f:
                for line in f:
                    if "model name" in line:
                        return line.split(":", 1)[1].strip()
        elif platform.system() == "Darwin":
            import subprocess
            return subprocess.check_output(
                ["sysctl", "-n", "machdep.cpu.brand_string"]
            ).decode().strip()
    except Exception:
        pass
    return platform.processor() or platform.machine()


def main():
    print(f"Student ID: {STUDENT_ID}")
    print(f"OS: {platform.system()} {platform.release()} ({platform.version()})")
    print(f"CPU: {get_cpu_model()}")
    print(f"Platform: {platform.platform()}")
    print(f"Logical cores: {os.cpu_count()}")
    print(f"Array size: {N:,}")

    # Генерация массива (не входит в замер времени)
    random.seed(STUDENT_ID)
    data = [random.random() for _ in range(N)]

    # Последовательное вычисление (один поток)
    start = time.perf_counter()

    total = 0.0
    for x in data:
        total += x

    mean = total / N

    sq_sum = 0.0
    for x in data:
        d = x - mean
        sq_sum += d * d

    std = math.sqrt(sq_sum / N)

    elapsed_ms = (time.perf_counter() - start) * 1000

    print(f"Sum: {total:.6f}")
    print(f"Standard deviation: {std:.10f}")
    print(f"Serial execution time: {elapsed_ms:.2f} ms")


if __name__ == "__main__":
    main()