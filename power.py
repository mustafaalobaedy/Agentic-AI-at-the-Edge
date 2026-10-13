"""Background power sampler for Jetson AGX Orin using tegrastats (module rails, mW)."""
import re
import subprocess
import threading
import time

RAILS = ("VDD_GPU_SOC", "VDD_CPU_CV", "VIN_SYS_5V0")
RAIL_RE = re.compile(r"(VDD_GPU_SOC|VDD_CPU_CV|VIN_SYS_5V0) (\d+)mW")
RAM_RE = re.compile(r"RAM (\d+)/\d+MB")


class PowerMeter:
    def __init__(self, interval_ms=100):
        self.samples = []  # (t, total_mW, ram_MB)
        self._p = subprocess.Popen(["tegrastats", "--interval", str(interval_ms)],
                                   stdout=subprocess.PIPE, text=True, bufsize=1)
        self._lock = threading.Lock()
        threading.Thread(target=self._read, daemon=True).start()

    def _read(self):
        for line in self._p.stdout:
            mw = sum(int(v) for _, v in RAIL_RE.findall(line))
            ram = RAM_RE.search(line)
            with self._lock:
                self.samples.append((time.time(), mw, int(ram.group(1)) if ram else 0))

    def window(self, t0, t1):
        with self._lock:
            s = [x for x in self.samples if t0 <= x[0] <= t1]
        if len(s) < 2:
            return {"energy_j": None, "avg_w": None, "peak_ram_mb": None}
        e = sum((b[0] - a[0]) * (a[1] + b[1]) / 2 for a, b in zip(s, s[1:])) / 1000.0
        return {"energy_j": e, "avg_w": e / (s[-1][0] - s[0][0]), "peak_ram_mb": max(x[2] for x in s)}

    def idle(self, seconds=10):
        t0 = time.time()
        time.sleep(seconds)
        return self.window(t0, time.time())

    def stop(self):
        self._p.terminate()
