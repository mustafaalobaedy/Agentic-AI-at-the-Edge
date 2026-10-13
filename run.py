"""Run the benchmark grid. Results are appended to results/runs.jsonl (resumable).

Example:
  python3 run.py --models qwen3:4b llama3.2:3b --archs single pe pev --reps 1
"""
import argparse
import json
import os
import time
import traceback

from agents import ARCHS, chat
from env import Building
from power import PowerMeter
from tasks import TASKS, evaluate

HERE = os.path.dirname(os.path.abspath(__file__))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="+", required=True)
    ap.add_argument("--archs", nargs="+", default=list(ARCHS))
    ap.add_argument("--reps", type=int, default=1)
    ap.add_argument("--think", choices=["off", "on", "default"], default="off",
                    help="thinking mode for models that support it (qwen3)")
    ap.add_argument("--tasks", nargs="*", help="subset of task ids")
    ap.add_argument("--out", default=os.path.join(HERE, "results", "runs.jsonl"))
    a = ap.parse_args()

    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    done = set()
    if os.path.exists(a.out):
        with open(a.out) as f:
            for line in f:
                r = json.loads(line)
                done.add((r["model"], r["arch"], r["think"], r["task"], r["rep"]))

    tasks = [t for t in TASKS if not a.tasks or t["id"] in a.tasks]
    meter = PowerMeter()
    time.sleep(1)

    for model in a.models:
        think = None
        if model.startswith("qwen3"):
            think = {"off": False, "on": True, "default": None}[a.think]
        chat(model, [{"role": "user", "content": "hi"}], think=think)  # warm-up / load weights
        idle = meter.idle(15)  # baseline with model resident
        print(f"{model}: idle power {idle['avg_w']:.2f} W", flush=True)
        for arch in a.archs:
            for rep in range(a.reps):
                for t in tasks:
                    key = (model, arch, a.think, t["id"], rep)
                    if key in done:
                        continue
                    b = Building(t["overrides"])
                    t0 = time.time()
                    err = None
                    try:
                        ans, st, extra = ARCHS[arch](model, b, t["prompt"], think, rep)
                    except Exception as e:
                        ans, st, extra, err = "", None, {}, f"{type(e).__name__}: {e}"
                        traceback.print_exc()
                    t1 = time.time()
                    ok, check_ok, viol = evaluate(t, b, ans)
                    pw = meter.window(t0, t1)
                    rec = {
                        "model": model, "arch": arch, "think": a.think, "task": t["id"], "level": t["level"], "rep": rep,
                        "success": ok, "check_ok": check_ok, "side_effect": viol, "error": err,
                        "latency_s": t1 - t0, **pw, "idle_w": idle["avg_w"],
                        **({k: getattr(st, k) for k in vars(st)} if st else {}),
                        "answer": ans, "calls": [[n, args, okc] for n, args, okc in b.calls],
                        "notifications": b.notifications, "changed": sorted(map(list, b.changed())), **extra,
                    }
                    with open(a.out, "a") as f:
                        f.write(json.dumps(rec) + "\n")
                    print(f"{model:16s} {arch:6s} r{rep} {t['id']:18s} {'PASS' if ok else 'FAIL'}"
                          f"{' SIDE' if viol else ''} {t1 - t0:6.1f}s {pw['energy_j'] or 0:7.1f}J", flush=True)
    meter.stop()


if __name__ == "__main__":
    main()
