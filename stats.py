"""Reliability statistics for the revision (AAI 2026 reviewers): bootstrap 95% CIs of success over the 52 tasks,
exact two-sided McNemar tests on paired task outcomes, and a two-way split of the variance of the 15 success rates
(model vs design). Writes results/stats.json."""
import json
import os
from math import comb

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
MODELS = ["qwen3:1.7b", "llama3.2:3b", "qwen3:4b-instruct-2507-q4_K_M", "granite3.3:8b", "qwen3:8b"]
ARCHS = ["single", "pe", "pev"]

rows = [json.loads(l) for l in open(os.path.join(HERE, "results", "runs.jsonl"))]
rows = [r for r in rows if r.get("think") == "off" and r["model"] in MODELS]
succ = {(r["model"], r["arch"], r["task"]): int(r["success"]) for r in rows}
tasks = sorted({r["task"] for r in rows})
vec = lambda m, a: np.array([succ[(m, a, t)] for t in tasks])  # noqa: E731
idx = np.random.default_rng(0).integers(0, len(tasks), (10000, len(tasks)))


def mcnemar(x, y):
    b, c = int(((x == 1) & (y == 0)).sum()), int(((x == 0) & (y == 1)).sum())
    n = b + c
    p = 1.0 if n == 0 else min(1.0, 2 * sum(comb(n, k) for k in range(min(b, c) + 1)) / 2 ** n)
    return {"a_only": b, "b_only": c, "p": p}


out = {"ci": {}, "design_tests": {}, "model_tests": {}}
for m in MODELS:
    for a in ARCHS:
        v = vec(m, a)
        lo, hi = np.percentile(v[idx].mean(1) * 100, [2.5, 97.5])
        out["ci"][f"{m}|{a}"] = {"succ": v.mean() * 100, "lo": lo, "hi": hi}
    for a in ["pe", "pev"]:
        out["design_tests"][f"{m}|single-vs-{a}"] = mcnemar(vec(m, "single"), vec(m, a))
    out["model_tests"][f"qwen3:8b-vs-{m}|single"] = mcnemar(vec("qwen3:8b", "single"), vec(m, "single"))
out["q8_single_vs_q4_pev"] = mcnemar(vec("qwen3:8b", "single"), vec(MODELS[2], "pev"))
X = np.array([[vec(m, a).mean() * 100 for a in ARCHS] for m in MODELS])
g = X.mean()
ss_m, ss_a, ss_t = 3 * ((X.mean(1) - g) ** 2).sum(), 5 * ((X.mean(0) - g) ** 2).sum(), ((X - g) ** 2).sum()
out["variance_share"] = {"model": ss_m / ss_t, "design": ss_a / ss_t, "interaction": (ss_t - ss_m - ss_a) / ss_t}
out["range_over_models_per_design"] = {a: float(X[:, j].max() - X[:, j].min()) for j, a in enumerate(ARCHS)}
out["range_over_designs_per_model"] = {m: float(X[i].max() - X[i].min()) for i, m in enumerate(MODELS)}
out["mean_ci_halfwidth"] = float(np.mean([(v["hi"] - v["lo"]) / 2 for v in out["ci"].values()]))
json.dump(out, open(os.path.join(HERE, "results", "stats.json"), "w"), indent=1)
print(json.dumps({k: out[k] for k in ["variance_share", "range_over_models_per_design", "range_over_designs_per_model",
                                       "mean_ci_halfwidth", "q8_single_vs_q4_pev"]}, indent=1))
