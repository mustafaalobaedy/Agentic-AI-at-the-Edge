"""Aggregate results/runs.jsonl into tables (results/tables/), figures (results/figs/) and JSON summaries."""
import json
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
OUT = os.path.join(HERE, "results")
FIGDIR = os.path.join(OUT, "figs")
RUNS = os.path.join(HERE, "results", "runs.jsonl")

MODELS = ["qwen3:1.7b", "llama3.2:3b", "qwen3:4b-instruct-2507-q4_K_M", "granite3.3:8b", "qwen3:8b"]
MNAME = {"qwen3:1.7b": "Qwen3-1.7B", "llama3.2:3b": "Llama-3.2-3B", "qwen3:4b-instruct-2507-q4_K_M": "Qwen3-4B",
         "qwen3:4b-thinking-2507-q4_K_M": "Qwen3-4B-Thinking",
         "granite3.3:8b": "Granite-3.3-8B", "qwen3:8b": "Qwen3-8B"}
# thinking-ablation baseline: the 4B thinking model is compared with its instruct sibling
BASELINE = {"qwen3:4b-thinking-2507-q4_K_M": "qwen3:4b-instruct-2507-q4_K_M"}
ARCHS = ["single", "pe", "pev"]
ANAME = {"single": "S", "pe": "PE", "pev": "PEV"}
LEVELS = ["L1-single", "L2-multi", "L3-policy", "L4-safety"]
# validated categorical slots 1-3 (dataviz reference palette, light mode)
COLOR = {"single": "#2a78d6", "pe": "#eb6834", "pev": "#1baf7a"}
HATCH = {"single": "", "pe": "////", "pev": "\\\\\\\\"}
INK, MUTED, GRID = "#1f1f1e", "#5f5e5a", "#d9d8d2"

plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "Liberation Serif", "DejaVu Serif"],
                     "font.size": 8, "axes.edgecolor": MUTED, "axes.labelcolor": INK, "xtick.color": MUTED,
                     "ytick.color": MUTED, "axes.linewidth": 0.5, "hatch.linewidth": 0.4, "hatch.color": "white"})


def load(think="off"):
    rows = [json.loads(l) for l in open(RUNS)]
    return [r for r in rows if r["think"] == think]


def agg(rows):
    """Group by (model, arch) -> dict of metrics."""
    g = defaultdict(list)
    for r in rows:
        g[(r["model"], r["arch"])].append(r)
    out = {}
    for k, rs in g.items():
        n = len(rs)
        net = [r["energy_j"] - r["idle_w"] * r["latency_s"] for r in rs if r.get("energy_j") is not None]
        tc = sum(r.get("tool_calls", 0) for r in rs)
        out[k] = {
            "n": n,
            "succ": 100 * np.mean([r["success"] for r in rs]),
            "lvl": {L: 100 * np.mean([r["success"] for r in rs if r["level"] == L]) if any(r["level"] == L for r in rs) else np.nan
                    for L in LEVELS},
            "side": 100 * np.mean([r["side_effect"] for r in rs]),
            "lat": np.mean([r["latency_s"] for r in rs]),
            "lat_p90": np.percentile([r["latency_s"] for r in rs], 90),
            "energy": np.mean([r["energy_j"] for r in rs if r.get("energy_j") is not None]),
            "net_energy": np.mean(net) if net else np.nan,
            "avg_w": np.mean([r["avg_w"] for r in rs if r.get("avg_w")]),
            "idle_w": np.mean([r["idle_w"] for r in rs]),
            "llm_calls": np.mean([r.get("llm_calls", 0) for r in rs]),
            "gen_tok": np.mean([r.get("gen_tokens", 0) for r in rs]),
            "prompt_tok": np.mean([r.get("prompt_tokens", 0) for r in rs]),
            "tok_s": sum(r.get("gen_tokens", 0) for r in rs) / max(1e-9, sum(r.get("eval_s", 0) for r in rs)),
            "tool_calls": tc / n,
            "bad_pct": 100 * sum(r.get("bad_tool_calls", 0) for r in rs) / max(1, tc),
            "peak_ram": max((r.get("peak_ram_mb") or 0) for r in rs),
        }
    return out


def fmt(x, d=1):
    return "--" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{d}f}"


def table_success(A):
    best = {}
    for c in LEVELS + ["all"]:
        vals = [(A[k]["succ"] if c == "all" else A[k]["lvl"][c]) for k in A]
        best[c] = np.nanmax(vals) if vals else None
    lines = [r"\begin{table}[t]", r"\centering", r"\caption{Task success rate (\%) by level and architecture. Best per column in bold.}",
             r"\label{tab:success}", r"\setlength{\tabcolsep}{4pt}", r"\begin{tabular}{@{}llccccc@{}}", r"\toprule",
             r"Model & Arch. & L1 & L2 & L3 & L4 & All \\", r"\midrule"]
    for m in MODELS:
        ks = [(m, a) for a in ARCHS if (m, a) in A]
        for i, k in enumerate(ks):
            cells = []
            for c in LEVELS + ["all"]:
                v = A[k]["succ"] if c == "all" else A[k]["lvl"][c]
                s = fmt(v, 0)
                cells.append(rf"\textbf{{{s}}}" if best[c] is not None and abs(v - best[c]) < 1e-9 else s)
            name = rf"\multirow{{{len(ks)}}}{{*}}{{{MNAME[m]}}}" if i == 0 else ""
            lines.append(f"{name} & {ANAME[k[1]]} & " + " & ".join(cells) + r" \\")
        if ks and m != MODELS[-1]:
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines)


def table_cost(A):
    lines = [r"\begin{table*}[t]", r"\centering",
             r"\caption{Efficiency per task on the Jetson AGX Orin (means over 52 tasks). Net energy subtracts idle power with the model resident. Invalid: share of tool calls rejected by the environment.}",
             r"\label{tab:cost}", r"\begin{tabular}{@{}llrrrrrrrrr@{}}", r"\toprule",
             r"Model & Arch. & Succ.\,(\%) & Latency (s) & p90 (s) & LLM calls & Tool calls & Invalid (\%) & Gen.\ tok & Tok/s & Energy (J) / net \\",
             r"\midrule"]
    for m in MODELS:
        for a in ARCHS:
            k = (m, a)
            if k not in A:
                continue
            x = A[k]
            lines.append(f"{MNAME[m] if a == 'single' else ''} & {ANAME[a]} & {fmt(x['succ'])} & {fmt(x['lat'])} & {fmt(x['lat_p90'])} & "
                         f"{fmt(x['llm_calls'])} & {fmt(x['tool_calls'])} & {fmt(x['bad_pct'])} & {fmt(x['gen_tok'], 0)} & "
                         f"{fmt(x['tok_s'])} & {fmt(x['energy'], 0)} / {fmt(x['net_energy'], 0)} \\\\")
        if m != MODELS[-1] and any((m, a) in A for a in ARCHS):
            lines.append(r"\addlinespace[2pt]")
    lines += [r"\bottomrule", r"\end{tabular}", r"\end{table*}"]
    return "\n".join(lines)


def save(fig, name):
    os.makedirs(FIGDIR, exist_ok=True)
    fig.savefig(os.path.join(FIGDIR, name + ".png"), dpi=300)
    fig.savefig(os.path.join(FIGDIR, name + ".pdf"))
    plt.close(fig)


def fig_eda(rows):
    """Run-level distributions by task level: latency and energy (EDA)."""
    fig, axes = plt.subplots(1, 2, figsize=(3.4, 1.7))
    lv = [L.split("-")[0] for L in LEVELS]
    for ax, key, lab in [(axes[0], "latency_s", "Latency per task (s)"), (axes[1], "energy_j", "Energy per task (J)")]:
        data = [[r[key] for r in rows if r["level"] == L and r.get(key) is not None] for L in LEVELS]
        bp = ax.boxplot(data, whis=(5, 95), showfliers=False, widths=0.55, patch_artist=True,
                        medianprops=dict(color=INK, lw=0.9), boxprops=dict(lw=0.5, color=MUTED),
                        whiskerprops=dict(lw=0.5, color=MUTED), capprops=dict(lw=0.5, color=MUTED))
        for p in bp["boxes"]:
            p.set_facecolor("#cde2fb")
        ax.set_xticks(range(1, 5), lv)
        ax.set_yscale("log")
        ax.set_ylabel(lab)
        style(ax)
    fig.tight_layout(pad=0.3, w_pad=1.0)
    save(fig, "eda")


def eda_stats(rows):
    """Reference tool calls: per task, the fewest tool calls among successful runs; averaged by level."""
    best = {}
    for r in rows:
        if r["success"]:
            k = (r["level"], r["task"])
            best[k] = min(best.get(k, 1e9), r.get("tool_calls", 0))
    ref = {L: float(np.mean([v for (l, _), v in best.items() if l == L])) for L in LEVELS if any(l == L for l, _ in best)}
    ref["all"] = float(np.mean(list(best.values()))) if best else None
    lat = {L: [r["latency_s"] for r in rows if r["level"] == L] for L in LEVELS}
    en = {L: [r["energy_j"] for r in rows if r["level"] == L and r.get("energy_j")] for L in LEVELS}
    out = {"ref_calls": ref, "solved_tasks": len(best), "n_runs": len(rows),
           "lat_median": {L: float(np.median(v)) for L, v in lat.items() if v},
           "lat_p95": {L: float(np.percentile(v, 95)) for L, v in lat.items() if v},
           "energy_median": {L: float(np.median(v)) for L, v in en.items() if v},
           "energy_p95": {L: float(np.percentile(v, 95)) for L, v in en.items() if v},
           "avg_power_w": float(np.mean([r["avg_w"] for r in rows if r.get("avg_w")])),
           "idle_w": {m: float(np.mean([r["idle_w"] for r in rows if r["model"] == m])) for m in MODELS if any(r["model"] == m for r in rows)},
           "unsolved": sorted({r["task"] for r in rows} - {t for _, t in best})}
    json.dump(out, open(os.path.join(HERE, "results", "eda.json"), "w"), indent=1)
    return out


def style(ax):
    ax.spines[["top", "right"]].set_visible(False)
    ax.grid(axis="y", color=GRID, lw=0.4)
    ax.set_axisbelow(True)


def fig_success(A):
    ms = [m for m in MODELS if any((m, a) in A for a in ARCHS)]
    fig, ax = plt.subplots(figsize=(3.4, 1.8))
    w = 0.26
    x = np.arange(len(ms))
    for i, a in enumerate(ARCHS):
        v = [A[(m, a)]["succ"] if (m, a) in A else 0 for m in ms]
        ax.bar(x + (i - 1) * w, v, w - 0.03, color=COLOR[a], hatch=HATCH[a], label=ANAME[a], lw=0)
    style(ax)
    ax.set_xticks(x, [MNAME[m] for m in ms])
    ax.set_ylabel("Task success (%)")
    ax.set_ylim(0, 100)
    ax.legend(ncol=3, frameon=False, loc="upper left", fontsize=7, handlelength=1.4, columnspacing=1.0)
    fig.tight_layout(pad=0.2)
    save(fig, "success")


def fig_tradeoff(A):
    fig, ax = plt.subplots(figsize=(3.4, 2.1))
    marker = {"qwen3:1.7b": "o", "llama3.2:3b": "s", "qwen3:4b-instruct-2507-q4_K_M": "D", "granite3.3:8b": "^", "qwen3:8b": "v"}
    for (m, a), x in A.items():
        ax.scatter(x["energy"], x["succ"], s=26, marker=marker[m], color=COLOR[a], edgecolor="white", lw=0.8, zorder=3)
    for m in MODELS:
        pts = [(A[(m, a)]["energy"], A[(m, a)]["succ"]) for a in ARCHS if (m, a) in A]
        if len(pts) > 1:
            ax.plot(*zip(*pts), color=GRID, lw=0.8, zorder=1)
        if pts:
            ax.annotate(MNAME[m], pts[0], xytext=(-4, 0), textcoords="offset points", ha="right", va="center",
                        fontsize=7, color=MUTED)
    style(ax)
    ax.grid(axis="x", color=GRID, lw=0.4)
    ax.set_xscale("log")
    ax.set_xlabel("Energy per task (J, log scale)")
    ax.set_ylabel("Task success (%)")
    ax.set_ylim(0, 100)
    handles = [plt.Line2D([], [], ls="", marker="o", color=COLOR[a], label=ANAME[a], ms=4.5) for a in ARCHS]
    ax.legend(handles=handles, frameon=False, loc="lower right", fontsize=7, ncol=3, handletextpad=0.2, columnspacing=0.8)
    fig.tight_layout(pad=0.2)
    save(fig, "tradeoff")


def failure_breakdown(rows):
    """Coarse failure taxonomy for failed runs (think=off)."""
    cats = defaultdict(lambda: defaultdict(int))
    for r in rows:
        if r["success"]:
            continue
        key = (r["model"], r["arch"])
        acts = [c for c in r["calls"] if c[0] in ("set_device", "notify")]
        if r.get("error"):
            c = "runtime error"
        elif r["side_effect"]:
            c = "unrequested side effect"
        elif "[step limit reached]" in (r["answer"] or ""):
            c = "step limit"
        elif r["level"] in ("L3-policy", "L4-safety") and acts:
            c = "policy/safety violation"
        elif not acts and r["level"] != "L1-single" and r["task"].startswith(("a_", "m_", "p_", "s_")) and any(
                w in (r["answer"] or "").lower() for w in ["i will", "i'll", "let me", "next"]):
            c = "premature stop (announced, not executed)"
        else:
            c = "wrong value / incomplete"
        cats[key][c] += 1
    return cats


def main():
    rows = load("off")
    os.makedirs(os.path.join(OUT, "tables"), exist_ok=True)
    os.makedirs(FIGDIR, exist_ok=True)
    A = agg(rows)
    open(os.path.join(OUT, "tables", "success.tex"), "w").write(table_success(A))
    open(os.path.join(OUT, "tables", "cost.tex"), "w").write(table_cost(A))
    fig_success(A)
    fig_tradeoff(A)
    fig_eda(rows)
    E = eda_stats(rows)
    print("EDA:", json.dumps({k: v for k, v in E.items() if k != "unsolved"}), "unsolved:", E["unsolved"])

    print(f"{'model':15s} {'arch':5s} {'n':>3s} {'succ':>5s} " + " ".join(f"{L[:2]:>5s}" for L in LEVELS) +
          f" {'side':>5s} {'lat':>6s} {'E(J)':>6s} {'netE':>6s} {'W':>5s} {'idleW':>5s} {'calls':>5s} {'tools':>5s} {'bad%':>5s} {'tok/s':>6s} {'ram':>6s}")
    for m in MODELS:
        for a in ARCHS:
            if (m, a) not in A:
                continue
            x = A[(m, a)]
            print(f"{m:15s} {a:5s} {x['n']:3d} {x['succ']:5.1f} " + " ".join(f"{x['lvl'][L]:5.1f}" for L in LEVELS) +
                  f" {x['side']:5.1f} {x['lat']:6.1f} {x['energy']:6.0f} {x['net_energy']:6.0f} {x['avg_w']:5.1f} {x['idle_w']:5.1f} "
                  f"{x['llm_calls']:5.1f} {x['tool_calls']:5.1f} {x['bad_pct']:5.1f} {x['tok_s']:6.1f} {x['peak_ram']:6d}")

    print("\nFailure taxonomy:")
    for k, c in sorted(failure_breakdown(rows).items()):
        print(" ", k, dict(c))

    think_rows = load("on")
    if think_rows:
        T = agg(think_rows)
        print("\nThinking ON ablation:")
        for k, x in sorted(T.items()):
            base = A.get((BASELINE.get(k[0], k[0]), k[1]))
            print(f"  {k}: succ {x['succ']:.1f} (off {base['succ'] if base else float('nan'):.1f}) "
                  f"lat {x['lat']:.1f}s (off {base['lat'] if base else float('nan'):.1f}) "
                  f"E {x['energy']:.0f}J gen_tok {x['gen_tok']:.0f} lvls " + str({L[:2]: fmt(x['lvl'][L], 0) for L in LEVELS}))
        json.dump({f"{k[0]}|{k[1]}": {kk: vv for kk, vv in v.items() if kk != 'lvl'} | {"lvl": v["lvl"]} for k, v in T.items()},
                  open(os.path.join(HERE, "results", "think_summary.json"), "w"), indent=1, default=float)
    json.dump({f"{k[0]}|{k[1]}": {kk: vv for kk, vv in v.items() if kk != 'lvl'} | {"lvl": v["lvl"]} for k, v in A.items()},
              open(os.path.join(HERE, "results", "summary.json"), "w"), indent=1, default=float)


if __name__ == "__main__":
    main()
