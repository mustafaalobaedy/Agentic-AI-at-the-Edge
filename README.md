# Agentic AI at the Edge

A benchmark that runs small open-weight LLM agents **fully offline** on an NVIDIA Jetson AGX Orin and measures task success, latency, and energy on a simulated smart building.

## What is inside

| File | Purpose |
|---|---|
| `env.py` | Deterministic smart-building digital twin: 5 zones, 25 sensors, 25 actuators, 5 operating policies, 8 JSON-schema tools |
| `tasks.py` | 52 tasks in four levels (L1 single step, L2 multi-step, L3 policy, L4 safety / prompt injection) with automatic checkers |
| `agents.py` | Three agent designs: single ReAct agent (`single`), Planner–Executor (`pe`), Planner–Executor–Verifier (`pev`) |
| `power.py` | Energy measurement from Jetson power rails via `tegrastats` |
| `run.py` | Runs a model × design grid; appends one JSON line per run to `results/runs.jsonl` (resumable) |
| `run_all.sh` | The full experiment (main grid + thinking-mode ablation) |
| `analyze.py` | Tables, figures, failure taxonomy, and EDA statistics |
| `results/` | Raw results of every run (`runs.jsonl`) and summaries |

A task succeeds only if its checker passes **and** no device outside the task's allowed set was changed.

## Requirements

- NVIDIA Jetson AGX Orin (JetPack 6) for energy measurement; the benchmark itself runs on any machine with Ollama (energy fields will then be empty)
- [Ollama](https://github.com/ollama/ollama) serving on `127.0.0.1:11434`
- Python 3.10+, `numpy`, `matplotlib` (analysis only; the runner uses the standard library)

Models used (default Q4_K_M quantization):

```bash
ollama pull qwen3:1.7b && ollama pull llama3.2:3b && ollama pull qwen3:4b-instruct-2507-q4_K_M
ollama pull granite3.3:8b && ollama pull qwen3:8b
# thinking-mode ablation
ollama pull qwen3:4b-thinking-2507-q4_K_M
```

## Run

```bash
# one model, all three designs
python3 run.py --models qwen3:4b-instruct-2507-q4_K_M --archs single pe pev

# the complete experiment (main grid + thinking-mode ablation)
./run_all.sh

# tables, figures and statistics
python3 analyze.py
```

Granite note: Ollama's Granite 3.3 template drops its tool-calling instruction when a custom system prompt is given, so `agents.py` re-adds IBM's official sentence for Granite models.

Default settings: temperature 0, fixed seed, 8192-token context, at most 12 tool-calling steps per agent, at most 1024 generated tokens per model call (4096 with Qwen3 thinking on), Jetson MAXN power mode.

Note: the plain `qwen3:4b` tag in Ollama now points to the thinking-only Qwen3-4B-Thinking-2507 model, so use the explicit tags above.

## License

MIT License — see [LICENSE](LICENSE).
