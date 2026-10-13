#!/usr/bin/env bash
# Full experiment grid: 5 models x 3 architectures x 52 tasks, then the thinking-mode ablation.
# Waits for each model to be available in Ollama; resumable via results/runs.jsonl
# (already-finished runs are skipped, so re-running this script continues where it stopped).
# Optional first argument: PID of a run.py still in progress; waits for it before starting.
cd "$(dirname "$0")"
export OLLAMA_HOST=127.0.0.1:11434
if [ -n "$1" ]; then
  while kill -0 "$1" 2>/dev/null; do sleep 10; done
fi
for m in qwen3:1.7b llama3.2:3b qwen3:4b-instruct-2507-q4_K_M granite3.3:8b qwen3:8b; do
  until ollama list | awk '{print $1}' | grep -qx "$m"; do sleep 30; done
  python3 run.py --models "$m" --archs single pe pev
done
# Ablation (single agent only; thinking makes multi-agent runs ~3x longer):
#  - Qwen3-4B: Thinking-2507 variant (thinking-only model) vs. the Instruct-2507 run above
#  - Qwen3-8B (hybrid): thinking switched on vs. off
python3 run.py --models qwen3:4b-thinking-2507-q4_K_M qwen3:8b --archs single --think on
echo ALL_DONE
