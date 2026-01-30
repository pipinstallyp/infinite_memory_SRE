# infinite_memory_SRE (experiment)

RLM-based (DSPy) log root-cause analysis experiment.

## Quick start

1. Create a `.env` file with your OpenRouter key:

   - `OPENROUTER_API_KEY=...`

2. Install deps with `uv`:

   ```bash
   uv venv .venv
   uv pip install -r requirements.txt
   ```

3. Run:

   ```bash
   .venv/bin/python rlm_log_analyze.py
   ```

Artifacts (run logs, trajectories) are written under `runs/`. Input logs live under `log/`.

