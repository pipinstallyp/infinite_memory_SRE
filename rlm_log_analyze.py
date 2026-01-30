#!/usr/bin/env python3
"""
RLM-based Log Root Cause Analyzer

This script uses DSPy's RLM (Recursive Language Model) to analyze Apache logs
and identify root causes of errors. RLM uses a REPL-based approach where the
LLM writes Python code to iteratively explore large log files and query
sub-LLMs for semantic analysis.

Models:
- Main LLM (GPT-5.2 with thinking): Used for code generation and reasoning
- Sub LLM (GLM-4.7-flash): Used for llm_query/llm_query_batched semantic analysis

Usage:
    python rlm_log_analyze.py
"""

import os
import sys
import json
import time
import threading
import shutil
import re
import urllib.request
from pathlib import Path
from typing import Any

# Load environment variables from .env file
from dotenv import load_dotenv

# Load .env from this repo (gitignored).
env_path = Path(__file__).parent / ".env"
load_dotenv(env_path)

import dspy
from dspy.predict.rlm import RLM
from dspy.primitives.repl_types import REPLHistory
from dspy.utils.callback import BaseCallback


OPENROUTER_MODELS_URL = "https://openrouter.ai/api/v1/models"

# Fallback pricing (USD per token) for the models used by default in this script.
# These can drift over time; we prefer live fetch from OpenRouter when available.
DEFAULT_PRICING_BY_OPENROUTER_ID: dict[str, dict[str, float]] = {
    "openai/gpt-5.2": {
        "prompt": 0.00000175,
        "completion": 0.000014,
        "input_cache_read": 0.000000175,
        "web_search": 0.01,
    },
    "z-ai/glm-4.7-flash": {
        "prompt": 0.00000007,
        "completion": 0.0000004,
        "input_cache_read": 0.00000001,
    },
}

_CODE_FENCE_PATTERN = re.compile(r"^```(?:python|py)?\\s*\\n(.*)\\n```\\s*$", re.DOTALL)


def _strip_code_fences(code: str) -> str:
    code = code.strip()
    match = _CODE_FENCE_PATTERN.match(code)
    if match:
        return match.group(1)
    return code


def ensure_deno_on_path() -> None:
    """DSPy RLM's default PythonInterpreter requires `deno` on PATH."""
    if shutil.which("deno"):
        return
    candidate = Path.home() / ".deno" / "bin" / "deno"
    if candidate.exists():
        os.environ["PATH"] = f"{candidate.parent}{os.pathsep}" + os.environ.get("PATH", "")


def _dspy_model_to_openrouter_id(model: str) -> str:
    # DSPy/LiteLLM uses "openrouter/<provider>/<model>".
    return model[len("openrouter/") :] if model.startswith("openrouter/") else model


def _safe_int(x: Any, default: int = 0) -> int:
    try:
        return int(x)
    except Exception:
        return default


def _safe_float(x: Any, default: float = 0.0) -> float:
    try:
        return float(x)
    except Exception:
        return default


def _extract_prompt_and_completion_tokens(usage: dict[str, Any]) -> tuple[int, int]:
    # OpenAI-style usage: prompt_tokens/completion_tokens. Some providers use input_tokens/output_tokens.
    prompt = usage.get("prompt_tokens", usage.get("input_tokens", 0))
    completion = usage.get("completion_tokens", usage.get("output_tokens", 0))
    return _safe_int(prompt), _safe_int(completion)


def fetch_openrouter_pricing(retries: int = 3, backoff_s: float = 0.75) -> dict[str, dict[str, float]]:
    """
    Fetch OpenRouter per-token pricing for all models.

    Returns:
        Map of {openrouter_model_id: {"prompt": float, "completion": float, ...}}
        Values are USD per token (not per 1K/1M tokens).
    """
    last_err: Exception | None = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(OPENROUTER_MODELS_URL, timeout=30) as resp:
                payload = json.load(resp)
            break
        except Exception as e:
            last_err = e
            if attempt < retries - 1:
                time.sleep(backoff_s * (2**attempt))
            else:
                raise last_err

    pricing: dict[str, dict[str, float]] = {}
    for item in payload.get("data", []):
        model_id = item.get("id")
        if not model_id:
            continue
        raw = item.get("pricing") or {}
        parsed: dict[str, float] = {}
        for k, v in raw.items():
            try:
                parsed[k] = float(v)
            except Exception:
                continue
        pricing[model_id] = parsed
    return pricing


def estimate_cost_usd(
    usage_by_model: dict[str, dict[str, Any]],
    pricing_by_openrouter_id: dict[str, dict[str, float]],
) -> tuple[float, dict[str, dict[str, Any]]]:
    """
    Estimate USD cost from token usage + OpenRouter pricing.

    Returns:
        (total_cost, per_model_summary)
    """
    total_cost = 0.0
    per_model: dict[str, dict[str, Any]] = {}

    for dspy_model, usage in usage_by_model.items():
        openrouter_id = _dspy_model_to_openrouter_id(dspy_model)
        prompt_tokens, completion_tokens = _extract_prompt_and_completion_tokens(usage)
        prices = pricing_by_openrouter_id.get(openrouter_id) or {}
        prompt_price = float(prices.get("prompt", 0.0) or 0.0)
        completion_price = float(prices.get("completion", 0.0) or 0.0)
        cost = (prompt_tokens * prompt_price) + (completion_tokens * completion_price)
        total_cost += cost

        per_model[openrouter_id] = {
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "total_tokens": prompt_tokens + completion_tokens,
            "prompt_usd_per_token": prompt_price,
            "completion_usd_per_token": completion_price,
            "estimated_cost_usd": cost,
        }

    return total_cost, per_model


class LiveCostCallback(BaseCallback):
    """Print per-call token usage and estimated USD cost as LMs are invoked."""

    def __init__(self, pricing_by_openrouter_id: dict[str, dict[str, float]]):
        self._pricing_by_openrouter_id = pricing_by_openrouter_id
        self._lock = threading.Lock()
        self._start_usage_by_call_id: dict[str, dict[str, dict[str, Any]]] = {}
        self._start_time_by_call_id: dict[str, float] = {}
        self._model_by_call_id: dict[str, str] = {}

    def on_lm_start(self, call_id: str, instance: Any, inputs: dict[str, Any]):
        tracker = dspy.settings.usage_tracker
        usage_snapshot = tracker.get_total_tokens() if tracker else {}
        with self._lock:
            self._start_usage_by_call_id[call_id] = usage_snapshot
            self._start_time_by_call_id[call_id] = time.time()
            self._model_by_call_id[call_id] = getattr(instance, "model", "unknown")

        model_id = _dspy_model_to_openrouter_id(getattr(instance, "model", "unknown"))
        print(f"[lm] call start: {model_id}", flush=True)

    def on_lm_end(self, call_id: str, outputs: dict[str, Any] | None, exception: Exception | None = None):
        tracker = dspy.settings.usage_tracker
        usage_now = tracker.get_total_tokens() if tracker else {}

        with self._lock:
            usage_start = self._start_usage_by_call_id.pop(call_id, {})
            start_ts = self._start_time_by_call_id.pop(call_id, None)
            dspy_model = self._model_by_call_id.pop(call_id, "unknown")

        openrouter_id = _dspy_model_to_openrouter_id(dspy_model)

        start_prompt, start_completion = _extract_prompt_and_completion_tokens(usage_start.get(dspy_model, {}))
        end_prompt, end_completion = _extract_prompt_and_completion_tokens(usage_now.get(dspy_model, {}))
        delta_prompt = max(end_prompt - start_prompt, 0)
        delta_completion = max(end_completion - start_completion, 0)
        delta_total = delta_prompt + delta_completion

        duration_s = None if start_ts is None else (time.time() - start_ts)

        if self._pricing_by_openrouter_id:
            prices = self._pricing_by_openrouter_id.get(openrouter_id) or {}
            prompt_price = float(prices.get("prompt", 0.0) or 0.0)
            completion_price = float(prices.get("completion", 0.0) or 0.0)
            delta_cost = (delta_prompt * prompt_price) + (delta_completion * completion_price)
            total_cost, _ = estimate_cost_usd(usage_now, self._pricing_by_openrouter_id)

            dur = "" if duration_s is None else f" duration={duration_s:.2f}s"
            err = "" if exception is None else f" error={type(exception).__name__}"
            print(
                f"[lm] call end:   {openrouter_id}{dur}{err} "
                f"+tokens(p/c/t)={delta_prompt}/{delta_completion}/{delta_total} "
                f"+cost=${delta_cost:.4f} total=${total_cost:.4f}",
                flush=True,
            )
        else:
            dur = "" if duration_s is None else f" duration={duration_s:.2f}s"
            err = "" if exception is None else f" error={type(exception).__name__}"
            print(
                f"[lm] call end:   {openrouter_id}{dur}{err} "
                f"+tokens(p/c/t)={delta_prompt}/{delta_completion}/{delta_total}",
                flush=True,
            )


class BudgetedRLM(RLM):
    """RLM with an approximate USD budget, enforced between iterations."""

    def __init__(
        self,
        *args,
        budget_usd: float,
        reserve_usd: float,
        pricing_by_openrouter_id: dict[str, dict[str, float]],
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.budget_usd = budget_usd
        self.reserve_usd = reserve_usd
        self.pricing_by_openrouter_id = pricing_by_openrouter_id

    def _make_llm_tools(self, max_workers: int = 8) -> dict[str, Any]:
        """
        Override RLM's llm_query_batched implementation to run sequentially.

        DSPy's default implementation uses ThreadPoolExecutor, but that can cause
        usage tracking (and our per-call cost deltas) to miss sub-LM calls because
        dspy.settings contexts don't automatically propagate to worker threads.
        """
        state = {"call_count": 0}
        lock = threading.Lock()
        lm = self.sub_lm

        def _check_and_increment(n: int = 1) -> None:
            with lock:
                if state["call_count"] + n > self.max_llm_calls:
                    raise RuntimeError(
                        f"LLM call limit exceeded: {state['call_count']} + {n} > {self.max_llm_calls}. "
                        "Use Python code for aggregation instead of making more LLM calls."
                    )
                state["call_count"] += n

        def _query_lm(prompt: str) -> str:
            target_lm = lm if lm is not None else dspy.settings.lm
            if target_lm is None:
                raise RuntimeError("No LM configured. Use dspy.configure(lm=...) or pass sub_lm to RLM.")
            response = target_lm(prompt)
            if isinstance(response, list) and response:
                item = response[0]
                if isinstance(item, dict) and "text" in item:
                    return item["text"]
                return item
            return str(response)

        def llm_query(prompt: str) -> str:
            """Query the LLM with a prompt string."""
            if not prompt:
                raise ValueError("prompt cannot be empty")
            _check_and_increment(1)
            return _query_lm(prompt)

        def llm_query_batched(prompts: list[str]) -> list[str]:
            """Query the LLM with multiple prompts (sequential)."""
            if not prompts:
                return []
            _check_and_increment(len(prompts))
            return [_query_lm(p) for p in prompts]

        return {"llm_query": llm_query, "llm_query_batched": llm_query_batched}

    def _execute_iteration(
        self,
        repl: Any,
        variables: list[Any],
        history: REPLHistory,
        iteration: int,
        input_args: dict[str, Any],
        output_field_names: list[str],
    ) -> dspy.Prediction | REPLHistory:
        """
        Like DSPy's RLM._execute_iteration, but resilient to models returning null/empty `code`.

        Some providers/models occasionally yield structured outputs where fields are present but `code`
        is null/empty; instead of crashing, we append an error to the REPL history and continue.
        """
        variables_info = [variable.format() for variable in variables]
        action = self.generate_action(
            variables_info=variables_info,
            repl_history=history,
            iteration=f"{iteration + 1}/{self.max_iterations}",
        )

        reasoning = getattr(action, "reasoning", "") or ""
        code = getattr(action, "code", "") or ""

        if not isinstance(code, str) or not code.strip():
            output = "[Error] LLM returned empty/non-string `code`. Please provide executable Python code."
            return history.append(reasoning=reasoning, code=str(code), output=output)

        try:
            code_str = _strip_code_fences(code)
            result = repl.execute(code_str, variables=dict(input_args))
        except Exception as e:
            result = f"[Error] {e}"

        return self._process_execution_result(action, result, history, output_field_names)

    def _estimated_total_cost_usd(self) -> float:
        tracker = dspy.settings.usage_tracker
        if tracker is None:
            return 0.0
        total_cost, _ = estimate_cost_usd(tracker.get_total_tokens(), self.pricing_by_openrouter_id)
        return total_cost

    def forward(self, **input_args) -> dspy.Prediction:
        self._validate_inputs(input_args)

        output_field_names = list(self.signature.output_fields.keys())
        execution_tools = self._prepare_execution_tools()
        variables = self._build_variables(**input_args)

        budget_cap = max(self.budget_usd - self.reserve_usd, 0.0)

        with self._interpreter_context(execution_tools) as repl:
            history: REPLHistory = REPLHistory()

            for iteration in range(self.max_iterations):
                est_cost = self._estimated_total_cost_usd()
                if self.verbose:
                    if self.pricing_by_openrouter_id:
                        print(
                            f"[rlm] iteration {iteration + 1}/{self.max_iterations} "
                            f"estimated_total_cost=${est_cost:.4f}",
                            flush=True,
                        )
                    else:
                        print(
                            f"[rlm] iteration {iteration + 1}/{self.max_iterations}",
                            flush=True,
                        )
                if self.pricing_by_openrouter_id and self.budget_usd > 0 and est_cost >= budget_cap:
                    if self.verbose:
                        print(
                            f"\n[budget] Reached cost cap (${est_cost:.4f} >= ${budget_cap:.2f}); forcing extract.\n"
                        )
                    break

                result = self._execute_iteration(repl, variables, history, iteration, input_args, output_field_names)
                if isinstance(result, dspy.Prediction):
                    return result
                history = result

            return self._extract_fallback(variables, history, output_field_names)


def load_log_file(log_path: str) -> str:
    """Load log file contents."""
    with open(log_path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def create_lm(model_name: str, **kwargs) -> dspy.LM:
    """
    Create a DSPy LM instance for OpenRouter.
    
    Args:
        model_name: Model name in OpenRouter format (e.g., 'openai/gpt-5.2')
        **kwargs: Additional arguments to pass to the LM
    
    Returns:
        Configured dspy.LM instance
    """
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("OPENROUTER_API_KEY not found in environment variables")
    
    # Remove any spaces from the API key
    api_key = api_key.strip()
    
    return dspy.LM(
        model=f"openrouter/{model_name}",
        api_key=api_key,
        api_base="https://openrouter.ai/api/v1",
        **kwargs
    )


class LogAnalysisSignature(dspy.Signature):
    """Analyze server logs to identify the root cause of errors and issues.
    
    You are an expert SRE (Site Reliability Engineer) analyzing server logs to 
    determine the root cause of system issues. Carefully examine the logs for 
    patterns, error messages, and correlations that indicate the underlying problem.
    """
    
    log_content: str = dspy.InputField(desc="The server log content to analyze")
    
    root_cause: str = dspy.OutputField(
        desc="A detailed explanation of the root cause of the errors in the logs. "
             "Include specific evidence from the logs, patterns observed, and "
             "technical explanation of why this is happening."
    )
    
    severity: str = dspy.OutputField(
        desc="The severity level of the issue: 'critical', 'high', 'medium', or 'low'"
    )
    
    recommendations: str = dspy.OutputField(
        desc="Specific actionable recommendations to fix the identified issue"
    )


def main():
    ensure_deno_on_path()

    # Configuration
    LOG_FOLDER = Path(__file__).parent / "log"
    LOG_FILE = LOG_FOLDER / "Apache_2k.log"
    RUNS_FOLDER = Path(os.getenv("RLM_RUN_DIR", str(Path(__file__).parent / "runs")))
    if not RUNS_FOLDER.is_absolute():
        RUNS_FOLDER = (Path(__file__).parent / RUNS_FOLDER).resolve()
    RUNS_FOLDER.mkdir(parents=True, exist_ok=True)
    RUN_ID = time.strftime("%Y%m%d_%H%M%S")
    
    # Model configuration  
    # Main model: code generation / reasoning
    MAIN_MODEL = os.getenv("RLM_MAIN_MODEL", "openai/gpt-5.2")
    
    # Sub model: GLM-4.7-flash for quick semantic queries (llm_query calls)
    SUB_MODEL = os.getenv("RLM_SUB_MODEL", "z-ai/glm-4.7-flash")

    MAIN_MAX_TOKENS = _safe_int(os.getenv("RLM_MAIN_MAX_TOKENS"), 8192)
    SUB_MAX_TOKENS = _safe_int(os.getenv("RLM_SUB_MAX_TOKENS"), 2048)
    THINKING_EFFORT = os.getenv("RLM_THINKING_EFFORT", "high")

    # Budgeting (approximate, based on OpenRouter /models pricing + token usage).
    # Reserve is kept for the final "extract" call if we stop early.
    BUDGET_USD = _safe_float(os.getenv("RLM_BUDGET_USD"), 2.0)
    RESERVE_USD = _safe_float(os.getenv("RLM_BUDGET_RESERVE_USD"), 0.35)
    MAX_ITERATIONS = _safe_int(os.getenv("RLM_MAX_ITERATIONS"), 50)
    MAX_LLM_CALLS = _safe_int(os.getenv("RLM_MAX_LLM_CALLS"), 80)
    
    print("=" * 60)
    print("RLM Log Analyzer - Root Cause Analysis")
    print("=" * 60)
    print(f"\nLog file: {LOG_FILE}")
    print(f"Run artifacts: {RUNS_FOLDER}")
    print(f"Main model (code generation): {MAIN_MODEL}")
    print(f"Sub model (semantic queries): {SUB_MODEL}")
    print(f"Budget: ${BUDGET_USD:.2f} (reserve ${RESERVE_USD:.2f})")
    print()
    
    # Check if log file exists
    if not LOG_FILE.exists():
        print(f"Error: Log file not found at {LOG_FILE}")
        sys.exit(1)
    
    # Load log content
    print("Loading log file...")
    log_content = load_log_file(LOG_FILE)
    print(f"Loaded {len(log_content):,} characters ({len(log_content.splitlines())} lines)")
    print()
    
    # Create LM instances
    print("Initializing language models...")
    
    main_kwargs = {"max_tokens": MAIN_MAX_TOKENS}
    # Only attach OpenAI-style "reasoning.effort" for gpt-5 family (avoid surprising other models).
    if MAIN_MODEL.startswith("openai/gpt-5"):
        main_kwargs["extra_body"] = {"reasoning": {"effort": THINKING_EFFORT}}
    main_lm = create_lm(MAIN_MODEL, **main_kwargs)
    
    # Sub LM for quick semantic queries (fast model for llm_query calls)
    sub_lm = create_lm(SUB_MODEL, max_tokens=SUB_MAX_TOKENS)
    
    pricing_by_openrouter_id: dict[str, dict[str, float]] = dict(DEFAULT_PRICING_BY_OPENROUTER_ID)
    try:
        pricing_by_openrouter_id.update(fetch_openrouter_pricing())
    except Exception as e:
        print(
            "Warning: failed to fetch OpenRouter pricing; using embedded fallback pricing "
            f"for {', '.join(sorted(DEFAULT_PRICING_BY_OPENROUTER_ID.keys()))}. Error: {e}"
        )

    # Configure DSPy with the main LM and a live cost logger.
    # TwoStepAdapter makes structured outputs more reliable across providers/models
    # by using the sub-LM to extract the required fields.
    adapter = dspy.TwoStepAdapter(sub_lm)
    dspy.configure(
        lm=main_lm,
        adapter=adapter,
        callbacks=[LiveCostCallback(pricing_by_openrouter_id)],
    )
    
    print("Models initialized successfully!")
    print()
    
    # Create RLM instance
    # - Main LM (configured globally) is used for generate_action (code generation)
    # - sub_lm is used for llm_query/llm_query_batched (semantic analysis)
    print("Creating RLM analyzer...")

    rlm_analyzer = BudgetedRLM(
        signature=LogAnalysisSignature,
        max_iterations=MAX_ITERATIONS,  # Max REPL iterations (budget may stop earlier)
        max_llm_calls=MAX_LLM_CALLS,    # Max sub-LLM calls (llm_query)
        max_output_chars=50000, # Max output chars to show per iteration
        verbose=True,           # Show detailed execution logs
        sub_lm=sub_lm,          # Small/fast model for semantic queries
        budget_usd=BUDGET_USD,
        reserve_usd=RESERVE_USD,
        pricing_by_openrouter_id=pricing_by_openrouter_id,
    )
    print("RLM analyzer created!")
    print()
    
    # Run analysis
    print("=" * 60)
    print("Starting RLM Analysis...")
    print("=" * 60)
    print()
    print("The RLM will iteratively write Python code to:")
    print("  1. Explore and understand the log structure")
    print("  2. Identify error patterns and frequencies")
    print("  3. Use llm_query() for semantic analysis of log entries")
    print("  4. Correlate events to find root causes")
    print("  5. Submit final analysis with recommendations")
    print()
    print("-" * 60)
    
    try:
        with dspy.track_usage() as usage_tracker:
            result = rlm_analyzer(log_content=log_content)
            usage_by_model = usage_tracker.get_total_tokens()

        # Save trajectory for debugging (even if later printing fails).
        if hasattr(result, "trajectory"):
            trajectory_file = RUNS_FOLDER / f"analysis_trajectory_{RUN_ID}.json"
            with open(trajectory_file, "w") as f:
                json.dump(result.trajectory, f, indent=2)
            print(f"\n📁 Analysis trajectory saved to: {trajectory_file}")
        
        print()
        print("=" * 60)
        print("ANALYSIS RESULTS")
        print("=" * 60)
        
        print("\n📋 ROOT CAUSE:")
        print("-" * 40)
        print(getattr(result, "root_cause", None) or "N/A")
        
        severity = getattr(result, "severity", None)
        severity_text = (severity or "").strip()
        print(f"\n⚠️  SEVERITY: {(severity_text.upper() if severity_text else 'N/A')}")
        
        print("\n💡 RECOMMENDATIONS:")
        print("-" * 40)
        print(getattr(result, "recommendations", None) or "N/A")

        if usage_by_model:
            print("\nUSAGE" + (" & COST (estimated)" if pricing_by_openrouter_id else ""))
            print("-" * 40)

            if pricing_by_openrouter_id:
                total_cost_usd, per_model = estimate_cost_usd(usage_by_model, pricing_by_openrouter_id)
                for model_id in sorted(per_model.keys()):
                    m = per_model[model_id]
                    print(
                        f"{model_id}: prompt={m['prompt_tokens']:,} completion={m['completion_tokens']:,} "
                        f"total={m['total_tokens']:,} cost=${m['estimated_cost_usd']:.4f}"
                    )
                print(f"TOTAL estimated cost: ${total_cost_usd:.4f} (budget ${BUDGET_USD:.2f})")
            else:
                for dspy_model, usage in sorted(usage_by_model.items()):
                    prompt_tokens, completion_tokens = _extract_prompt_and_completion_tokens(usage)
                    model_id = _dspy_model_to_openrouter_id(dspy_model)
                    print(
                        f"{model_id}: prompt={prompt_tokens:,} completion={completion_tokens:,} "
                        f"total={prompt_tokens + completion_tokens:,}"
                    )
    except Exception as e:
        print(f"\n❌ Error during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
    
    print()
    print("=" * 60)
    print("Analysis Complete!")
    print("=" * 60)


if __name__ == "__main__":
    main()
