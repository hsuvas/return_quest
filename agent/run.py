"""
CLI entry point for the agent simulation system.

Usage::

    python src/agent/run.py \\
        --input_path ./dataset/outputs/scenarios_03/scenarios.jsonl \\
        --output_dir ./outputs/agent_run_01 \\
        --agent_model gpt-4.1-2025-04-14 \\
        --customer_model gpt-4.1-2025-04-14 \\
        --max_turns 10 \\
        --agent_persona RANDOM \\
        --use_native_tools \\
        --use_academic_ai \\
        --num_scenarios 20 \\
        --concurrency 2
"""

import argparse
import json
import os
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Dict, List

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------

_PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
_SRC_DIR = str(Path(__file__).resolve().parent.parent)

if _SRC_DIR not in sys.path:
    sys.path.insert(0, _SRC_DIR)

from dotenv import load_dotenv  # noqa: E402

from agent.academic_ai_client import AcademicAIClient  # noqa: E402
from agent.huggingface_client import HuggingFaceClient  # noqa: E402
from agent.llm_provider import LLMProvider  # noqa: E402
from agent.orchestrator import run_multitype_conversations, AGENT_PERSONAS  # noqa: E402


# ---------------------------------------------------------------------------
# Scenario loading
# ---------------------------------------------------------------------------

def load_scenarios(path: str) -> List[Dict[str, Any]]:
    """Load scenarios from a JSONL or JSON file."""
    p = Path(path)
    if not p.exists():
        raise FileNotFoundError(f"Input file not found: {path}")

    if p.suffix.lower() == ".jsonl":
        items: List[Dict[str, Any]] = []
        with open(p, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(json.loads(line))
        return items

    if p.suffix.lower() == ".json":
        with open(p, "r", encoding="utf-8") as f:
            data = json.load(f)
        return data if isinstance(data, list) else [data]

    raise ValueError(f"Unsupported file format: {p.suffix}")


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def _load_completion_status(output_dir: str) -> Dict[str, bool]:
    """Load already-completed scenario IDs from the per-scenario dir."""
    per_dir = os.path.join(output_dir, "scenarios")
    status: Dict[str, bool] = {}
    if os.path.isdir(per_dir):
        for fname in os.listdir(per_dir):
            if fname.endswith(".json"):
                status[fname[:-5]] = True
    return status


def _mark_complete(
    output_dir: str,
    row_id: str,
    status: Dict[str, bool],
) -> None:
    status[row_id] = True


def _rebuild_jsonl(per_scenario_dir: str, out_jsonl: str) -> None:
    """Rebuild the aggregated JSONL from all per-scenario JSON files."""
    files = sorted(f for f in os.listdir(per_scenario_dir) if f.endswith(".json"))
    with open(out_jsonl, "w", encoding="utf-8") as out:
        for fname in files:
            fpath = os.path.join(per_scenario_dir, fname)
            with open(fpath, "r", encoding="utf-8") as f:
                rec = json.load(f)
            out.write(json.dumps(rec, ensure_ascii=False) + "\n")
    print(f"[REBUILD] Wrote {len(files)} records to {out_jsonl}")


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Agent simulation system — generates customer support conversations",
    )

    # I/O
    p.add_argument("--input_path", required=True, help="Path to scenarios JSONL/JSON")
    p.add_argument("--output_dir", required=True, help="Output directory")

    # Model configuration
    p.add_argument("--agent_model", default="gpt-4.1-2025-04-14")
    p.add_argument("--customer_model", default="gpt-4.1-2025-04-14")

    # Temperature
    p.add_argument("--agent_temperature", type=float, default=0.7)
    p.add_argument("--customer_temperature", type=float, default=0.8)
    p.add_argument("--top_p", type=float, default=1.0)

    # Conversation settings
    p.add_argument("--max_turns", type=int, default=10)
    p.add_argument("--num_scenarios", type=int, default=None,
                    help="Limit number of scenarios to process")
    p.add_argument("--num_variants", type=int, default=1,
                    help="Number of conversation variants per scenario (default 1 = single-conversation mode)")
    p.add_argument("--max_output_tokens_agent", type=int, default=2500)
    p.add_argument("--max_output_tokens_customer", type=int, default=800)

    # Agent persona
    p.add_argument("--agent_persona", default="RANDOM",
                    choices=AGENT_PERSONAS + ["RANDOM"],
                    help="Agent persona (or RANDOM to sample per scenario)")

    # Tool calling mode
    p.add_argument("--use_native_tools", action="store_true", default=True,
                    help="Use native function calling API")
    p.add_argument("--no_native_tools", dest="use_native_tools",
                    action="store_false",
                    help="Inject tools into prompt instead of using API")

    # Resolution output control
    p.add_argument("--include_resolution", action="store_true", default=True,
                    help="Include resolution in output (default)")
    p.add_argument("--no_resolution", dest="include_resolution",
                    action="store_false",
                    help="Strip resolution from output (agent still reasons internally)")

    # Prompt size control (useful for providers with low free-tier context limits)
    p.add_argument("--max_policy_chars", type=int, default=None,
                   help="Truncate primary policy text to this many characters. "
                        "Use ~5500 for OpenRouter free tier (11k token limit).")

    # Execution
    p.add_argument("--concurrency", type=int, default=2)
    p.add_argument("--sleep_s", type=float, default=0.0)
    p.add_argument("--fresh", action="store_true", default=False,
                    help="Discard previous outputs and start from scratch")

    # Academic AI
    p.add_argument("--use_academic_ai", action="store_true", default=False,
                    help="Use Academic AI as primary provider (falls back to OpenAI)")

    # HuggingFace fallback
    p.add_argument("--use_huggingface_fallback", action="store_true", default=False,
                    help="Use HuggingFace Inference API as fallback when LiteLLM fails")

    # Environment
    p.add_argument("--env_path", type=str, default=None,
                    help="Path to .env file (defaults to configs/.env)")

    return p.parse_args()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    args = parse_args()

    # Load environment variables
    if args.env_path:
        load_dotenv(args.env_path)
    else:
        default_env = _PROJECT_ROOT / "configs" / ".env"
        if default_env.exists():
            load_dotenv(default_env)

    # Validate required API keys early
    has_api_key = (
        os.environ.get("OPENAI_API_KEY")
        or os.environ.get("OPENROUTER_API_KEY")
    )
    if not args.use_academic_ai and not has_api_key:
        print("ERROR: No API key set. Add OPENAI_API_KEY or OPENROUTER_API_KEY to configs/.env, or use --use_academic_ai.")
        sys.exit(1)

    # Academic AI client (primary provider when enabled)
    academic_client = None
    if args.use_academic_ai:
        client_id = os.environ.get("ACADEMIC_CLIENT_ID", "")
        client_secret = os.environ.get("ACADEMIC_CLIENT_SECRET", "")
        if not client_id or not client_secret:
            print("ERROR: --use_academic_ai requires ACADEMIC_CLIENT_ID and "
                  "ACADEMIC_CLIENT_SECRET in the environment / .env file")
            sys.exit(1)
        academic_client = AcademicAIClient(
            client_id=client_id,
            client_secret=client_secret,
        )
        print("[AcademicAI] Primary provider enabled")

    # HuggingFace Inference API client (required for huggingface/ models; optional fallback)
    hf_client = None
    if args.use_huggingface_fallback or args.agent_model.startswith("huggingface/"):
        hf_token = os.environ.get("HF_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN") or os.environ.get("HUGGINGFACE_API_KEY", "")
        if not hf_token:
            print("ERROR: HuggingFace models require HF_TOKEN or HUGGINGFACE_TOKEN in the environment / .env file")
            sys.exit(1)
        hf_client = HuggingFaceClient(token=hf_token)
        print("[HuggingFace] Provider enabled")

    if os.environ.get("OPENROUTER_API_KEY"):
        print("[OpenRouter] OPENROUTER_API_KEY found — will use OpenRouter for non-HuggingFace models")

    agent_model = args.agent_model
    customer_model = args.customer_model

    # Build output directory with model name and resolution type
    model_tag = args.agent_model.replace("/", "_").replace(":", "_")
    resolution_tag = "with_resolution" if args.include_resolution else "no_resolution"
    output_dir = os.path.join(args.output_dir, model_tag, resolution_tag)

    # Output directories
    os.makedirs(output_dir, exist_ok=True)
    per_scenario_dir = os.path.join(output_dir, "scenarios")
    os.makedirs(per_scenario_dir, exist_ok=True)

    # Load scenarios
    scenarios = load_scenarios(args.input_path)
    if args.num_scenarios:
        scenarios = scenarios[: args.num_scenarios]

    if args.max_policy_chars:
        for sc in scenarios:
            txt = sc["Policy"]["Primary Policy"]["text"]
            if len(txt) > args.max_policy_chars:
                sc["Policy"]["Primary Policy"]["text"] = txt[: args.max_policy_chars]
        print(f"[Truncate] Primary policy text capped at {args.max_policy_chars} chars")

    print(f"Output dir: {output_dir}")
    print(f"Loaded {len(scenarios)} scenarios")
    print(f"Agent model: {args.agent_model}")
    print(f"Variants per scenario: {args.num_variants}")
    print(f"Mode: {'single-conversation' if args.num_variants == 1 else 'multi-variant'}")
    print(f"Include resolution: {args.include_resolution}")
    print(f"Agent persona: {args.agent_persona}")
    print(f"Native tools: {args.use_native_tools}")
    print(f"Academic AI: {args.use_academic_ai}")
    print(f"HuggingFace fallback: {args.use_huggingface_fallback}")
    print(f"Concurrency: {args.concurrency}")

    # Resume support — on by default, use --fresh to start over
    out_jsonl = os.path.join(output_dir, "conversations.jsonl")

    if args.fresh:
        # Wipe previous per-scenario files and JSONL
        for fname in os.listdir(per_scenario_dir):
            if fname.endswith(".json"):
                os.remove(os.path.join(per_scenario_dir, fname))
        if os.path.exists(out_jsonl):
            os.remove(out_jsonl)
        completion_status: Dict[str, bool] = {}
        print("[FRESH] Cleared previous outputs")
    else:
        completion_status = _load_completion_status(output_dir)
        n_done = len(completion_status)
        if n_done:
            print(f"[RESUME] {n_done} scenarios already completed, skipping them")

    failures: List[Dict[str, Any]] = []

    t_start = time.time()

    def _process_scenario(
        idx: int,
        sc: Dict[str, Any],
        persona: str,
    ) -> Dict[str, Any]:
        provider = LLMProvider(
            model=agent_model,
            temperature=args.agent_temperature,
            max_tokens=args.max_output_tokens_agent,
            top_p=args.top_p,
            academic_ai_client=academic_client,
            huggingface_client=hf_client,
        )
        return run_multitype_conversations(
            scenario=sc,
            llm_provider=provider,
            agent_persona=persona,
            num_variants=args.num_variants,
            max_turns=args.max_turns,
            use_native_tools=args.use_native_tools,
            sleep_between_turns=args.sleep_s,
            include_resolution=args.include_resolution,
        )

    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        futures: Dict[Any, tuple] = {}
        for idx, sc in enumerate(scenarios, start=1):
            scenario_id = sc.get("scenario_id", f"scenario_{idx:05d}")
            row_id = scenario_id

            if completion_status.get(row_id):
                continue

            persona = (
                random.choice(AGENT_PERSONAS)
                if args.agent_persona == "RANDOM"
                else args.agent_persona
            )

            fut = executor.submit(_process_scenario, idx, sc, persona)
            futures[fut] = (idx, row_id, scenario_id)

        for fut in as_completed(futures):
            idx, row_id, scenario_id = futures[fut]
            try:
                rec = fut.result()
                rec["row_id"] = row_id
                rec["user_idx"] = idx
                rec["scenario_idx"] = idx

                # Save per-scenario file (source of truth for resume)
                row_path = os.path.join(per_scenario_dir, f"{row_id}.json")
                with open(row_path, "w") as rf:
                    json.dump(rec, rf, indent=2, ensure_ascii=False)

                _mark_complete(output_dir, row_id, completion_status)
                print(f"[{idx}/{len(scenarios)}] Saved: {row_id}")

            except Exception as e:
                print(f"[{idx}/{len(scenarios)}] FAILED: {row_id} -> {e}")
                failures.append({"row_id": row_id, "error": str(e)})

    # Rebuild JSONL from all per-scenario files (including previous runs)
    _rebuild_jsonl(per_scenario_dir, out_jsonl)

    # Save failures
    if failures:
        fail_path = os.path.join(output_dir, "failures.json")
        with open(fail_path, "w") as f:
            json.dump(failures, f, indent=2)
        print(f"\n{len(failures)} failures saved to {fail_path}")

    total_completed = len([f for f in os.listdir(per_scenario_dir) if f.endswith(".json")])
    elapsed = time.time() - t_start
    print(f"\nDone. {total_completed} total completed scenarios in {elapsed:.1f}s ({elapsed / 60:.1f} min)")
    print(f"Output: {out_jsonl}")


if __name__ == "__main__":
    main()
