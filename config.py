"""
Configuration module for the Multi-Agent Question-Answering System.

Defines default settings, model names, API keys, and worker agent personas.
"""

import os
from pathlib import Path

# Base directory for the project
BASE_DIR = Path(__file__).resolve().parent

# Load environment variables from .env if present
_env_file = BASE_DIR / ".env"
try:
    from dotenv import load_dotenv
    load_dotenv(dotenv_path=_env_file)
except ImportError:
    # Minimal fallback parser if python-dotenv is not installed
    if _env_file.exists():
        with open(_env_file, "r", encoding="utf-8") as _f:
            for _line in _f:
                _line = _line.strip()
                if not _line or _line.startswith("#") or "=" not in _line:
                    continue
                _key, _val = _line.split("=", 1)
                _key = _key.strip()
                _val = _val.strip().strip('"').strip("'")
                if _key and _key not in os.environ:
                    os.environ[_key] = _val

# Default LLM model (can be overridden via OPENAI_MODEL env var or CLI)
DEFAULT_MODEL = os.environ.get("OPENAI_MODEL", "gpt-4o").strip() or "gpt-4o"

# OpenAI API Key from environment
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY", "").strip()

# Optional custom base URL for OpenAI-compatible proxies / local models
OPENAI_BASE_URL = os.environ.get("OPENAI_BASE_URL", "").strip() or None

# Check if we should default to offline mock mode
DEFAULT_USE_MOCK = not bool(OPENAI_API_KEY)

# ---------------------------------------------------------------------------
# Worker Agent Personas
# ---------------------------------------------------------------------------
# To prevent "carbon copy" responses when querying the same LLM, each worker
# is given a distinct analytical persona. This generates authentic reasoning
# diversity while attacking the question from different cognitive angles.
# ---------------------------------------------------------------------------

WORKER_PERSONAS = [
    {
        "id": "agent_1",
        "name": "Agent 1",
        "role": "First-Principles Reasoner",
        "system_prompt": (
            "You are an analytical AI reasoning agent. You evaluate problems strictly from "
            "first principles, formal logic, and literal definitions. You break down each "
            "premise, test validity step-by-step, and strictly avoid intuitive assumptions. "
            "You must return your response in the specified JSON format."
        ),
        "temperature": 0.2,
    },
    {
        "id": "agent_2",
        "name": "Agent 2",
        "role": "Critical Skeptic & Edge-Case Hunter",
        "system_prompt": (
            "You are a critical, skeptical AI reasoning agent. Your mission is to actively "
            "hunt for semantic traps, hidden premises, counter-intuitive edge cases, and "
            "common human cognitive biases. You play devil's advocate before concluding. "
            "You must return your response in the specified JSON format."
        ),
        "temperature": 0.5,
    },
    {
        "id": "agent_3",
        "name": "Agent 3",
        "role": "Pragmatic Contextualist",
        "system_prompt": (
            "You are a pragmatic, common-sense AI reasoning agent. You interpret problems "
            "through standard real-world context, conversational pragmatics, and high-probability "
            "scenarios. You weigh practical plausibility against rigid formalisms. "
            "You must return your response in the specified JSON format."
        ),
        "temperature": 0.7,
    },
]

# System prompt for the Reconciliation Agent (Judge)
RECONCILIATION_SYSTEM_PROMPT = (
    "You are an expert Reconciliation Agent and Master Arbitrator. "
    "Your objective is to evaluate answers and reasoning chains from 3 independent AI worker agents, "
    "identify points of agreement and disagreement, expose any logical fallacies or missed edge cases, "
    "and determine the definitive final answer.\n\n"
    "CRITICAL RULE: Do NOT simply adopt majority rule. If one agent identified a subtle trap or edge case "
    "that the other two missed, rule in favor of the soundest logic even if it is a minority opinion. "
    "You must return your evaluation strictly in the specified JSON format."
)

# ---------------------------------------------------------------------------
# Pre-configured Sample Benchmark Questions
# ---------------------------------------------------------------------------
SAMPLE_QUESTIONS = [
    {
        "title": "Disagreement Demo: Gold vs Feathers trap",
        "question": "Which is heavier: a pound of gold or a pound of feathers?",
        "note": "Demonstrates qualitative arbitration when 2 agents give colloquial answers ('equal') and 1 catches the Troy vs Avoirdupois weight trap (feathers are heavier: ~453.6g vs ~373.2g).",
    },
    {
        "title": "Cognitive Bias Demo: Bat and Ball problem",
        "question": (
            "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
            "How much does the ball cost?"
        ),
        "note": "Demonstrates unanimous algebraic verification debunking the fast intuitive trap of $0.10, correctly solving for $0.05.",
    },
    {
        "title": "Probability Demo: Monty Hall problem",
        "question": (
            "In the Monty Hall problem, if you choose Door 1 and the host opens Door 3 to reveal a goat, "
            "should you switch to Door 2 or stay with Door 1? Explain the probabilities."
        ),
        "note": "Demonstrates cross-agent alignment across probabilistic, conditional filtering, and game-theoretic reasoning.",
    },
    {
        "title": "UPSC Prelims Demo: Legal & Constitutional Status of the Preamble (CSE 2020)",
        "question": (
            "The Preamble to the Constitution of India is:\n"
            "(a) a part of the Constitution but has no legal effect\n"
            "(b) not a part of the Constitution and has no legal effect either\n"
            "(c) a part of the Constitution and has the same legal effect as any other part\n"
            "(d) a part of the Constitution but has no legal effect independently of other parts"
        ),
        "note": (
            "Demonstrates resolving a classic UPSC controversy: Agent 1 falls into the non-justiciability trap (a), "
            "while Agent 2 & Agent 3 identify that it has interpretive legal effect in conjunction with other provisions (d)."
        ),
    },
]

