"""
main.py

Terminal interface for the Multi-Agent Question-Answering System.
Accepts user questions, runs the multi-agent orchestration pipeline,
and prints the complete flow clearly in the terminal.
"""

import argparse
import asyncio
import sys
from typing import Any, Dict
from config import DEFAULT_MODEL, DEFAULT_USE_MOCK, OPENAI_API_KEY
from orchestrator import Orchestrator

# Pre-configured sample questions showcasing multi-agent dynamics
SAMPLE_QUESTIONS = [
    {
        "title": "Disagreement Demo: 'Which is heavier: a pound of gold or a pound of feathers?'",
        "question": "Which is heavier: a pound of gold or a pound of feathers?",
        "note": "Demonstrates arbitration when 2 agents give colloquial answers and 1 catches the Troy vs Avoirdupois trap.",
    },
    {
        "title": "Cognitive Bias Demo: Bat and Ball problem",
        "question": (
            "A bat and a ball cost $1.10 in total. The bat costs $1.00 more than the ball. "
            "How much does the ball cost?"
        ),
        "note": "Demonstrates unanimous algebraic verification debunking the intuitive $0.10 trap.",
    },
    {
        "title": "Probability Demo: Monty Hall problem",
        "question": (
            "In the Monty Hall problem, if you choose Door 1 and the host opens Door 3 to reveal a goat, "
            "should you switch to Door 2 or stay with Door 1? Explain the probabilities."
        ),
        "note": "Demonstrates alignment across probabilistic and game-theoretic reasoning.",
    },
]


def print_result_flow(result: Dict[str, Any]) -> None:
    """
    Renders the multi-agent question answering flow to stdout
    matching the required format.
    """
    question = result["question"]
    worker_responses = result["worker_responses"]
    reconciliation = result["reconciliation"]

    print("\n" + "=" * 60)
    print("QUESTION")
    print("-" * 60)
    print(question.strip())
    print("-" * 60)

    for resp in worker_responses:
        agent_header = resp.get("agent", "Agent")
        role = resp.get("role", "")
        if role:
            title = f"{agent_header} ({role})"
        else:
            title = agent_header

        print(f"\n{title}")
        print(f"Answer: {resp.get('answer', 'N/A')}")
        print(f"Reasoning: {resp.get('reasoning', 'N/A')}")
        confidence = resp.get("confidence", 0.0)
        print(f"Confidence: {confidence:.2f}")

    print("\n" + "=" * 60)
    print("RECONCILIATION")
    print("-" * 60)
    print(f"Final Answer: {reconciliation.get('final_answer', 'N/A')}")
    print(f"Reasoning:\n{reconciliation.get('reasoning', 'N/A')}")
    final_conf = reconciliation.get("confidence", 0.0)
    print(f"Confidence: {final_conf:.2f}")
    print("=" * 60 + "\n")


async def run_single_question(question: str, model: str, use_mock: bool) -> None:
    """Executes a single question through the orchestrator and prints results."""
    orchestrator = Orchestrator(model=model, use_mock=use_mock)
    mode_label = "OFFLINE SIMULATION (Mock Mode)" if use_mock else f"LIVE OPENAI ({model})"
    print(f"\n[Running Pipeline] Mode: {mode_label}...")
    result = await orchestrator.run(question)
    print_result_flow(result)


async def interactive_loop(model: str, use_mock: bool) -> None:
    """Runs an interactive terminal loop for exploring multi-agent Q&A."""
    mode_label = "OFFLINE SIMULATION (Mock Mode)" if use_mock else f"LIVE OPENAI ({model})"

    print("\n" + "#" * 60)
    print("  MULTI-AGENT QUESTION-ANSWERING SYSTEM PROTOTYPE")
    print(f"  Mode: {mode_label}")
    if use_mock and not OPENAI_API_KEY:
        print("  (Tip: Set OPENAI_API_KEY environment variable to use live API)")
    print("#" * 60)

    while True:
        print("\nSelect an option:")
        for idx, sample in enumerate(SAMPLE_QUESTIONS, 1):
            print(f"  [{idx}] {sample['title']}")
        print(f"  [{len(SAMPLE_QUESTIONS) + 1}] Enter a custom question")
        print(f"  [{len(SAMPLE_QUESTIONS) + 2}] Exit")

        choice = input("\nEnter choice [1-5]: ").strip()

        if choice in [str(i) for i in range(1, len(SAMPLE_QUESTIONS) + 1)]:
            sample = SAMPLE_QUESTIONS[int(choice) - 1]
            print(f"\n>>> Selected: {sample['title']}")
            print(f">>> Note: {sample['note']}")
            await run_single_question(sample["question"], model, use_mock)

        elif choice == str(len(SAMPLE_QUESTIONS) + 1):
            custom_q = input("\nEnter your question: ").strip()
            if not custom_q:
                print("Question cannot be empty.")
                continue
            await run_single_question(custom_q, model, use_mock)

        elif choice in [str(len(SAMPLE_QUESTIONS) + 2), "q", "quit", "exit"]:
            print("\nExiting Multi-Agent QA System. Goodbye!\n")
            break
        else:
            print("Invalid selection. Please try again.")


def main():
    parser = argparse.ArgumentParser(
        description="Simple Multi-Agent Question-Answering System Prototype"
    )
    parser.add_argument(
        "-q", "--question",
        type=str,
        help="Run a single question directly without entering interactive mode",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"OpenAI model to use (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        default=DEFAULT_USE_MOCK,
        help="Force offline mock simulation mode even if OPENAI_API_KEY is present",
    )

    args = parser.parse_args()

    # If --mock was explicitly not given, use DEFAULT_USE_MOCK
    use_mock = args.mock

    try:
        if args.question:
            asyncio.run(run_single_question(args.question, args.model, use_mock))
        else:
            asyncio.run(interactive_loop(args.model, use_mock))
    except KeyboardInterrupt:
        print("\nOperation cancelled by user.")
        sys.exit(0)


if __name__ == "__main__":
    main()
