"""
orchestrator.py

Coordinates the multi-agent question-answering workflow:
  1. The orchestrator starts
  2. Independent agents are called (parallel fan-out)
  3. Their outputs are collected (fan-in)
  4. Reconciliation happens
  5. The final answer is produced
"""

import asyncio
from typing import Any, Dict, List
from config import DEFAULT_MODEL, WORKER_PERSONAS
from worker_agent import WorkerAgent
from reconciliation_agent import ReconciliationAgent


class Orchestrator:
    """
    Coordinates the fan-out / fan-in execution flow across independent
    workers and the reconciliation judge.
    """

    def __init__(self, model: str = DEFAULT_MODEL, use_mock: bool = False):
        self.model = model
        self.use_mock = use_mock

        # Create 3 independent worker agents with diverse analytical personas
        self.workers: List[WorkerAgent] = [
            WorkerAgent(
                name=persona["name"],
                role=persona["role"],
                system_prompt=persona["system_prompt"],
                temperature=persona["temperature"],
                model=self.model,
                use_mock=self.use_mock,
            )
            for persona in WORKER_PERSONAS
        ]

        # Create the separate reconciliation agent
        self.reconciler = ReconciliationAgent(
            name="Reconciliation Agent",
            model=self.model,
            use_mock=self.use_mock,
        )

    async def run(self, question: str) -> Dict[str, Any]:
        """
        Executes the 5-step multi-agent orchestration lifecycle.
        """

        # =====================================================================
        # 1. THE ORCHESTRATOR STARTS
        # =====================================================================
        # The orchestrator receives the user's question, logs initialization,
        # and prepares independent tasks for each worker agent.
        # At this stage, no worker has been contacted yet.

        # =====================================================================
        # 2. INDEPENDENT AGENTS ARE CALLED (PARALLEL FAN-OUT)
        # =====================================================================
        # The orchestrator dispatches the question to all 3 worker agents
        # simultaneously using asyncio.gather.
        # KEY ARCHITECTURAL PRINCIPLE:
        # Each agent runs in strict cognitive isolation. Agent 1 cannot see
        # Agent 2's or Agent 3's reasoning, preventing anchoring bias and groupthink.
        tasks = [worker.solve(question) for worker in self.workers]

        # =====================================================================
        # 3. THEIR OUTPUTS ARE COLLECTED (FAN-IN)
        # =====================================================================
        # The orchestrator waits for all worker agents to finish their independent
        # analysis and collects their structured JSON outputs:
        #   [ {answer, reasoning, confidence}, ... ]
        worker_responses: List[Dict[str, Any]] = await asyncio.gather(*tasks)

        # =====================================================================
        # 4. RECONCILIATION HAPPENS
        # =====================================================================
        # All 3 collected responses, along with the original question, are
        # handed over to the Reconciliation Agent.
        # The Reconciliation Agent:
        #   - Compares the answers and reasoning
        #   - Identifies any points of disagreement or edge cases
        #   - Decides the final answer via qualitative arbitration (not simple voting)
        #   - Explains the rationale for the verdict
        reconciliation_result: Dict[str, Any] = await self.reconciler.reconcile(
            question, worker_responses
        )

        # =====================================================================
        # 5. THE FINAL ANSWER IS PRODUCED
        # =====================================================================
        # The orchestrator packages the complete audit trail (the question,
        # all independent worker outputs, and the final reconciled verdict)
        # and returns it to the caller for presentation.
        return {
            "question": question,
            "worker_responses": worker_responses,
            "reconciliation": reconciliation_result,
        }
