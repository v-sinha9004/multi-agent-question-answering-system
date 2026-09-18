"""
reconciliation_agent.py

Defines the ReconciliationAgent (Judge / Arbitrator).
Receives the question and all worker responses, evaluates conflicting logic,
identifies agreements and disagreements, and decides the definitive final answer.
"""

import asyncio
import json
import re
from typing import Any, Dict, List, Optional
from config import DEFAULT_MODEL, OPENAI_API_KEY, OPENAI_BASE_URL, RECONCILIATION_SYSTEM_PROMPT

try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

try:
    from openai import AsyncOpenAI
    HAS_OPENAI_SDK = True
except ImportError:
    HAS_OPENAI_SDK = False


if HAS_PYDANTIC:
    class ReconciliationOutput(BaseModel):
        """Strict Structured Output schema for the Reconciliation Agent."""
        final_answer: str = Field(description="Your chosen definitive final answer")
        reasoning: str = Field(description="Comparison of agents, points of disagreement, and justification")
        confidence: float = Field(description="Confidence score between 0.0 and 1.0")


class ReconciliationAgent:
    """
    Arbitrates among independent worker agents.
    Examines arguments, detects fallacies, and produces a reconciled verdict.
    """

    def __init__(
        self,
        name: str = "Reconciliation Agent",
        model: str = DEFAULT_MODEL,
        use_mock: bool = False,
        api_key: Optional[str] = None,
    ):
        self.name = name
        self.model = model
        self.use_mock = use_mock
        self.api_key = api_key or OPENAI_API_KEY

        self._openai_client = None
        if not self.use_mock and self.api_key and HAS_OPENAI_SDK:
            kwargs = {"api_key": self.api_key}
            if OPENAI_BASE_URL:
                kwargs["base_url"] = OPENAI_BASE_URL
            self._openai_client = AsyncOpenAI(**kwargs)

    async def reconcile(
        self, question: str, worker_responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Takes worker outputs, compares them, and returns:
        {
            "final_answer": str,
            "reasoning": str,
            "confidence": float
        }
        """
        if self.use_mock or not OPENAI_API_KEY:
            await asyncio.sleep(0.4)
            return self._mock_reconcile(question, worker_responses)

        return await self._call_llm(question, worker_responses)

    async def _call_llm(
        self, question: str, worker_responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """Sends worker responses to OpenAI for qualitative evaluation."""
        worker_summary = ""
        for i, resp in enumerate(worker_responses, 1):
            worker_summary += (
                f"\n--- {resp.get('agent', f'Agent {i}')} ({resp.get('role', 'Worker')}) ---\n"
                f"Answer: {resp.get('answer')}\n"
                f"Reasoning: {resp.get('reasoning')}\n"
                f"Confidence: {resp.get('confidence')}\n"
            )

        prompt = (
            f"Original Question:\n{question}\n\n"
            f"Independent Worker Agent Responses:\n{worker_summary}\n\n"
            "Your Task as Reconciliation Agent:\n"
            "1. Compare the answers provided by all agents.\n"
            "2. Compare their underlying reasoning chains.\n"
            "3. Explicitly identify any disagreements or contrasting assumptions.\n"
            "4. Decide the single best final answer based on logical validity, not mere voting.\n"
            "5. Explain clearly why you selected this answer and why any dissenting opinions were accepted or rejected.\n\n"
            "Respond ONLY with a valid JSON object matching this schema:\n"
            "{\n"
            '  "final_answer": "<your chosen final answer>",\n'
            '  "reasoning": "<comparison of agents, points of disagreement, and justification>",\n'
            '  "confidence": <float between 0.0 and 1.0>\n'
            "}"
        )

        messages = [
            {"role": "system", "content": RECONCILIATION_SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ]

        raw_content = ""
        try:
            if self._openai_client is not None:
                if HAS_PYDANTIC:
                    # Using official AsyncOpenAI SDK Structured Outputs (beta parse)
                    response = await self._openai_client.beta.chat.completions.parse(
                        model=self.model,
                        messages=messages,
                        temperature=0.2,
                        response_format=ReconciliationOutput,
                    )
                    parsed_data = response.choices[0].message.parsed
                    if parsed_data:
                        return {
                            "final_answer": parsed_data.final_answer,
                            "reasoning": parsed_data.reasoning,
                            "confidence": parsed_data.confidence,
                        }
                    raw_content = response.choices[0].message.content or "{}"
                else:
                    response = await self._openai_client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=0.2,
                        response_format={"type": "json_object"},
                    )
                    raw_content = response.choices[0].message.content or "{}"
            else:
                raw_content = await asyncio.to_thread(
                    self._http_request, messages
                )
        except Exception as e:
            fallback = self._mock_reconcile(question, worker_responses)
            fallback["reasoning"] += f" (Live API call error: {e}; used simulated fallback)"
            return fallback

        parsed = self._extract_json(raw_content)
        return {
            "final_answer": str(parsed.get("final_answer", "Undetermined")),
            "reasoning": str(parsed.get("reasoning", "No reconciliation reasoning provided.")),
            "confidence": float(parsed.get("confidence", 0.5)),
        }

    def _http_request(self, messages: list) -> str:
        """Fallback standard-library HTTP requester with strict JSON schema Structured Outputs."""
        import urllib.request
        base_url = OPENAI_BASE_URL or "https://api.openai.com/v1"
        endpoint = f"{base_url.rstrip('/')}/chat/completions"
        req = urllib.request.Request(
            endpoint,
            data=json.dumps({
                "model": self.model,
                "messages": messages,
                "temperature": 0.2,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "reconciliation_response",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "final_answer": {
                                    "type": "string",
                                    "description": "Your chosen final answer",
                                },
                                "reasoning": {
                                    "type": "string",
                                    "description": "Comparison of agents and justification",
                                },
                                "confidence": {
                                    "type": "number",
                                    "description": "Confidence score between 0.0 and 1.0",
                                },
                            },
                            "required": ["final_answer", "reasoning", "confidence"],
                            "additionalProperties": False,
                        },
                    },
                },
            }).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self.api_key}",
            },
        )
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read().decode("utf-8"))
            return data["choices"][0]["message"]["content"]

    def _extract_json(self, text: str) -> Dict[str, Any]:
        """Extracts and parses JSON object from model output."""
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            match = re.search(r"\{.*\}", text, re.DOTALL)
            if match:
                try:
                    return json.loads(match.group(0))
                except json.JSONDecodeError:
                    pass
        return {
            "final_answer": "Parsing Error",
            "reasoning": f"Could not parse valid JSON from reconciliation response: {text[:100]}...",
            "confidence": 0.0,
        }

    def _mock_reconcile(
        self, question: str, worker_responses: List[Dict[str, Any]]
    ) -> Dict[str, Any]:
        """
        Simulates qualitative arbitration for offline mode.
        Demonstrates reconciling consensus vs. adopting a minority insight.
        """
        q_lower = question.lower()

        # Scenario 1: Gold vs Feathers (Minority insight wins over majority!)
        if "gold" in q_lower and "feather" in q_lower:
            return {
                "final_answer": "A pound of feathers is heavier (approx. 453.6g vs 373.2g)",
                "reasoning": (
                    "DISAGREEMENT ANALYSIS:\n"
                    "- Agent 1 and Agent 3 concluded they weigh the same based on the colloquial "
                    "assumption that 'a pound' is invariant.\n"
                    "- Agent 2 (Critical Skeptic) identified a crucial domain technicality: gold and other "
                    "precious metals are weighed in Troy pounds (12 troy oz = 373.24 grams), whereas feathers "
                    "are weighed in standard Avoirdupois pounds (16 oz = 453.59 grams).\n\n"
                    "JUDGMENT:\n"
                    "While majority vote would be 2-to-1 for 'equal', Agent 2's reasoning exposes an objective "
                    "measurement reality that invalidates the naive premise. Therefore, the minority opinion "
                    "of Agent 2 is adopted as the definitive answer."
                ),
                "confidence": 0.96,
            }

        # Scenario 2: Bat and Ball ($1.10 total)
        if "bat" in q_lower and "ball" in q_lower:
            return {
                "final_answer": "$0.05 (5 cents)",
                "reasoning": (
                    "DISAGREEMENT ANALYSIS:\n"
                    "None. All three agents independently arrived at $0.05 with high confidence.\n\n"
                    "JUDGMENT:\n"
                    "Agent 1 provided strict algebraic formulation (2x + 1.00 = 1.10). Agent 2 preemptively "
                    "debunked the common cognitive trap of $0.10. Agent 3 verified total cost and difference. "
                    "With unanimous mathematical alignment across all perspectives, confidence is near absolute."
                ),
                "confidence": 0.99,
            }

        # Scenario 3: Monty Hall
        if "monty" in q_lower or "door" in q_lower:
            return {
                "final_answer": "Switch doors (2/3 probability of winning)",
                "reasoning": (
                    "DISAGREEMENT ANALYSIS:\n"
                    "Unanimous agreement across all 3 agents to switch doors.\n\n"
                    "JUDGMENT:\n"
                    "Agent 1 correctly highlighted the host's action as conditional filtering. Agent 2 addressed "
                    "the false 50/50 intuition. Agent 3 cited game-theoretic simulations. All arguments are sound."
                ),
                "confidence": 0.95,
            }

        # Scenario 4: UPSC Prelims (Preamble / Constitutional status)
        if "preamble" in q_lower or "upsc" in q_lower:
            return {
                "final_answer": "(d) a part of the Constitution but has no legal effect independently of other parts",
                "reasoning": (
                    "DISAGREEMENT ANALYSIS:\n"
                    "- Agent 1 concluded (a), arguing that because the Preamble is non-justiciable, it possesses no direct legal effect.\n"
                    "- Agent 2 (Critical Skeptic) and Agent 3 (Pragmatic Contextualist) both selected (d), correctly distinguishing between "
                    "'having no legal effect at all' and 'having no legal effect independently of other parts'.\n\n"
                    "JUDGMENT:\n"
                    "Agent 1 fell into the classic UPSC trap of conflating non-justiciability with complete legal nullity. As Agent 2 correctly "
                    "articulated citing Kesavananda Bharati (1973) and LIC of India (1995), the Preamble is an integral part of the Constitution "
                    "and carries profound legal and interpretive weight when read alongside other constitutional provisions (such as Fundamental "
                    "Rights and Directive Principles). Therefore, Option (d) is legally exact and adopted as the definitive answer."
                ),
                "confidence": 0.96,
            }

        # Default fallback for arbitrary questions
        answers = [r.get("answer", "") for r in worker_responses]
        return {
            "final_answer": answers[0] if answers else "Synthesized conclusion",
            "reasoning": (
                f"Evaluated responses from all {len(worker_responses)} agents. "
                "Synthesized perspectives across deductive logic, critical edge cases, and pragmatic context "
                "to form a coherent and balanced consensus."
            ),
            "confidence": 0.88,
        }
