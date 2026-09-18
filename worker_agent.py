"""
worker_agent.py

Defines the WorkerAgent class.
Each worker agent operates independently in complete isolation,
evaluates the question using its unique analytical persona,
and returns a structured JSON response containing:
  - answer
  - reasoning
  - confidence
"""

import asyncio
import json
import re
from typing import Any, Dict, Optional
from config import DEFAULT_MODEL, OPENAI_API_KEY

try:
    from pydantic import BaseModel, Field
    HAS_PYDANTIC = True
except ImportError:
    HAS_PYDANTIC = False

# Optional import of official OpenAI SDK
try:
    from openai import AsyncOpenAI
    HAS_OPENAI_SDK = True
except ImportError:
    HAS_OPENAI_SDK = False


if HAS_PYDANTIC:
    class WorkerOutput(BaseModel):
        """Strict Structured Output schema for worker agents."""
        answer: str = Field(description="Your concise answer")
        reasoning: str = Field(description="Your step-by-step reasoning")
        confidence: float = Field(description="Confidence score between 0.0 and 1.0")


class WorkerAgent:
    """
    An independent LLM agent with a specific analytical persona.
    Does NOT share state or communicate with any other worker agents.
    """

    def __init__(
        self,
        name: str,
        role: str,
        system_prompt: str,
        temperature: float = 0.5,
        model: str = DEFAULT_MODEL,
        use_mock: bool = False,
    ):
        self.name = name
        self.role = role
        self.system_prompt = system_prompt
        self.temperature = temperature
        self.model = model
        self.use_mock = use_mock

        # Initialize AsyncOpenAI client if available and not mocking
        self._openai_client = None
        if not self.use_mock and OPENAI_API_KEY and HAS_OPENAI_SDK:
            self._openai_client = AsyncOpenAI(api_key=OPENAI_API_KEY)

    async def solve(self, question: str) -> Dict[str, Any]:
        """
        Solves the question independently and returns structured JSON:
        {
            "agent": str,
            "role": str,
            "answer": str,
            "reasoning": str,
            "confidence": float
        }
        """
        if self.use_mock or not OPENAI_API_KEY:
            # Simulate processing delay to reflect realistic network latency
            await asyncio.sleep(0.3)
            return self._mock_solve(question)

        return await self._call_llm(question)

    async def _call_llm(self, question: str) -> Dict[str, Any]:
        """Calls the OpenAI API with JSON schema enforcement."""
        prompt = (
            f"Question:\n{question}\n\n"
            "Evaluate this question independently. Provide your reasoning, select an answer, "
            "and assign your confidence score between 0.0 and 1.0.\n\n"
            "Respond ONLY with a valid JSON object matching this schema:\n"
            "{\n"
            '  "answer": "<your concise answer>",\n'
            '  "reasoning": "<your step-by-step reasoning>",\n'
            '  "confidence": <float between 0.0 and 1.0>\n'
            "}"
        )

        messages = [
            {"role": "system", "content": self.system_prompt},
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
                        temperature=self.temperature,
                        response_format=WorkerOutput,
                    )
                    parsed_data = response.choices[0].message.parsed
                    if parsed_data:
                        return {
                            "agent": self.name,
                            "role": self.role,
                            "answer": parsed_data.answer,
                            "reasoning": parsed_data.reasoning,
                            "confidence": parsed_data.confidence,
                        }
                    raw_content = response.choices[0].message.content or "{}"
                else:
                    response = await self._openai_client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        response_format={"type": "json_object"},
                    )
                    raw_content = response.choices[0].message.content or "{}"
            else:
                # Direct HTTP call via standard library (as backup)
                raw_content = await asyncio.to_thread(
                    self._http_request, messages
                )
        except Exception as e:
            # Fallback to mock on connection/auth errors with informative message
            fallback = self._mock_solve(question)
            fallback["reasoning"] += f" (Live API call error: {e}; used simulated fallback)"
            return fallback

        parsed = self._extract_json(raw_content)
        return {
            "agent": self.name,
            "role": self.role,
            "answer": str(parsed.get("answer", "Unknown")),
            "reasoning": str(parsed.get("reasoning", "No reasoning provided.")),
            "confidence": float(parsed.get("confidence", 0.5)),
        }

    def _http_request(self, messages: list) -> str:
        """Fallback standard-library HTTP requester with strict JSON schema Structured Outputs."""
        import urllib.request
        req = urllib.request.Request(
            "https://api.openai.com/v1/chat/completions",
            data=json.dumps({
                "model": self.model,
                "messages": messages,
                "temperature": self.temperature,
                "response_format": {
                    "type": "json_schema",
                    "json_schema": {
                        "name": "worker_response",
                        "strict": True,
                        "schema": {
                            "type": "object",
                            "properties": {
                                "answer": {
                                    "type": "string",
                                    "description": "Your concise answer",
                                },
                                "reasoning": {
                                    "type": "string",
                                    "description": "Your step-by-step reasoning",
                                },
                                "confidence": {
                                    "type": "number",
                                    "description": "Confidence score between 0.0 and 1.0",
                                },
                            },
                            "required": ["answer", "reasoning", "confidence"],
                            "additionalProperties": False,
                        },
                    },
                },
            }).encode("utf-8"),
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {OPENAI_API_KEY}",
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
            "answer": "Parsing Error",
            "reasoning": f"Could not parse valid JSON from response: {text[:100]}...",
            "confidence": 0.0,
        }

    def _mock_solve(self, question: str) -> Dict[str, Any]:
        """
        Simulates intelligent, persona-specific responses for offline testing.
        Highlights how distinct analytical lenses arrive at differing conclusions.
        """
        q_lower = question.lower()

        # Scenario 1: Pound of gold vs pound of feathers (Troy vs Avoirdupois trap)
        if "gold" in q_lower and "feather" in q_lower:
            if "agent 1" in self.name.lower():
                return {
                    "agent": self.name,
                    "role": self.role,
                    "answer": "They weigh the same (1 pound = 1 pound)",
                    "reasoning": (
                        "From pure deductive definitions: a pound is a standardized unit of weight. "
                        "By formal identity, 1 lb = 1 lb regardless of the material."
                    ),
                    "confidence": 0.85,
                }
            elif "agent 2" in self.name.lower():
                return {
                    "agent": self.name,
                    "role": self.role,
                    "answer": "A pound of feathers is heavier",
                    "reasoning": (
                        "Caught the classic measurement trap! Precious metals like gold are measured in "
                        "Troy weight (1 Troy lb = 12 troy oz = ~373.24 g), while common commodities like feathers "
                        "are measured in Avoirdupois weight (1 Avoirdupois lb = 16 oz = ~453.59 g). 453.6g > 373.2g."
                    ),
                    "confidence": 0.95,
                }
            else:
                return {
                    "agent": self.name,
                    "role": self.role,
                    "answer": "They weigh the same",
                    "reasoning": (
                        "In standard contemporary English and conversational context, people use 'a pound' "
                        "as a generic invariant quantity of weight. Without explicit mention of Troy weight, "
                        "they are equal."
                    ),
                    "confidence": 0.80,
                }

        # Scenario 2: Bat and ball ($1.10 total, bat is $1.00 more)
        if "bat" in q_lower and "ball" in q_lower:
            if "agent 1" in self.name.lower():
                return {
                    "agent": self.name,
                    "role": self.role,
                    "answer": "$0.05 (5 cents)",
                    "reasoning": (
                        "Formulating linear system: Bat + Ball = 1.10; Bat = Ball + 1.00. "
                        "Substituting: 2*Ball + 1.00 = 1.10 => 2*Ball = 0.10 => Ball = 0.05."
                    ),
                    "confidence": 0.99,
                }
            elif "agent 2" in self.name.lower():
                return {
                    "agent": self.name,
                    "role": self.role,
                    "answer": "$0.05 (5 cents)",
                    "reasoning": (
                        "Alert to intuitive cognitive bias: fast thinking assumes $0.10, but if the ball were $0.10, "
                        "the bat would be $1.10, totaling $1.20. Therefore, the ball must be exactly $0.05."
                    ),
                    "confidence": 0.98,
                }
            else:
                return {
                    "agent": self.name,
                    "role": self.role,
                    "answer": "$0.05 (5 cents)",
                    "reasoning": (
                        "Verification check: $1.05 bat + $0.05 ball = $1.10 total. "
                        "Difference is $1.05 - $0.05 = $1.00. Both conditions hold."
                    ),
                    "confidence": 0.95,
                }

        # Scenario 3: Monty Hall problem (switch or stay)
        if "monty" in q_lower or "door" in q_lower:
            if "agent 1" in self.name.lower():
                return {
                    "agent": self.name,
                    "role": self.role,
                    "answer": "Switch doors (Win probability = 2/3)",
                    "reasoning": (
                        "Initial selection has P(Car) = 1/3, P(Goat) = 2/3. The host's reveal filters out "
                        "a losing door conditioned on your choice. Switching inverts the 2/3 probability of picking a goat."
                    ),
                    "confidence": 0.95,
                }
            elif "agent 2" in self.name.lower():
                return {
                    "agent": self.name,
                    "role": self.role,
                    "answer": "Switch doors (Win probability = 2/3)",
                    "reasoning": (
                        "Challenging the 50/50 intuition: People assume 2 remaining doors mean equal odds, "
                        "but the host is not picking at random; the host acts as a deterministic filter."
                    ),
                    "confidence": 0.92,
                }
            else:
                return {
                    "agent": self.name,
                    "role": self.role,
                    "answer": "Switch doors",
                    "reasoning": (
                        "Extensive empirical simulations and game theory demonstrate switching doubles chances "
                        "from 33.3% to 66.7%."
                    ),
                    "confidence": 0.90,
                }

        # Default fallback for arbitrary custom questions in offline mode
        role_perspectives = {
            "agent 1": (
                "Deconstructed the question into formal components and premises; "
                "evaluated validity and derived the most logically direct answer."
            ),
            "agent 2": (
                "Scrutinized the question for potential ambiguities, boundary cases, "
                "or deceptive phrasing before reaching this verdict."
            ),
            "agent 3": (
                "Interpreted the core intent through everyday context and practical consensus "
                "to yield a pragmatic conclusion."
            ),
        }
        agent_key = "agent 1" if "agent 1" in self.name.lower() else ("agent 2" if "agent 2" in self.name.lower() else "agent 3")

        return {
            "agent": self.name,
            "role": self.role,
            "answer": f"Proposed solution based on {self.role}",
            "reasoning": f"[{self.role}] {role_perspectives[agent_key]} Evaluated: '{question}'.",
            "confidence": 0.85 if agent_key == "agent 1" else (0.80 if agent_key == "agent 2" else 0.75),
        }
