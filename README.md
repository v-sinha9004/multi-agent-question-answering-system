# Simple Multi-Agent Question-Answering System

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://v-sinha9004-multi-agent-question-answering-system-app-fted6r.streamlit.app/)

> 🚀 **Live Demo:** Try the interactive application directly online at [v-sinha9004-multi-agent-question-answering-system-app-fted6r.streamlit.app](https://v-sinha9004-multi-agent-question-answering-system-app-fted6r.streamlit.app/) without any local installation.

A lightweight, readable Python prototype demonstrating the **Fan-Out / Fan-In (Ensemble & Arbitrator)** multi-agent architecture.

This project is built for **educational clarity**: zero heavy frameworks (no LangChain, AutoGen, or CrewAI), no complex infrastructure, and standard Python conventions so you can easily understand how multi-agent coordination actually works under the hood.

---

## Architecture & Data Flow

```text
                           User Question
                                 │
                                 ▼
                     ┌───────────────────────┐
                     │     Orchestrator      │
                     └───────────┬───────────┘
                                 │
               ┌─────────────────┼─────────────────┐
     (Fan-Out) │                 │                 │ (Fan-Out)
               ▼                 ▼                 ▼
        ┌─────────────┐   ┌─────────────┐   ┌─────────────┐
        │   Agent 1   │   │   Agent 2   │   │   Agent 3   │
        │ First-Princ.│   │  Skeptic /  │   │ Pragmatic / │
        │  Reasoner   │   │  Edge-Case  │   │ Contextual  │
        └──────┬──────┘   └──────┬──────┘   └──────┬──────┘
               │                 │                 │
               └─────────────────┼─────────────────┘
                                 │ (Fan-In: collect JSONs)
                                 ▼
                     ┌───────────────────────┐
                     │ Reconciliation Agent  │
                     │    (Master Judge)     │
                     └───────────┬───────────┘
                                 │
                                 ▼
                        Final Reconciled
                         Answer & Audit
```

### Key Principles

1. **Cognitive Isolation (Zero Groupthink):**
   Workers 1, 2, and 3 have no knowledge of each other and receive the prompt simultaneously. This prevents **anchoring bias**, where one agent's premature conclusion or hallucination skews the rest.
2. **Analytical Diversity:**
   To avoid carbon-copy answers when querying the same LLM, each worker is assigned a distinct cognitive persona:
   * **Agent 1 (First-Principles Reasoner):** Breaks down the problem using formal logic, axioms, and literal definitions.
   * **Agent 2 (Critical Skeptic):** Hunts for semantic traps, counter-intuitive boundary cases, and common biases.
   * **Agent 3 (Pragmatic Contextualist):** Evaluates real-world communicative intent and high-probability expectations.
3. **Qualitative Arbitration (Soundness Over Voting):**
   The Reconciliation Agent does **not** perform naive majority voting (`2 vs 1 -> majority wins`). Instead, it scrutinizes each agent's reasoning chain. If a single agent uncovers an overlooked domain technicality or trick that the other two missed, the judge can award the final verdict to the minority.

---

## File Structure

```text
.
├── config.py                 # Configuration, model settings, and worker personas
├── worker_agent.py           # Independent worker agents (isolated reasoning & JSON parsing)
├── reconciliation_agent.py   # Reconciliation judge (qualitative synthesis & arbitration)
├── orchestrator.py           # Concurrency coordinator with explicit 5-step lifecycle markers
├── app.py                    # Interactive Streamlit Web UI (dropdown, custom paste, mock/API toggle)
├── main.py                   # Terminal CLI (interactive menu and single-shot flags)
├── pyproject.toml            # Project configuration and dependencies (uv)
├── uv.lock                   # Exact reproducible lockfile
└── README.md                 # System overview and architecture guide

```

---

## The 5-Step Orchestration Lifecycle

In `orchestrator.py`, the flow is organized into 5 explicitly numbered stages:

1. **The Orchestrator Starts:** Receives the question and configures the execution pipeline.
2. **Independent Agents are Called (Fan-Out):** Dispatches the question concurrently to all 3 worker agents using `asyncio.gather`.
3. **Their Outputs are Collected (Fan-In):** Gathers structured JSON responses (`answer`, `reasoning`, `confidence`) once all parallel tasks complete.
4. **Reconciliation Happens:** Passes the original question and all worker outputs to the Reconciliation Agent to identify agreements, expose flaws, and decide the outcome.
5. **The Final Answer is Produced:** Bundles the complete audit trail and returns the unified result.

---

## How to Run

### 1. Interactive Web UI (Streamlit)

You can explore the system directly in your browser or run it locally:

- **🌐 Live Deployed App:** [Streamlit Community Cloud Demo](https://v-sinha9004-multi-agent-question-answering-system-app-fted6r.streamlit.app/)
- **💻 Run Locally:**
  ```bash
  uv run streamlit run app.py
  ```

Features available in the UI:
- **Choose mock questions from dropdown**: Instantly populate pre-configured benchmark dilemmas (Gold vs Feathers, Bat & Ball, Monty Hall, UPSC Prelims).
- **Custom Question Paste**: Type or paste any custom question into the text area.
- **Toggle Mock or Live API**: Switch easily between zero-cost offline simulation and live OpenAI models (`gpt-4o`, `gpt-4o-mini`, etc.).
- **Side-by-Side Agent Reasoning**: Inspect each agent's analytical lens, confidence score, and step-by-step reasoning chain in parallel columns.
- **Master Arbitrator Card**: View the definitive reconciled verdict and qualitative arbitration justification.
- **Developer Audit Trail**: Inspect raw JSON payloads for all agent interactions.

---

### 2. Terminal CLI: Offline Simulation (Mock Mode)

The prototype also includes a full terminal interface:

```bash
uv run main.py
```

Or pass a specific question:
```bash
uv run main.py --mock -q "Which is heavier: a pound of gold or a pound of feathers?"
```

### 3. Terminal CLI: Live OpenAI Mode (Using `gpt-4o`)

To connect to live OpenAI models:

1. Configure your `.env` file (the system automatically loads environment variables from `.env` via `python-dotenv`):
   ```bash
   cp .env.example .env
   # Edit .env and paste your API key:
   # OPENAI_API_KEY="sk-..."
   ```
   *Alternatively, export the key in your terminal session:*
   ```bash
   export OPENAI_API_KEY="sk-..."
   ```

2. Run with any model (`uv` automatically syncs dependencies):
   ```bash
   # Uses default gpt-4o (or the model configured in .env)
   uv run main.py

   # Or specify a different model via CLI flag:
   uv run main.py --model gpt-4o-mini -q "Solve: If 5 machines take 5 minutes to make 5 widgets, how long do 100 machines take to make 100 widgets?"
   ```
