"""
app.py

Interactive Streamlit Web Interface for the Multi-Agent Question-Answering System.
Features:
  - Select pre-configured benchmark questions from a dropdown or type/paste custom questions
  - Toggle between Offline Simulation (Mock Mode) and Live API (OpenAI)
  - Configure models and API keys dynamically
  - Visual side-by-side display of individual agent answers, confidence scores, and reasoning
  - Prominent Reconciliation Agent card with arbitration breakdown
  - Collapsible developer audit trail with raw JSON output
"""

import asyncio
import os
import time
from typing import Any, Dict

import streamlit as st

from config import (
    DEFAULT_MODEL,
    DEFAULT_USE_MOCK,
    OPENAI_API_KEY,
    SAMPLE_QUESTIONS,
    WORKER_PERSONAS,
)
from orchestrator import Orchestrator

# -----------------------------------------------------------------------------
# Streamlit Page Configuration
# -----------------------------------------------------------------------------
st.set_page_config(
    page_title="Multi-Agent Q&A System",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------------------------------------------------------
# Custom Styling (CSS)
# -----------------------------------------------------------------------------
st.markdown(
    """
    <style>
    /* Global styling enhancements */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2.5rem;
    }
    
    /* Header hero banner */
    .hero-banner {
        background: linear-gradient(135deg, #1e293b 0%, #0f172a 100%);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 1.5rem 1.8rem;
        margin-bottom: 1.5rem;
        color: #f8fafc;
    }
    .hero-banner h1 {
        color: #f8fafc !important;
        font-size: 1.8rem !important;
        font-weight: 700 !important;
        margin-bottom: 0.3rem !important;
    }
    .hero-banner p {
        color: #94a3b8 !important;
        font-size: 0.95rem !important;
        margin: 0 !important;
    }

    /* Agent Cards */
    .agent-card {
        border-radius: 10px;
        padding: 1rem 1.2rem;
        margin-bottom: 1rem;
        height: 100%;
        background-color: var(--secondary-background-color);
        border: 1px solid rgba(128, 128, 128, 0.2);
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05);
    }
    .agent-1-card {
        border-top: 4px solid #3b82f6 !important;
    }
    .agent-2-card {
        border-top: 4px solid #f59e0b !important;
    }
    .agent-3-card {
        border-top: 4px solid #10b981 !important;
    }
    .reconciliation-card {
        background: linear-gradient(145deg, rgba(124, 58, 237, 0.08) 0%, rgba(79, 70, 229, 0.05) 100%);
        border: 1px solid rgba(124, 58, 237, 0.3);
        border-left: 5px solid #8b5cf6;
        border-radius: 10px;
        padding: 1.3rem 1.5rem;
        margin-bottom: 1.5rem;
    }

    /* Badges */
    .badge {
        display: inline-block;
        padding: 0.2rem 0.6rem;
        border-radius: 9999px;
        font-size: 0.75rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
    }
    .badge-blue { background-color: rgba(59, 130, 246, 0.15); color: #3b82f6; border: 1px solid rgba(59, 130, 246, 0.3); }
    .badge-amber { background-color: rgba(245, 158, 11, 0.15); color: #f59e0b; border: 1px solid rgba(245, 158, 11, 0.3); }
    .badge-emerald { background-color: rgba(16, 185, 129, 0.15); color: #10b981; border: 1px solid rgba(16, 185, 129, 0.3); }
    .badge-purple { background-color: rgba(139, 92, 246, 0.2); color: #8b5cf6; border: 1px solid rgba(139, 92, 246, 0.4); }

    .answer-highlight {
        font-size: 1.05rem;
        font-weight: 600;
        padding: 0.6rem 0.8rem;
        border-radius: 6px;
        margin: 0.6rem 0;
        background-color: rgba(128, 128, 128, 0.08);
        border-left: 3px solid #64748b;
    }

    .reconciliation-answer {
        font-size: 1.25rem;
        font-weight: 700;
        padding: 0.8rem 1rem;
        border-radius: 8px;
        margin: 0.7rem 0;
        background-color: rgba(139, 92, 246, 0.12);
        color: #c084fc;
        border: 1px solid rgba(139, 92, 246, 0.3);
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Sidebar: System Configuration & Controls
# -----------------------------------------------------------------------------
with st.sidebar:
    st.header("⚙️ Configuration")
    
    # Mode selection
    mode_option = st.radio(
        "Execution Engine",
        ["Offline Simulation (Mock Mode)", "Live API (OpenAI)"],
        index=0 if DEFAULT_USE_MOCK else 1,
        help=(
            "Mock Mode demonstrates deterministic multi-agent behavior with zero API keys or network latency. "
            "Live API connects directly to OpenAI models."
        ),
    )
    use_mock = mode_option.startswith("Offline")

    # API Configuration (when Live API is chosen)
    custom_api_key = None
    selected_model = DEFAULT_MODEL
    
    if not use_mock:
        st.subheader("OpenAI Settings")
        selected_model = st.selectbox(
            "Model",
            ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo", "gpt-3.5-turbo"],
            index=0,
            help="Select the OpenAI model to power both workers and the reconciliation judge.",
        )
        
        env_has_key = bool(OPENAI_API_KEY)
        if env_has_key:
            st.success("API Key detected from `.env`", icon="✅")
            override_key = st.text_input(
                "Override API Key (Optional)",
                type="password",
                placeholder="sk-...",
                help="Leave blank to use the API key found in your .env file.",
            )
            if override_key.strip():
                custom_api_key = override_key.strip()
        else:
            st.warning("No API key detected in `.env`.", icon="⚠️")
            custom_api_key = st.text_input(
                "Enter OpenAI API Key",
                type="password",
                placeholder="sk-...",
                help="Enter your key to query live models.",
            ).strip()
            if not custom_api_key:
                st.info("Without an API key, queries will fall back to simulated mock mode.")
    else:
        st.info("💡 **Mock Mode active**: No API key or internet required. Perfect for testing arbitration logic.", icon="ℹ️")

    st.markdown("---")
    
    # Analytical Personas Reference
    with st.expander("👥 Worker Personas & Temperature", expanded=False):
        for persona in WORKER_PERSONAS:
            st.markdown(f"**{persona['name']}**: *{persona['role']}*")
            st.caption(f"Temperature: `{persona['temperature']}` | Prompt: {persona['system_prompt']}")
            st.markdown("---")

    # Architecture Reference
    with st.expander("🏛️ Multi-Agent Architecture", expanded=False):
        st.markdown(
            """
            1. **Orchestrator Starts**: Receives question and initializes isolated tasks.
            2. **Parallel Fan-Out**: Workers operate simultaneously with zero inter-agent communication (strict cognitive isolation).
            3. **Fan-In Collection**: Raw structured answers and reasoning are aggregated.
            4. **Qualitative Reconciliation**: The master judge examines reasoning, uncovers hidden traps, and determines the verdict.
            5. **Final Answer**: Synthesized decision and audit trail are returned.
            """
        )

# -----------------------------------------------------------------------------
# Main Header
# -----------------------------------------------------------------------------
st.markdown(
    """
    <div class="hero-banner">
        <h1>🤖 Multi-Agent Question-Answering System</h1>
        <p>Parallel cognitive diversity + qualitative reconciliation for high-stakes reasoning tasks.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# -----------------------------------------------------------------------------
# Question Selection & Input Section
# -----------------------------------------------------------------------------
st.subheader("1. Select or Enter a Question")

# Initialize session state for the question box
if "current_question" not in st.session_state:
    st.session_state.current_question = ""

# Dropdown list: Custom + Sample Benchmarks
dropdown_options = ["✍️ Custom Question (Type or paste your own)"] + [
    f"📌 {q['title']}" for q in SAMPLE_QUESTIONS
]

def on_dropdown_change():
    selected = st.session_state.dropdown_selection
    if selected.startswith("📌"):
        # Match sample question
        title = selected[2:].strip()
        for q in SAMPLE_QUESTIONS:
            if q["title"] == title:
                st.session_state.current_question = q["question"]
                break
    elif selected.startswith("✍️"):
        st.session_state.current_question = ""

selected_option = st.selectbox(
    "Choose from pre-configured benchmark questions or paste a custom one:",
    dropdown_options,
    index=0,
    key="dropdown_selection",
    on_change=on_dropdown_change,
)

# Display scenario note if a sample question is chosen
active_sample = None
if selected_option.startswith("📌"):
    selected_title = selected_option[2:].strip()
    for q in SAMPLE_QUESTIONS:
        if q["title"] == selected_title:
            active_sample = q
            break

if active_sample:
    st.info(f"**Demo Insight**: {active_sample['note']}", icon="💡")

# Editable question input area directly bound to current_question state
question_input = st.text_area(
    "Question:",
    key="current_question",
    height=120,
    placeholder="Paste or write your question here...",
    help="You can freely edit this question or paste your own custom question.",
)

col_btn, col_info = st.columns([1, 4])
with col_btn:
    run_clicked = st.button("🚀 Run Analysis", type="primary", use_container_width=True)

with col_info:
    if use_mock:
        st.caption("🟢 Running in **Offline Mock Mode** (instant response, zero cost).")
    else:
        st.caption(f"🔵 Running in **Live OpenAI Mode** using `{selected_model}`.")

# -----------------------------------------------------------------------------
# Execution & Result Pipeline
# -----------------------------------------------------------------------------
if run_clicked:
    trimmed_q = question_input.strip()
    if not trimmed_q:
        st.error("Please enter a non-empty question to analyze.", icon="⚠️")
    else:
        # Determine effective API key
        effective_key = custom_api_key or OPENAI_API_KEY
        if not use_mock and not effective_key:
            st.error(
                "No OpenAI API key detected. Please provide your OpenAI API key in the sidebar, "
                "or switch the Execution Engine to Offline Mock Mode.",
                icon="🔑",
            )
            st.stop()
            
        actual_mock = use_mock
        
        # Display progress spinner
        with st.spinner("Dispatching question in parallel to Worker Agents and reconciling..."):
            orchestrator = Orchestrator(
                model=selected_model,
                use_mock=actual_mock,
                api_key=effective_key,
            )
            
            start_time = time.perf_counter()
            # Run async orchestrator pipeline
            result = asyncio.run(orchestrator.run(trimmed_q))
            latency = time.perf_counter() - start_time
            
            st.session_state.last_result = result
            st.session_state.last_latency = latency
            st.session_state.last_mode = "Mock Simulation" if actual_mock else f"OpenAI ({selected_model})"

# -----------------------------------------------------------------------------
# Results Presentation
# -----------------------------------------------------------------------------
if "last_result" in st.session_state:
    result = st.session_state.last_result
    latency = st.session_state.get("last_latency", 0.0)
    mode_label = st.session_state.get("last_mode", "Mock Simulation")
    
    worker_responses = result.get("worker_responses", [])
    reconciliation = result.get("reconciliation", {})

    st.markdown("---")
    st.subheader("2. Multi-Agent Analysis Results")
    
    # Telemetry metrics
    m1, m2, m3 = st.columns(3)
    m1.metric("Execution Mode", mode_label)
    m2.metric("Total Latency", f"{latency:.2f}s")
    m3.metric("Reconciliation Confidence", f"{reconciliation.get('confidence', 0.0) * 100:.1f}%")

    # -------------------------------------------------------------------------
    # Reconciliation Agent (Master Arbitrator Verdict)
    # -------------------------------------------------------------------------
    st.markdown("### 🏆 Master Arbitrator (Reconciliation Agent)")
    
    reconcile_conf = reconciliation.get("confidence", 0.0)
    conf_pct = int(reconcile_conf * 100)
    
    st.markdown(
        f"""
        <div class="reconciliation-card">
            <span class="badge badge-purple">Definitive Judgment</span>
            <div class="reconciliation-answer">
                {reconciliation.get('final_answer', 'Undetermined')}
            </div>
            <p style="margin-bottom: 0.3rem; font-weight: 600;">Arbitration Confidence: {conf_pct}%</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.progress(reconcile_conf)
    
    with st.expander("⚖️ View Detailed Arbitration Reasoning & Disagreement Analysis", expanded=True):
        st.markdown(reconciliation.get("reasoning", "No reconciliation reasoning provided."))

    # -------------------------------------------------------------------------
    # Independent Worker Agents (Parallel 3-Column Layout)
    # -------------------------------------------------------------------------
    st.markdown("### 👥 Independent Worker Agents")
    st.caption("Each agent evaluated the question in complete cognitive isolation with a dedicated analytical lens.")

    badge_classes = ["badge-blue", "badge-amber", "badge-emerald"]
    card_classes = ["agent-1-card", "agent-2-card", "agent-3-card"]
    
    cols = st.columns(len(worker_responses))
    
    for idx, resp in enumerate(worker_responses):
        with cols[idx]:
            card_class = card_classes[idx % len(card_classes)]
            badge_class = badge_classes[idx % len(badge_classes)]
            agent_name = resp.get("agent", f"Agent {idx + 1}")
            role = resp.get("role", "Worker")
            answer = resp.get("answer", "N/A")
            confidence = resp.get("confidence", 0.0)
            reasoning = resp.get("reasoning", "N/A")
            
            st.markdown(
                f"""
                <div class="agent-card {card_class}">
                    <span class="badge {badge_class}">{role}</span>
                    <h4 style="margin: 0.5rem 0 0.2rem 0;">{agent_name}</h4>
                    <p style="font-size: 0.85rem; color: #64748b; margin-bottom: 0.5rem;">Confidence: {confidence * 100:.0f}%</p>
                    <div class="answer-highlight">
                        <strong>Answer:</strong> {answer}
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )
            st.progress(confidence)
            
            with st.expander(f"🔍 {agent_name} Reasoning", expanded=True):
                st.markdown(f"**Step-by-Step Rationale:**\n\n{reasoning}")

    # -------------------------------------------------------------------------
    # Raw JSON Audit Trail
    # -------------------------------------------------------------------------
    st.markdown("---")
    with st.expander("🛠️ Developer Audit Trail (Raw JSON Payload)", expanded=False):
        st.json(result)
