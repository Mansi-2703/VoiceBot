"""
Voice Agent UI - Professional Streamlit Application
Minimalist design with clean typography and efficient layout.
"""

import streamlit as st
from src.pipeline import run_pipeline

# ============================================================================
# PAGE CONFIG
# ============================================================================
st.set_page_config(
    page_title="Voice Agent",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ============================================================================
# CSS STYLING
# ============================================================================
st.markdown("""
    <style>
    /* Remove default padding */
    section[data-testid="stAppViewContainer"] {
        padding-top: 2rem;
    }
    
    /* Main container */
    .main {
        max-width: 1400px;
        margin: 0 auto;
    }
    
    /* Card containers */
    .card {
        background: #f8f8f7;
        border: 1px solid #e2e0da;
        border-radius: 8px;
        padding: 1.25rem 1.5rem;
        margin-bottom: 1rem;
        display: block;
        width: 100%;
    }
    
    /* Output label style */
    .output-label {
        font-size: 11px;
        font-weight: 500;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        color: #888780;
        margin-bottom: 6px;
        display: block;
        margin-top: 0;
    }
    
    /* Output value style */
    .output-value {
        font-size: 15px;
        color: #1a1a18;
        line-height: 1.6;
        word-break: break-word;
    }
    
    /* Intent badge */
    .intent-badge {
        display: inline-block;
        background: #f1efea;
        border: 1px solid #d3d1c7;
        border-radius: 4px;
        padding: 2px 10px;
        font-size: 13px;
        font-family: monospace;
        color: #3d3d3a;
    }
    
    /* Run button */
    .run-button {
        width: 100%;
        background: #1a1a18;
        color: #ffffff;
        border: none;
        border-radius: 6px;
        padding: 0.65rem 1rem;
        font-size: 14px;
        font-weight: 500;
        cursor: pointer;
        transition: background 0.2s ease;
    }
    
    .run-button:hover {
        background: #2d2d2a;
    }
    
    /* Divider */
    .divider {
        height: 1px;
        background: #e2e0da;
        margin: 1rem 0;
    }
    
    /* Muted text */
    .muted {
        color: #888780;
        font-size: 13px;
        line-height: 1.5;
    }
    
    /* Heading */
    h3 {
        font-weight: 500;
        margin: 0 0 1rem 0;
        font-size: 18px;
        color: #1a1a18;
    }
    
    /* Heading 2 */
    h2 {
        font-weight: 400;
        margin: 0 0 1.5rem 0;
        font-size: 24px;
        color: #1a1a18;
    }
    
    /* Waiting text */
    .waiting {
        color: #888780;
        font-style: italic;
    }
    
    /* Error text */
    .error-text {
        color: #d32f2f;
    }
    
    /* Ensure text inside cards is visible */
    .card p, .card div {
        color: #1a1a18;
        font-size: 15px;
        margin: 0;
        padding: 0;
    }
    
    /* Session history item */
    .history-item {
        padding: 0.5rem 0;
        font-size: 13px;
        color: #1a1a18;
        border-bottom: 1px solid #e2e0da;
    }
    
    .history-item:last-child {
        border-bottom: none;
    }
    
    /* Override primary button color */
    button[kind="primary"] {
        background-color: #1a1a18 !important;
        color: #ffffff !important;
    }
    
    button[kind="primary"]:hover {
        background-color: #2d2d2a !important;
    }
    
    button[kind="primary"]:active {
        background-color: #1a1a18 !important;
    }
    </style>
    """, unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if "transcript" not in st.session_state:
    st.session_state.transcript = None
if "intent" not in st.session_state:
    st.session_state.intent = None
if "intents" not in st.session_state:
    st.session_state.intents = []
if "confidence" not in st.session_state:
    st.session_state.confidence = None
if "action_taken" not in st.session_state:
    st.session_state.action_taken = None
if "actions_taken" not in st.session_state:
    st.session_state.actions_taken = []
if "result" not in st.session_state:
    st.session_state.result = None
if "error" not in st.session_state:
    st.session_state.error = None
if "run_history" not in st.session_state:
    st.session_state.run_history = []

# ============================================================================
# HEADER
# ============================================================================
col1, col2, col3 = st.columns(3)
with col2:
    st.markdown("<h1 style='text-align: center; margin: 0;'>Voice Agent</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; margin: 0; color: #888780; font-size: 14px;'>Local AI pipeline</p>", unsafe_allow_html=True)
st.markdown('<div class="divider"></div>', unsafe_allow_html=True)

# ============================================================================
# MAIN LAYOUT
# ============================================================================
col_input, col_output = st.columns([1, 2], gap="large")

# ============================================================================
# LEFT COLUMN - INPUT
# ============================================================================
with col_input:
    # Audio input
    audio_input = st.audio_input("Record audio", label_visibility="collapsed")
    
    st.write("")  # Spacing
    
    # File uploader
    file_input = st.file_uploader(
        "Or upload a file",
        type=["wav", "mp3"],
        label_visibility="collapsed"
    )
    
    # Note
    st.markdown(
        '<p class="muted">Only one input is used. File upload takes priority if both are provided.</p>',
        unsafe_allow_html=True
    )
    
    st.write("")  # Spacing
    
    # Run button
    run_button = st.button(
        "Run agent",
        key="run_agent_btn",
        use_container_width=True,
        type="primary"
    )

# ============================================================================
# RIGHT COLUMN - OUTPUT
# ============================================================================
with col_output:
    # ========== Card 1: Transcription ==========
    transcript_value = st.session_state.transcript if st.session_state.transcript else "Waiting for input..."
    st.markdown(f"""
    <div style="
        background: #1e1e1e;
        border: 1px solid #2e2e2e;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
    ">
        <div style="
            font-size: 11px;
            font-weight: 500;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #888780;
            margin-bottom: 8px;
        ">Transcription</div>
        <div style="
            font-size: 14px;
            color: #e8e6e0;
            line-height: 1.6;
        ">{transcript_value}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # ========== Card 2: Detected Intent ==========
    intents_display = []
    if st.session_state.intent:
        intents_display = st.session_state.get("intents", [st.session_state.intent])
    
    if intents_display:
        # Display all intents (for compound commands)
        intent_badges = []
        for intent in intents_display:
            badge = f'<span style="display: inline-block; background: #2a2a2a; border: 1px solid #3a3a3a; border-radius: 4px; padding: 2px 10px; font-size: 12px; font-family: monospace; color: #c8c6c0; margin-right: 6px;">{intent}</span>'
            intent_badges.append(badge)
        
        intent_html = "".join(intent_badges)
        if st.session_state.confidence is not None:
            confidence_text = f'<div style="font-size: 12px; color: #888780; margin-top: 6px;">Confidence: {st.session_state.confidence:.0%}</div>'
        else:
            confidence_text = ""
        intent_content = intent_html + confidence_text
    else:
        intent_content = '<div style="font-size: 14px; color: #e8e6e0;">—</div>'
    
    st.markdown(f"""
    <div style="
        background: #1e1e1e;
        border: 1px solid #2e2e2e;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
    ">
        <div style="
            font-size: 11px;
            font-weight: 500;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #888780;
            margin-bottom: 8px;
        ">Detected Intent</div>
        {intent_content}
    </div>
    """, unsafe_allow_html=True)
    
    # ========== Card 3: Action Taken ==========
    actions_display = st.session_state.get("actions_taken", [st.session_state.action_taken]) if st.session_state.action_taken else []
    action_value = ", ".join(actions_display) if actions_display else "—"
    st.markdown(f"""
    <div style="
        background: #1e1e1e;
        border: 1px solid #2e2e2e;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
    ">
        <div style="
            font-size: 11px;
            font-weight: 500;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #888780;
            margin-bottom: 8px;
        ">Action Taken</div>
        <div style="
            font-size: 14px;
            color: #e8e6e0;
            line-height: 1.6;
        ">{action_value}</div>
    </div>
    """, unsafe_allow_html=True)
    
    # ========== Card 4: Result ==========
    st.markdown("""
    <div style="
        background: #1e1e1e;
        border: 1px solid #2e2e2e;
        border-radius: 8px;
        padding: 16px 20px;
        margin-bottom: 12px;
    ">
        <div style="
            font-size: 11px;
            font-weight: 500;
            letter-spacing: 0.08em;
            text-transform: uppercase;
            color: #888780;
            margin-bottom: 8px;
        ">Result</div>
    </div>
    """, unsafe_allow_html=True)
    
    if st.session_state.error:
        st.error(st.session_state.error)
    elif st.session_state.result:
        # Check if result is code
        if st.session_state.action_taken == "write_code":
            st.code(st.session_state.result, language="python")
        else:
            st.write(st.session_state.result)
    else:
        st.markdown('<div style="font-size: 14px; color: #e8e6e0;">—</div>', unsafe_allow_html=True)

# ============================================================================
# PROCESS LOGIC
# ============================================================================
if run_button:
    # Validate input
    if file_input is None and audio_input is None:
        st.warning("Please provide an audio input before running.")
    else:
        # Prepare input (file takes priority)
        if file_input is not None:
            pipeline_input = file_input.read()
        else:
            pipeline_input = audio_input.getvalue() if hasattr(audio_input, 'getvalue') else audio_input
        
        # Run pipeline with spinner
        with st.spinner("Processing..."):
            pipeline_result = run_pipeline(pipeline_input)
        
        # Update session state
        st.session_state.transcript = pipeline_result.get("transcript", "")
        st.session_state.intent = pipeline_result.get("intent", "")
        st.session_state.intents = pipeline_result.get("intents", [])
        st.session_state.confidence = pipeline_result.get("confidence", 0)
        st.session_state.action_taken = pipeline_result.get("action_taken", "")
        st.session_state.actions_taken = pipeline_result.get("actions_taken", [])
        st.session_state.error = pipeline_result.get("error", "")
        
        # Handle result - extract relevant output
        result_dict = pipeline_result.get("result", {})
        if result_dict:
            if st.session_state.action_taken == "write_code":
                st.session_state.result = result_dict.get("code_preview", "")
            elif st.session_state.action_taken == "create_file":
                st.session_state.result = f"File created at: {result_dict.get('path', 'N/A')}"
            elif st.session_state.action_taken == "summarize_text":
                st.session_state.result = result_dict.get("summary", "")
            else:  # general_chat
                st.session_state.result = result_dict.get("response", "")
        else:
            st.session_state.result = None
        
        # Check for audio clarity issues
        if not st.session_state.transcript or st.session_state.transcript == "":
            st.warning("Audio was unclear or silent. Please try again.")
        
        # Add to history
        if st.session_state.intent and st.session_state.transcript:
            transcript_preview = st.session_state.transcript[:40]
            if len(st.session_state.transcript) > 40:
                transcript_preview += "..."
            history_entry = f"[{st.session_state.intent}] — {transcript_preview}"
            st.session_state.run_history.insert(0, history_entry)
            # Keep only last 5
            st.session_state.run_history = st.session_state.run_history[:5]
        
        # Rerun to update UI
        st.rerun()

# ============================================================================
# SIDEBAR - SESSION HISTORY
# ============================================================================
with st.sidebar:
    st.markdown("### Session History")
    
    if st.session_state.run_history:
        for i, entry in enumerate(st.session_state.run_history, 1):
            st.markdown(f'<div class="history-item">{i}. {entry}</div>', unsafe_allow_html=True)
    else:
        st.markdown('<p class="muted">No runs yet.</p>', unsafe_allow_html=True)
