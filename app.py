"""
app.py
Streamlit UI for the Lab Manual Conversational Assistant (Track A, Week 7-8: Polish & Production).

Flow:
1. Student uploads a lab manual PDF.
2. App validates and extracts text, detects individual experiments.
3. App builds a FAISS index over chunks for retrieval.
4. Student uses dedicated tabs: procedure, equipment, safety, troubleshooting,
   open-ended Q&A, pre-lab, post-lab, report generation, revision notes.
5. All AI calls are wrapped with friendly error handling instead of raw tracebacks.
"""

import os
import streamlit as st
from dotenv import load_dotenv

from utils.pdf_processor import extract_text_from_pdf, split_into_experiments, chunk_text
from utils.vector_store import ManualVectorStore
from utils.groq_client import ask_lab_assistant
from utils.feature_extractor import (
    get_procedure_steps,
    get_equipment_list,
    get_safety_precautions,
    get_troubleshooting_help,
    get_prelab_prep,
    get_postlab_analysis,
    generate_lab_report,
    generate_revision_notes,
)

load_dotenv()

st.set_page_config(page_title="Lab Manual Assistant", page_icon="🧪", layout="wide")

MAX_FILE_SIZE_MB = 50

# ---------- Session state ----------
if "experiments" not in st.session_state:
    st.session_state.experiments = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "manual_processed" not in st.session_state:
    st.session_state.manual_processed = False
if "manual_name" not in st.session_state:
    st.session_state.manual_name = ""


def safe_ai_call(func, *args, spinner_text="Thinking...", **kwargs):
    """
    Run an AI-calling function with a spinner, catching errors so the user
    sees a friendly message instead of a raw traceback.
    Returns the result, or None if an error occurred (already shown to the user).
    """
    try:
        with st.spinner(spinner_text):
            return func(*args, **kwargs)
    except ValueError as e:
        # Raised by groq_client.get_client() when the API key is missing
        st.error(f"⚠️ Setup issue: {e}")
    except Exception as e:
        msg = str(e)
        if "401" in msg or "invalid_api_key" in msg.lower() or "authentication" in msg.lower():
            st.error("⚠️ Your Groq API key appears to be invalid. Please check it in the sidebar and try again.")
        elif "429" in msg or "rate_limit" in msg.lower():
            st.error("⚠️ Rate limit reached on the Groq API. Please wait a few seconds and try again.")
        elif "timeout" in msg.lower() or "connection" in msg.lower():
            st.error("⚠️ Couldn't reach the Groq API — check your internet connection and try again.")
        else:
            st.error(f"⚠️ Something went wrong while generating this response: {msg}")
    return None


# ---------- Sidebar: API key + upload ----------
with st.sidebar:
    st.header("⚙️ Setup")

    env_key = os.environ.get("GROQ_API_KEY", "")
    api_key_input = st.text_input(
        "Groq API Key",
        value=env_key,
        type="password",
        help="Get a free key at console.groq.com. Not stored anywhere except this session.",
    )
    if api_key_input:
        os.environ["GROQ_API_KEY"] = api_key_input

    st.divider()
    st.header("📄 Upload Lab Manual")
    uploaded_file = st.file_uploader(
        "Upload a PDF lab manual",
        type=["pdf"],
        help=f"Maximum file size: {MAX_FILE_SIZE_MB} MB",
    )

    process_clicked = st.button("Process Manual", type="primary", use_container_width=True)

    if process_clicked:
        if not uploaded_file:
            st.warning("Please choose a PDF file first.")
        elif not api_key_input:
            st.error("Please enter your Groq API key above before processing a manual.")
        elif uploaded_file.size == 0:
            st.error("That file appears to be empty. Please upload a valid PDF.")
        elif uploaded_file.size > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.error(f"File is too large ({uploaded_file.size / 1024 / 1024:.1f} MB). "
                      f"Please upload a PDF under {MAX_FILE_SIZE_MB} MB.")
        else:
            try:
                with st.spinner("Extracting text and identifying experiments..."):
                    full_text = extract_text_from_pdf(uploaded_file)

                    if not full_text or not full_text.strip():
                        st.error(
                            "No readable text was found in this PDF. It may be a scanned "
                            "image-only document — this app currently requires a text-based PDF."
                        )
                        st.stop()

                    experiments = split_into_experiments(full_text)

                    all_chunks = []
                    for exp in experiments:
                        all_chunks.extend(chunk_text(exp["content"]))

                    if not all_chunks:
                        st.error("Couldn't extract any usable content from this PDF. Please try a different file.")
                        st.stop()

                    store = ManualVectorStore()
                    store.build(all_chunks)

                    st.session_state.experiments = experiments
                    st.session_state.vector_store = store
                    st.session_state.manual_processed = True
                    st.session_state.manual_name = uploaded_file.name

                st.success(f"Found {len(experiments)} experiment(s)!")
            except Exception as e:
                st.error(
                    "⚠️ Couldn't process this PDF. It may be corrupted, password-protected, "
                    f"or in an unsupported format.\n\nDetails: {e}"
                )

    if st.session_state.manual_processed:
        st.caption(f"✅ Currently loaded: **{st.session_state.manual_name}**")
        if st.button("🗑️ Clear and start over", use_container_width=True):
            st.session_state.experiments = []
            st.session_state.vector_store = None
            st.session_state.manual_processed = False
            st.session_state.manual_name = ""
            st.rerun()

# ---------- Main area ----------
st.title("🧪 Lab Manual Conversational Assistant")
st.caption("Upload your lab manual and ask questions about procedures, theory, or safety.")

if not st.session_state.manual_processed:
    st.info("👈 Upload a lab manual PDF and click **Process Manual** to get started.")
else:
    experiments = st.session_state.experiments

    st.subheader("📋 Detected Experiments")
    exp_labels = [f"Exp {e['number']}: {e['title'][:60]}" for e in experiments]
    selected_idx = st.selectbox(
        "Select an experiment (optional — narrows the assistant's focus)",
        options=range(len(exp_labels)),
        format_func=lambda i: exp_labels[i],
    )

    with st.expander("View raw extracted content for this experiment"):
        st.text(experiments[selected_idx]["content"][:3000])

    selected_content = experiments[selected_idx]["content"]

    st.divider()
    st.subheader("🔬 Experiment Assistant")

    tab_procedure, tab_equipment, tab_safety, tab_troubleshoot, tab_ask = st.tabs(
        ["📝 Procedure", "🧰 Equipment", "⚠️ Safety", "🛠️ Troubleshooting", "💬 Ask Anything"]
    )

    if not os.environ.get("GROQ_API_KEY"):
        st.warning("Enter your Groq API key in the sidebar to use these features.")
    else:
        with tab_procedure:
            if st.button("Generate step-by-step procedure", key="btn_procedure"):
                result = safe_ai_call(get_procedure_steps, selected_content,
                                       spinner_text="Breaking down the procedure...")
                if result:
                    st.markdown(result)

        with tab_equipment:
            if st.button("Identify equipment & materials", key="btn_equipment"):
                result = safe_ai_call(get_equipment_list, selected_content,
                                       spinner_text="Scanning for equipment and materials...")
                if result:
                    st.markdown(result)

        with tab_safety:
            if st.button("Extract safety precautions", key="btn_safety"):
                result = safe_ai_call(get_safety_precautions, selected_content,
                                       spinner_text="Checking for safety information...")
                if result:
                    st.markdown(result)

        with tab_troubleshoot:
            issue = st.text_area(
                "Describe the problem you're facing",
                placeholder="e.g. My output doesn't match the expected result / equipment isn't responding...",
                key="troubleshoot_input",
            )
            if st.button("Get troubleshooting help", key="btn_troubleshoot"):
                if not issue or not issue.strip():
                    st.warning("Please describe the problem first.")
                else:
                    result = safe_ai_call(get_troubleshooting_help, selected_content, issue,
                                           spinner_text="Thinking through possible causes...")
                    if result:
                        st.markdown(result)

        with tab_ask:
            if "question_input" not in st.session_state:
                st.session_state.question_input = ""

            example_qs = [
                "What is the procedure for this experiment?",
                "Explain the theory behind this experiment in simple terms.",
                "What safety precautions should I take?",
                "What equipment do I need?",
            ]
            cols = st.columns(len(example_qs))
            for c, q in zip(cols, example_qs):
                if c.button(q, use_container_width=True, key=f"ex_{q[:10]}"):
                    st.session_state.question_input = q

            question = st.text_input(
                "Your question",
                key="question_input",
                placeholder="e.g. What is the procedure of Exp 3 in Physics?",
            )

            if st.button("Ask", type="primary", key="btn_ask"):
                if not question or not question.strip():
                    st.warning("Please type a question first.")
                else:
                    store = st.session_state.vector_store
                    full_query = f"{exp_labels[selected_idx]}. {question}"
                    context_chunks = store.search(full_query, top_k=4)
                    answer = safe_ai_call(ask_lab_assistant, question, context_chunks,
                                           spinner_text="Thinking...")
                    if answer:
                        st.markdown("### 🧑‍🏫 Answer")
                        st.write(answer)
                        with st.expander("View retrieved manual context used for this answer"):
                            for i, chunk in enumerate(context_chunks, 1):
                                st.markdown(f"**Chunk {i}:**")
                                st.text(chunk[:1000])

    # ---------- Study & Reports ----------
    st.divider()
    st.subheader("📚 Study & Reports")
    st.caption("Complete experiment lifecycle: prepare before, analyze after, and generate study materials.")

    tab_prelab, tab_postlab, tab_report, tab_revision = st.tabs(
        ["📘 Pre-Lab Prep", "📕 Post-Lab Analysis", "📄 Lab Report", "🗒️ Revision Notes"]
    )

    if os.environ.get("GROQ_API_KEY"):
        with tab_prelab:
            if st.button("Generate pre-lab preparation guide", key="btn_prelab"):
                result = safe_ai_call(get_prelab_prep, selected_content,
                                       spinner_text="Preparing objectives and prerequisites...")
                if result:
                    st.markdown(result)

        with tab_postlab:
            if st.button("Generate post-lab analysis & viva questions", key="btn_postlab"):
                result = safe_ai_call(get_postlab_analysis, selected_content,
                                       spinner_text="Preparing analysis questions...")
                if result:
                    st.markdown(result)

        with tab_report:
            if st.button("Generate lab report template", key="btn_report"):
                report_text = safe_ai_call(generate_lab_report, selected_content, exp_labels[selected_idx],
                                            spinner_text="Drafting report structure...")
                if report_text:
                    st.markdown(report_text)
                    st.download_button(
                        "⬇️ Download report as text file",
                        data=report_text,
                        file_name=f"lab_report_exp_{experiments[selected_idx]['number']}.txt",
                        mime="text/plain",
                    )

        with tab_revision:
            if st.button("Generate revision notes", key="btn_revision"):
                result = safe_ai_call(generate_revision_notes, selected_content,
                                       spinner_text="Summarizing key points...")
                if result:
                    st.markdown(result)

    # ---------- Progress tracking ----------
    st.divider()
    st.subheader("📊 Course Progress")
    st.caption("Track which experiments you've completed in this session.")

    if "completed_experiments" not in st.session_state:
        st.session_state.completed_experiments = set()

    for i, exp in enumerate(experiments):
        label = exp_labels[i]
        checked = label in st.session_state.completed_experiments
        new_checked = st.checkbox(label, value=checked, key=f"progress_{i}")
        if new_checked:
            st.session_state.completed_experiments.add(label)
        else:
            st.session_state.completed_experiments.discard(label)

    done_count = len(st.session_state.completed_experiments)
    total_count = len(experiments)
    st.progress(done_count / total_count if total_count else 0)
    st.caption(f"{done_count} of {total_count} experiments marked complete (resets when you close the app).")
