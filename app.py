"""
app.py
Streamlit UI for the Lab Manual Conversational Assistant (Track A, Week 1-2).

Flow:
1. Student uploads a lab manual PDF.
2. App extracts text and detects individual experiments.
3. App builds a FAISS index over chunks for retrieval.
4. Student picks an experiment and asks a question.
5. App retrieves relevant chunks and asks Groq (Llama 3.1) to answer.
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

# ---------- Session state ----------
if "experiments" not in st.session_state:
    st.session_state.experiments = []
if "vector_store" not in st.session_state:
    st.session_state.vector_store = None
if "manual_processed" not in st.session_state:
    st.session_state.manual_processed = False

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
    uploaded_file = st.file_uploader("Upload a PDF lab manual", type=["pdf"])

    if uploaded_file and st.button("Process Manual", type="primary", use_container_width=True):
        with st.spinner("Extracting text and identifying experiments..."):
            full_text = extract_text_from_pdf(uploaded_file)
            experiments = split_into_experiments(full_text)

            all_chunks = []
            for exp in experiments:
                all_chunks.extend(chunk_text(exp["content"]))

            store = ManualVectorStore()
            store.build(all_chunks)

            st.session_state.experiments = experiments
            st.session_state.vector_store = store
            st.session_state.manual_processed = True

        st.success(f"Found {len(experiments)} experiment(s)!")

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

    st.divider()
    st.subheader("🔬 Experiment Assistant")

    selected_content = experiments[selected_idx]["content"]

    tab_procedure, tab_equipment, tab_safety, tab_troubleshoot, tab_ask = st.tabs(
        ["📝 Procedure", "🧰 Equipment", "⚠️ Safety", "🛠️ Troubleshooting", "💬 Ask Anything"]
    )

    if not os.environ.get("GROQ_API_KEY"):
        st.warning("Enter your Groq API key in the sidebar to use these features.")
    else:
        with tab_procedure:
            if st.button("Generate step-by-step procedure", key="btn_procedure"):
                with st.spinner("Breaking down the procedure..."):
                    st.markdown(get_procedure_steps(selected_content))

        with tab_equipment:
            if st.button("Identify equipment & materials", key="btn_equipment"):
                with st.spinner("Scanning for equipment and materials..."):
                    st.markdown(get_equipment_list(selected_content))

        with tab_safety:
            if st.button("Extract safety precautions", key="btn_safety"):
                with st.spinner("Checking for safety information..."):
                    st.markdown(get_safety_precautions(selected_content))

        with tab_troubleshoot:
            issue = st.text_area(
                "Describe the problem you're facing",
                placeholder="e.g. My output doesn't match the expected result / equipment isn't responding...",
                key="troubleshoot_input",
            )
            if st.button("Get troubleshooting help", key="btn_troubleshoot") and issue:
                with st.spinner("Thinking through possible causes..."):
                    st.markdown(get_troubleshooting_help(selected_content, issue))

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

            if st.button("Ask", type="primary", key="btn_ask") and question:
                with st.spinner("Thinking..."):
                    store = st.session_state.vector_store
                    full_query = f"{exp_labels[selected_idx]}. {question}"
                    context_chunks = store.search(full_query, top_k=4)
                    answer = ask_lab_assistant(question, context_chunks)

                st.markdown("### 🧑‍🏫 Answer")
                st.write(answer)

                with st.expander("View retrieved manual context used for this answer"):
                    for i, chunk in enumerate(context_chunks, 1):
                        st.markdown(f"**Chunk {i}:**")
                        st.text(chunk[:1000])

    # ---------- Week 5-6: Comprehensive Experiment Guide ----------
    st.divider()
    st.subheader("📚 Study & Reports")
    st.caption("Complete experiment lifecycle: prepare before, analyze after, and generate study materials.")

    tab_prelab, tab_postlab, tab_report, tab_revision = st.tabs(
        ["📘 Pre-Lab Prep", "📕 Post-Lab Analysis", "📄 Lab Report", "🗒️ Revision Notes"]
    )

    if os.environ.get("GROQ_API_KEY"):
        with tab_prelab:
            if st.button("Generate pre-lab preparation guide", key="btn_prelab"):
                with st.spinner("Preparing objectives and prerequisites..."):
                    st.markdown(get_prelab_prep(selected_content))

        with tab_postlab:
            if st.button("Generate post-lab analysis & viva questions", key="btn_postlab"):
                with st.spinner("Preparing analysis questions..."):
                    st.markdown(get_postlab_analysis(selected_content))

        with tab_report:
            if st.button("Generate lab report template", key="btn_report"):
                with st.spinner("Drafting report structure..."):
                    report_text = generate_lab_report(selected_content, exp_labels[selected_idx])
                    st.markdown(report_text)
                    st.download_button(
                        "⬇️ Download report as text file",
                        data=report_text,
                        file_name=f"lab_report_exp_{experiments[selected_idx]['number']}.txt",
                        mime="text/plain",
                    )

        with tab_revision:
            if st.button("Generate revision notes", key="btn_revision"):
                with st.spinner("Summarizing key points..."):
                    st.markdown(generate_revision_notes(selected_content))

    # ---------- Progress tracking across experiments ----------
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
