# 🧪 Lab Manual Conversational Assistant (Track A)

An AI-powered assistant that reads uploaded lab manuals (PDF) and answers
student questions like *"What is the procedure of Exp 3 in Physics?"* or
*"Explain the theory behind this experiment in simple terms."*

Built with **Streamlit + Retrieval-Augmented Generation (FAISS) + Groq LLM API**.

**🔗 Live demo:** [your Streamlit Cloud URL here]
**📦 Source code:** https://github.com/jeni2212/Lab_Manual_AI_Agent

## Features

**Core Assistant**
- Upload a lab manual PDF
- Automatic experiment detection (Experiment, Exp, Lab, Practical, Program, Unit, Assignment — with numbers)
- Chunking + FAISS vector search over manual content, grounded Q&A via Groq
- 📝 **Procedure** — numbered step-by-step breakdown
- 🧰 **Equipment** — identifies required equipment/materials mentioned in the manual
- ⚠️ **Safety** — extracts safety precautions (honestly flags when the manual doesn't cover them)
- 🛠️ **Troubleshooting** — describe a problem, get likely causes + fixes
- 💬 **Ask Anything** — open-ended Q&A with example question shortcuts

**Study & Reports**
- 📘 **Pre-Lab Prep** — objectives, prerequisites, checklist
- 📕 **Post-Lab Analysis** — reflection and viva questions
- 📄 **Lab Report Generator** — downloadable structured report template
- 🗒️ **Revision Notes** — concise exam-focused summary
- 📊 **Course Progress** — track completed experiments with a progress bar

**Reliability (Week 7-8 polish)**
- Input validation: empty files, oversized files, non-text PDFs, empty questions
- Friendly error messages for invalid API keys, rate limits, network issues, and corrupted PDFs — no raw crash tracebacks
- A "Clear and start over" button to reset the session

## Project Structure
```
lab-assistant/
├── app.py                      # Streamlit UI (main entry point)
├── utils/
│   ├── pdf_processor.py        # PDF text extraction + experiment splitting
│   ├── vector_store.py         # FAISS + sentence-transformers retrieval
│   ├── groq_client.py          # Groq API wrapper (general Q&A)
│   └── feature_extractor.py    # Procedure/equipment/safety/troubleshooting/pre-lab/post-lab/report/revision
├── requirements.txt
├── .env.example
└── README.md
```

## Setup Instructions

1. **Clone the repo & create a virtual environment**
   ```bash
   git clone https://github.com/jeni2212/Lab_Manual_AI_Agent.git
   cd Lab_Manual_AI_Agent
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Get a free Groq API key** at [console.groq.com](https://console.groq.com)

4. **Configure your API key** — copy `.env.example` to `.env` and paste your key:
   ```
   GROQ_API_KEY=your_actual_key_here
   ```
   (Or paste it into the sidebar text box when the app runs.)

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

6. **Use it** — upload a lab manual PDF, click **Process Manual**, select an experiment, and explore the tabs.

## Deployment (Streamlit Cloud)
1. Push this repo to GitHub
2. [share.streamlit.io](https://share.streamlit.io) → New app → select repo → `app.py`
3. In **Advanced settings → Secrets**, add:
   ```
   GROQ_API_KEY = "your_actual_key_here"
   ```
4. Deploy

## How It Works (Architecture)
1. **PDF → Text**: `PyPDF2` extracts raw text from every page.
2. **Experiment Splitting**: A regex pattern detects headings like
   `Experiment 3`, `Exp. 5`, `Lab 2`, `Practical 4`, `Program 1`, `Unit 6`, `Assignment 2`
   and splits the manual into per-experiment blocks.
3. **Chunking**: Each experiment's content is split into ~800-word chunks
   with overlap, so retrieval stays accurate even for long manuals.
4. **Embedding + FAISS**: Chunks are embedded with `all-MiniLM-L6-v2`
   (sentence-transformers) and indexed in a FAISS `IndexFlatL2` index.
5. **Retrieval**: A student's question + selected experiment title is used
   as the search query to pull the top-k most relevant chunks.
6. **Answer Generation**: The question + retrieved chunks are sent to
   Groq's `openai/gpt-oss-20b` model with a system prompt instructing it
   to answer only from the provided context (reduces hallucination) and
   to surface safety precautions when present — or honestly say when
   information isn't in the manual.

## Known Limitations
- Experiment detection relies on manuals having a recognizable heading format.
- Scanned/image-only PDFs (no embedded text layer) are not supported — the app
  now detects this case and shows a clear message rather than failing silently.
- All state (manual, index, progress) is session-only and resets on refresh;
  persistent storage is listed as future work.

## Future Scope
- Persistent storage (SQLite/ChromaDB) for manuals and progress across sessions
- Subject-specific specialization (Physics/Chemistry/CS calculation assistance)
- Multi-language support
- PDF/Word export of generated reports and revision notes

## Team Nexora

- JENITA REBEKKA C
- HASIN 

