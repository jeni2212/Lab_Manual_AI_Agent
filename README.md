# 🧪 Lab Manual Conversational Assistant (Track A)

An AI-powered assistant that reads uploaded lab manuals (PDF) and answers
student questions like *"What is the procedure of Exp 3 in Physics?"* or
*"Explain the theory behind this experiment in simple terms."*

Built with **Streamlit + LangChain-style RAG pipeline + FAISS + Groq (Llama 3.1)**.

## Features (Week 1-2 milestone)
- Upload a lab manual PDF
- Automatic experiment detection ("Experiment 1", "Exp. 2", etc.)
- Chunking + FAISS vector search over manual content
- Ask natural-language questions, get grounded answers from Groq's Llama 3.1
- Example quick-question buttons (procedure / theory / safety / equipment)

## Project Structure
```
lab-assistant/
├── app.py                  # Streamlit UI (main entry point)
├── utils/
│   ├── pdf_processor.py    # PDF text extraction + experiment splitting
│   ├── vector_store.py     # FAISS + sentence-transformers retrieval
│   └── groq_client.py      # Groq API wrapper (LLM calls)
├── requirements.txt
├── .env.example
└── README.md
```

## Setup Instructions

1. **Clone the repo & create a virtual environment**
   ```bash
   git clone <your-repo-url>
   cd lab-assistant
   python -m venv venv
   source venv/bin/activate   # Windows: venv\Scripts\activate
   ```

2. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

3. **Get a free Groq API key**
   - Sign up at [console.groq.com](https://console.groq.com)
   - Create an API key

4. **Configure your API key**
   Copy `.env.example` to `.env` and paste your key:
   ```
   GROQ_API_KEY=your_actual_key_here
   ```
   (Or just paste it into the sidebar text box when the app runs — either works.)

5. **Run the app**
   ```bash
   streamlit run app.py
   ```

6. **Use it**
   - Upload a lab manual PDF in the sidebar
   - Click **Process Manual**
   - Select an experiment (optional)
   - Ask a question or click an example question button

## Deployment (Streamlit Cloud)
1. Push this repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io) → New app → select repo → `app.py`
3. In **App settings → Secrets**, add:
   ```
   GROQ_API_KEY = "your_actual_key_here"
   ```
4. Deploy

## How It Works (Architecture)
1. **PDF → Text**: `PyPDF2` extracts raw text from every page.
2. **Experiment Splitting**: A regex pattern detects headings like
   `Experiment 3`, `Exp. 5`, `EXPERIMENT NO. 2` and splits the manual into
   per-experiment blocks.
3. **Chunking**: Each experiment's content is split into ~800-word chunks
   with overlap, so retrieval stays accurate even for long manuals.
4. **Embedding + FAISS**: Chunks are embedded with `all-MiniLM-L6-v2`
   (sentence-transformers) and indexed in a FAISS `IndexFlatL2` index.
5. **Retrieval**: A student's question + selected experiment title is used
   as the search query to pull the top-k most relevant chunks.
6. **Answer Generation**: The question + retrieved chunks are sent to
   Groq's `llama-3.1-8b-instant` model with a system prompt instructing it
   to answer only from the provided context (reduces hallucination) and
   to surface safety precautions when present.

## Roadmap (Weeks 3-8, per lab manual spec)
- [ ] Week 3-4: Add theory-to-practice correlation, equipment identification,
      safety precaution extraction as a dedicated feature
- [ ] Week 5-6: Subject specialization (Physics/Chemistry/CS), calculation
      assistance, Indian university curriculum pattern support
- [ ] Week 7-8: Polished dashboard UI, export (PDF/notes) of procedure guides,
      input validation, error handling, demo video

## Known Limitations (Week 1-2 stage)
- Experiment detection relies on manuals having a recognizable
  "Experiment N" heading — very unusually formatted manuals may fall back
  to a single combined block.
- Scanned/image-only PDFs (no embedded text layer) will need OCR — not yet
  implemented in this milestone.

## Team
_Add your team members and roles here._
