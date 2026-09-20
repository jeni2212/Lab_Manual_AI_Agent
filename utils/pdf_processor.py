"""
pdf_processor.py
Extracts text from an uploaded lab manual PDF and splits it into
individual experiments using pattern matching on common headings
like "Experiment 1", "Exp. 3", "EXPERIMENT NO. 5", etc.
"""

import re
import PyPDF2


def extract_text_from_pdf(file) -> str:
    """Extract raw text from a PDF file-like object."""
    reader = PyPDF2.PdfReader(file)
    full_text = []
    for page in reader.pages:
        text = page.extract_text() or ""
        full_text.append(text)
    return "\n".join(full_text)


# Matches: "Experiment 1", "Exp 1", "Exp. 1", "EXPERIMENT NO. 1", "Experiment-1", etc.
EXPERIMENT_HEADING_PATTERN = re.compile(
    r"(?im)^\s*(experiment|exp\.?)\s*(no\.?)?\s*[:\-]?\s*(\d+)\b.*$"
)


def split_into_experiments(full_text: str) -> list[dict]:
    """
    Split the full manual text into a list of experiments.
    Returns a list of dicts: {"number": int, "title": str, "content": str}
    """
    matches = list(EXPERIMENT_HEADING_PATTERN.finditer(full_text))

    if not matches:
        # Fallback: treat the whole manual as one block if no headings detected
        return [{
            "number": 1,
            "title": "Full Manual (no experiment headings detected)",
            "content": full_text.strip()
        }]

    experiments = []
    for i, match in enumerate(matches):
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(full_text)
        block = full_text[start:end].strip()

        # First line of the block is treated as the title line
        title_line = block.split("\n", 1)[0].strip()
        exp_number = int(match.group(3))

        experiments.append({
            "number": exp_number,
            "title": title_line,
            "content": block
        })

    return experiments


def chunk_text(text: str, chunk_size: int = 800, overlap: int = 100) -> list[str]:
    """Simple word-based chunking for embedding/retrieval."""
    words = text.split()
    chunks = []
    start = 0
    while start < len(words):
        end = start + chunk_size
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks
