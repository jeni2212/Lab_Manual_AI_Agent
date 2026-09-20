"""
vector_store.py
Builds a FAISS index over chunks of the lab manual using
sentence-transformers embeddings, and retrieves the most
relevant chunks for a given question.
"""

import numpy as np
import faiss
from sentence_transformers import SentenceTransformer

_MODEL_NAME = "all-MiniLM-L6-v2"


class ManualVectorStore:
    def __init__(self):
        self.model = SentenceTransformer(_MODEL_NAME)
        self.index = None
        self.chunks = []

    def build(self, chunks: list[str]):
        """Build a FAISS index from a list of text chunks."""
        self.chunks = chunks
        embeddings = self.model.encode(chunks, show_progress_bar=False)
        embeddings = np.array(embeddings).astype("float32")

        dim = embeddings.shape[1]
        self.index = faiss.IndexFlatL2(dim)
        self.index.add(embeddings)

    def search(self, query: str, top_k: int = 3) -> list[str]:
        """Return the top_k most relevant chunks for a query."""
        if self.index is None or not self.chunks:
            return []

        query_vec = self.model.encode([query]).astype("float32")
        distances, indices = self.index.search(query_vec, min(top_k, len(self.chunks)))

        results = [self.chunks[i] for i in indices[0] if i != -1]
        return results
