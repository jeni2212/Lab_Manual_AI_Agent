"""
groq_client.py
Thin wrapper around the Groq API for generating student-friendly
explanations of lab procedures and theory, grounded in the
retrieved manual content (RAG-style prompting).
"""

import os
from groq import Groq

MODEL_NAME = "openai/gpt-oss-20b"


def get_client() -> Groq:
    api_key = os.environ.get("GROQ_API_KEY")
    if not api_key:
        raise ValueError(
            "GROQ_API_KEY not found. Add it to your .env file or Streamlit secrets."
        )
    return Groq(api_key=api_key)


SYSTEM_PROMPT = """You are a helpful laboratory assistant for Indian university students.
You explain experiment procedures and theory clearly, in simple step-by-step language.
Always base your answer ONLY on the manual content provided in the context.
If the context does not contain enough information, say so honestly instead of guessing.
When relevant, mention any safety precautions found in the context."""


def ask_lab_assistant(question: str, context_chunks: list[str]) -> str:
    """
    Send the question + retrieved manual context to Groq and return
    a clear explanation.
    """
    client = get_client()
    context_text = "\n\n---\n\n".join(context_chunks) if context_chunks else "No relevant context found."

    user_prompt = f"""Lab manual context:
{context_text}

Student question: {question}

Answer clearly and step-by-step where applicable."""

    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.3,
        max_tokens=800,
    )
    return response.choices[0].message.content
