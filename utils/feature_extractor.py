"""
feature_extractor.py
Week 3-4: Dedicated laboratory assistant features beyond general Q&A.

Each function sends a focused, structured prompt to Groq so the output
is consistent and easy to render in the UI (numbered steps, bullet lists).
"""

from utils.groq_client import get_client, MODEL_NAME


def _ask(system_prompt: str, user_prompt: str) -> str:
    client = get_client()
    response = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        temperature=0.2,
        max_tokens=900,
    )
    return response.choices[0].message.content


def get_procedure_steps(experiment_content: str) -> str:
    """Break the experiment down into a clear, numbered step-by-step procedure."""
    system_prompt = (
        "You are a laboratory assistant. Extract and reorganize the experimental "
        "procedure from the manual content into a clean, numbered step-by-step list. "
        "Use ONLY information present in the content. If the content does not describe "
        "a clear procedure, say so honestly. Keep steps concise and actionable."
    )
    user_prompt = f"Manual content:\n\n{experiment_content[:6000]}\n\nList the procedure as numbered steps."
    return _ask(system_prompt, user_prompt)


def get_equipment_list(experiment_content: str) -> str:
    """Identify equipment, materials, tools, or software mentioned in the content."""
    system_prompt = (
        "You are a laboratory assistant. Identify all equipment, materials, chemicals, "
        "tools, or software mentioned in the manual content needed to perform this experiment. "
        "Present as a bullet list with a one-line note on how each is used if the manual mentions it. "
        "Use ONLY information present in the content. If nothing is mentioned, say so honestly."
    )
    user_prompt = f"Manual content:\n\n{experiment_content[:6000]}\n\nList the required equipment/materials."
    return _ask(system_prompt, user_prompt)


def get_safety_precautions(experiment_content: str) -> str:
    """Extract safety precautions, warnings, or hazard notes from the content."""
    system_prompt = (
        "You are a laboratory safety assistant. Extract any safety precautions, warnings, "
        "hazard notes, or handling instructions mentioned in the manual content. "
        "Present as a bullet list. Use ONLY information present in the content. "
        "If the manual does not mention safety information, say so honestly and suggest "
        "the student consult their instructor or institutional safety guidelines instead "
        "of inventing precautions."
    )
    user_prompt = f"Manual content:\n\n{experiment_content[:6000]}\n\nList any safety precautions mentioned."
    return _ask(system_prompt, user_prompt)


def get_troubleshooting_help(experiment_content: str, issue_description: str) -> str:
    """Suggest likely causes and fixes for a problem the student is experiencing."""
    system_prompt = (
        "You are a laboratory troubleshooting assistant. A student is stuck on a problem "
        "during this experiment. Using the manual content as context, suggest 2-4 likely "
        "causes and practical fixes. Be concise and practical. If the manual doesn't cover "
        "this issue, use general lab/technical knowledge but say clearly that it's general "
        "advice, not from the manual."
    )
    user_prompt = (
        f"Manual content:\n\n{experiment_content[:6000]}\n\n"
        f"Student's problem: {issue_description}\n\n"
        f"What might be causing this and how can they fix it?"
    )
    return _ask(system_prompt, user_prompt)


def get_prelab_prep(experiment_content: str) -> str:
    """Generate pre-lab preparation: objectives, prerequisite knowledge, things to review beforehand."""
    system_prompt = (
        "You are a laboratory preparation assistant. Based on the manual content, help a student "
        "prepare BEFORE starting this experiment. Provide: (1) Clear learning objectives, "
        "(2) Prerequisite concepts/knowledge they should review beforehand, "
        "(3) A short pre-lab checklist (things to read, bring, or set up). "
        "Use ONLY information reasonably inferable from the manual content. Keep it practical."
    )
    user_prompt = f"Manual content:\n\n{experiment_content[:6000]}\n\nGenerate pre-lab preparation guidance."
    return _ask(system_prompt, user_prompt)


def get_postlab_analysis(experiment_content: str) -> str:
    """Generate post-lab guidance: analysis questions, viva questions, what to reflect on."""
    system_prompt = (
        "You are a laboratory assistant helping a student AFTER completing this experiment. "
        "Based on the manual content, provide: (1) 3-5 analysis/reflection questions to check "
        "understanding, (2) 3-5 likely viva/oral exam questions an examiner might ask about this "
        "experiment, (3) Common mistakes students make with this type of experiment. "
        "Use ONLY information reasonably inferable from the manual content."
    )
    user_prompt = f"Manual content:\n\n{experiment_content[:6000]}\n\nGenerate post-lab analysis and viva questions."
    return _ask(system_prompt, user_prompt)


def generate_lab_report(experiment_content: str, experiment_title: str) -> str:
    """Generate a structured lab report template filled in with manual content."""
    system_prompt = (
        "You are a laboratory report writing assistant. Generate a well-structured lab report "
        "template for this experiment, using standard Indian university lab report format: "
        "Aim, Apparatus/Requirements, Theory (brief), Procedure (numbered), Observation/Result "
        "section (as a table or placeholder if data isn't in the manual), Precautions, and "
        "Conclusion (as a fill-in-the-blank template since the student's actual results aren't "
        "known). Fill in whatever the manual content actually provides; use clearly marked "
        "placeholders like [Student to fill in] for anything the student must complete themselves."
    )
    user_prompt = (
        f"Experiment: {experiment_title}\n\nManual content:\n\n{experiment_content[:6000]}\n\n"
        f"Generate a lab report template."
    )
    return _ask(system_prompt, user_prompt)


def generate_revision_notes(experiment_content: str) -> str:
    """Generate concise exam-focused revision notes/summary."""
    system_prompt = (
        "You are a study assistant creating exam revision notes. Based on the manual content, "
        "produce a concise summary suitable for last-minute exam revision: key definitions, "
        "important formulas or concepts, and 3-5 one-line takeaways. Keep it short and scannable "
        "— this is for quick review, not deep explanation. Use ONLY information from the manual content."
    )
    user_prompt = f"Manual content:\n\n{experiment_content[:6000]}\n\nGenerate concise revision notes."
    return _ask(system_prompt, user_prompt)
