"""Optional language-model step: Qwen2.5-1.5B-Instruct (GGUF, 4-bit) running on the CPU via llama.cpp.

The chatbot works without this module (exact lookups + extractive handbook answers).
Install the extras with:  pip install -r requirements-llm.txt
"""
from __future__ import annotations

import re
from collections.abc import Iterator

MODEL_REPO = "Qwen/Qwen2.5-1.5B-Instruct-GGUF"
MODEL_FILE = "qwen2.5-1.5b-instruct-q4_k_m.gguf"

SYSTEM_INSTRUCTIONS = (
    "You are a careful assistant for University of Zululand Engineering students. "
    "You will be given CONTEXT extracted from the university's own documents "
    "(timetable, calendar, handbook) and a QUESTION. Follow these rules strictly:\n"
    "1. Use ONLY facts explicitly stated in the CONTEXT. Never use outside knowledge.\n"
    "2. If the CONTEXT does not contain enough information, say plainly that this is "
    "not covered in the provided documents. Do not guess.\n"
    "3. Write complete, grammatically correct English sentences and never stop mid-sentence.\n"
    "4. If the question asks for a list, use a clean numbered or bulleted list.\n"
    "5. Answer directly without repeating the question.\n"
    "6. If the CONTEXT only partly answers the question, answer what it supports and "
    "say what is missing."
)

_SENTENCE_END = re.compile(r"[.!?][\"')\]]*\s*$")


def load_model(n_threads: int = 2):
    """Download (first run only) and load the model. Raises ImportError if llama-cpp-python is missing."""
    from huggingface_hub import hf_hub_download
    from llama_cpp import Llama

    path = hf_hub_download(repo_id=MODEL_REPO, filename=MODEL_FILE)
    return Llama(model_path=path, n_ctx=2048, n_threads=n_threads, n_batch=256, verbose=False)


def _trim_to_sentence(text: str, min_keep: int = 20) -> str:
    text = text.strip()
    if not text or _SENTENCE_END.search(text):
        return text
    last = max(text.rfind("."), text.rfind("!"), text.rfind("?"))
    return text[: last + 1].strip() if last != -1 and last + 1 >= min_keep else text


def stream_answer(llm, query: str, context: list[str], max_context_chars: int = 1200) -> Iterator[str]:
    """Yield the growing answer text (suitable for Streamlit placeholders)."""
    ctx = ""
    for chunk in context:
        if len(ctx) + len(chunk) > max_context_chars:
            room = max_context_chars - len(ctx)
            if room > 100:
                ctx += chunk[:room]
            break
        ctx += chunk + "\n\n"

    prompt = (
        f"<|im_start|>system\n{SYSTEM_INSTRUCTIONS}<|im_end|>\n"
        f"<|im_start|>user\nCONTEXT:\n{ctx.strip()}\n\nQUESTION: {query}\n\n"
        f"Answer using only the CONTEXT above, in complete sentences:<|im_end|>\n"
        f"<|im_start|>assistant\n"
    )
    stream = llm(
        prompt, max_tokens=400, temperature=0.2, top_p=0.9, repeat_penalty=1.15,
        stop=["<|im_end|>"], stream=True,
    )
    partial, n = "", 0
    for piece in stream:
        partial += piece["choices"][0]["text"]
        n += 1
        if n >= 6:
            yield partial.strip()
            n = 0
    yield _trim_to_sentence(partial)
