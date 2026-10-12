"""UniZulu Engineering Chatbot: Streamlit interface.

Run locally / in a Codespace:   streamlit run app.py
"""
from __future__ import annotations

import os

import streamlit as st

from chatbot.engine import KnowledgeBase, answer_query

st.set_page_config(page_title="UniZulu Engineering Chatbot", page_icon="⚡", layout="centered")
st.title("⚡ UniZulu Engineering Chatbot")
st.caption("Exact lookups • BM25 retrieval • Qwen2.5-1.5B (CPU, optional)")


@st.cache_resource(show_spinner="Loading the knowledge base...")
def get_kb() -> KnowledgeBase:
    return KnowledgeBase()


@st.cache_resource(show_spinner="Loading the language model (the first run downloads about 1 GB)...")
def get_llm():
    from chatbot.llm import load_model

    return load_model()


kb = get_kb()

with st.sidebar:
    st.header("Settings")
    use_llm = st.toggle(
        "Write handbook answers with the language model",
        value=os.getenv("USE_LLM", "0") == "1",
        help="Needs llama-cpp-python (requirements-llm.txt) and about 2 GB of memory. "
        "When off, handbook answers show the matching handbook text directly.",
    )
    st.markdown(
        "**Try asking**\n"
        "- At what time is 5EEE421L on Monday?\n"
        "- When does the first semester start?\n"
        "- When do the second semester exams start?\n"
        "- What is the DP requirement?\n"
        "- What is the entry requirement for engineering?\n"
        "- What are the prerequisites for 5EEE421L?"
    )
    if st.button("Clear conversation"):
        st.session_state.messages = []
        st.rerun()

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

query = st.chat_input("Ask about your timetable, calendar, or handbook...")
if query:
    st.session_state.messages.append({"role": "user", "content": query})
    with st.chat_message("user"):
        st.markdown(query)

    with st.chat_message("assistant"):
        result = answer_query(kb, query)
        final = result.text

        if result.source == "handbook" and use_llm:
            try:
                from chatbot.llm import stream_answer

                llm = get_llm()
                holder = st.empty()
                for partial in stream_answer(llm, query, result.context):
                    final = partial or final
                    holder.markdown(final)
            except Exception as err:  # missing package, no internet for the model download, ...
                st.warning(f"Language model unavailable ({type(err).__name__}); showing the handbook text instead.")
                st.markdown(final)
        else:
            st.markdown(final)

        if result.context:
            with st.expander("Show retrieved source text (verify this answer yourself)"):
                for part in result.context:
                    st.text(part)
        st.caption(f"Answered from: {result.source}")

    st.session_state.messages.append({"role": "assistant", "content": final})
