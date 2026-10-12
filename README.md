# UniZulu Engineering Chatbot

A chatbot that answers questions about the **2026 first-semester timetable**, the **2026 General Calendar**
and the **2024 Engineering Department handbook** (UNIZULU, Richards Bay campus).

How an answer is produced:

| Step | Used when | How |
| --- | --- | --- |
| 1. Timetable lookup | the question names a module code and asks about its schedule (`At what time is 5EEE421L on Monday?`) | exact lookup in `data/processed/timetable_entries.json` |
| 2. Calendar search | the question mentions semester, exams, registration, holidays... | exact search of the dated events in the calendar |
| 3. Handbook retrieval | everything else (`What is the DP requirement?`) | BM25 ranking ([`rank_bm25`](https://pypi.org/project/rank-bm25/)) over the handbook; the matching passage is shown, or written up by **Qwen2.5-1.5B-Instruct** when the language model is switched on |
| 4. Out of scope | nothing relevant is found | the chatbot says so instead of guessing |

The interface is [Streamlit](https://streamlit.io). The language model runs on the CPU through
[llama.cpp](https://github.com/ggml-org/llama.cpp) and is optional.

## Run it on GitHub (Codespaces)

1. Push this folder to a new GitHub repository. (`data/processed/` is optional: the Codespace builds it from `data/raw/` by itself.)
2. Click **Code → Codespaces → Create codespace on main**.
3. Wait for the set-up to finish. The chatbot starts by itself and opens in a browser tab (port 8501).
   If it doesn't, run `streamlit run app.py` in the terminal.

The Codespace installs the optional language model and turns it on (`USE_LLM=1`); the first question that needs it
downloads about 1 GB. A 4-core / 8 GB Codespace is enough. Use the sidebar switch to turn the model off for instant answers.

## Run it on your own computer

```bash
pip install -r requirements.txt            # chatbot with exact lookups and extractive answers
streamlit run app.py

# optional language model
pip install -r requirements-llm.txt --extra-index-url https://abetlen.github.io/llama-cpp-python/whl/cpu
USE_LLM=1 streamlit run app.py
```

## Deploy for free on Streamlit Community Cloud

Sign in at [share.streamlit.io](https://share.streamlit.io) with GitHub, choose this repository and `app.py`.
Leave the language model off: the free tier does not have enough memory for it, but everything else works.

## Share a running copy with ngrok (optional)

```bash
pip install pyngrok
export NGROK_AUTHTOKEN=<your token>
python scripts/share_with_ngrok.py
```

## GitHub Actions

`.github/workflows/ci.yml` runs on every push. It rebuilds the knowledge files from `data/raw/`, checks that the
committed `data/processed/` files are up to date, and runs the tests (real timetable, calendar and handbook questions,
plus a start-up test of the app).

## Updating the documents

Replace the files in `data/raw/` (keep the names `timetable.pdf`, `calendar.txt`, `handbook.txt`), then:

```bash
pip install -r requirements-dev.txt
python scripts/build_knowledge.py
pytest
```

`calendar.txt` and `handbook.txt` are text extracted from the university PDFs; the timetable is read straight from the PDF
by the position of each module code in the grid.

## Project layout

```
app.py                        Streamlit chat interface
chatbot/engine.py             routing, timetable lookup, calendar search, BM25 retrieval
chatbot/llm.py                optional Qwen2.5-1.5B step (llama.cpp, CPU)
scripts/build_knowledge.py    builds data/processed/ from data/raw/
scripts/share_with_ngrok.py   optional public link
data/raw/                     the three source documents
data/processed/               generated JSON used by the chatbot
tests/                        pytest suite
.devcontainer/                GitHub Codespaces set-up
.github/workflows/ci.yml      GitHub Actions
```

## Known limitations

- The timetable has no venues, only days, times and groups.
- Calendar answers show the (up to three) closest events; the calendar has no "end of lectures" entry.
- Handbook answers are the matching passage, so table-like questions (credits, NQF levels) can look messy.
  Turn the language model on for written answers, and always check important answers against the official documents.
- Only the Richards Bay engineering timetable is included.
