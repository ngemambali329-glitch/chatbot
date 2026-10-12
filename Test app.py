"""Smoke test: the Streamlit app starts and answers a question."""
from pathlib import Path

import pytest

AppTest = pytest.importorskip("streamlit.testing.v1").AppTest
APP = str(Path(__file__).resolve().parents[1] / "app.py")


def test_app_answers_a_question():
    at = AppTest.from_file(APP, default_timeout=60).run()
    assert not at.exception
    at.chat_input[0].set_value("At what time is 5EEE421L on Monday?").run()
    assert not at.exception
    texts = [m.markdown[0].value for m in at.chat_message if m.markdown]
    assert any("10:30-11:20" in t for t in texts)
