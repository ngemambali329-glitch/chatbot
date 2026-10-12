"""The language-model step, tested with a stand-in model (the real one needs a ~1 GB download)."""
from chatbot.llm import _trim_to_sentence, stream_answer


class FakeLlama:
    def __init__(self, text):
        self.text, self.prompt = text, None

    def __call__(self, prompt, **kwargs):
        self.prompt = prompt
        assert kwargs["stream"] is True and "<|im_end|>" in kwargs["stop"]
        return ({"choices": [{"text": tok}]} for tok in self.text.split(" ") for tok in (tok + " ",))


def test_prompt_contains_context_and_question():
    llm = FakeLlama("The DP requirement is 40%.")
    list(stream_answer(llm, "What is the DP requirement?", ["DP Requirement 40% Continuous Assessment Mark"]))
    assert "DP Requirement 40%" in llm.prompt and "QUESTION: What is the DP requirement?" in llm.prompt


def test_answer_is_cut_back_to_a_full_sentence():
    llm = FakeLlama("The DP requirement is 40%. It also needs 80% attendance and")
    assert list(stream_answer(llm, "q", ["ctx"]))[-1] == "The DP requirement is 40%."


def test_trim_keeps_text_without_a_full_stop():
    assert _trim_to_sentence("40%") == "40%"
