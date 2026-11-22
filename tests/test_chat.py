from src.chat import assistant


def test_fraud_questions_are_answered_from_fraud_sources() -> None:
    reply = assistant.answer("What should I do if my credit card was stolen?")

    assert reply["grounded"] is True
    assert reply["topic"] == "fraud"
    assert all(source["url"].startswith("https://") for source in reply["sources"])
    assert "[1]" in reply["answer"]


def test_news_questions_are_answered_from_news_sources() -> None:
    reply = assistant.answer("How can I tell if a news headline is fake?")

    assert reply["grounded"] is True
    assert reply["topic"] == "news"


def test_unrelated_questions_get_a_clarification_not_an_answer() -> None:
    reply = assistant.answer("What is a good recipe for chocolate cake?")

    assert reply["grounded"] is False
    assert reply["sources"] == []


def test_very_short_questions_ask_for_more_detail() -> None:
    reply = assistant.answer("fraud?")

    assert reply["grounded"] is False


class _FakeResponse:
    def __init__(self, body: bytes) -> None:
        self.body = body

    def __enter__(self):
        return self

    def __exit__(self, *args) -> None:
        pass

    def read(self) -> bytes:
        return self.body


def test_groq_writes_the_answer_when_a_key_is_set(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    sent = {}

    def fake_urlopen(request, timeout):
        sent["body"] = request.data.decode()
        return _FakeResponse(
            b'{"choices": [{"message": {"content": "Call your bank [1]."}}]}'
        )

    monkeypatch.setattr(assistant.urllib.request, "urlopen", fake_urlopen)

    reply = assistant.answer("What should I do if my credit card was stolen?")

    assert reply["mode"] == "groq"
    assert reply["answer"] == "Call your bank [1]."
    assert "[1]" in sent["body"]


def test_groq_failure_falls_back_to_the_passages(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")

    def broken_urlopen(request, timeout):
        raise OSError("network down")

    monkeypatch.setattr(assistant.urllib.request, "urlopen", broken_urlopen)

    reply = assistant.answer("What should I do if my credit card was stolen?")

    assert reply["mode"] == "retrieval"
    assert reply["grounded"] is True


def test_groq_saying_no_answer_becomes_a_clarification(monkeypatch) -> None:
    monkeypatch.setenv("GROQ_API_KEY", "test-key")
    monkeypatch.setattr(
        assistant.urllib.request,
        "urlopen",
        lambda request, timeout: _FakeResponse(
            b'{"choices": [{"message": {"content": "NO_ANSWER"}}]}'
        ),
    )

    reply = assistant.answer("best chocolate cake recipe please")

    assert reply["grounded"] is False
    assert reply["sources"] == []
