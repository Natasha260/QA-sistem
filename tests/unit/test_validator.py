# tests/unit/test_validator.py
"""Юнит-тесты rule-based Validator для QA-ответов."""
import pytest

from backend.agents.validator import Validator, _contains_word


def _state(**extra):
    base = {
        "question": "Что такое сила?",
        "answer": "Сила — мера взаимодействия тел.",
        "chunks_used": [{"text": "Сила — векторная величина, мера взаимодействия тел."}],
        "terminology_used": [],
    }
    base.update(extra)
    return base


@pytest.mark.asyncio
class TestValidatorProcess:
    async def test_passes_good_answer(self):
        result = await Validator().process(_state())
        assert result["validation_passed"] is True
        assert result["validator_errors"] == []

    async def test_empty_answer(self):
        result = await Validator().process(_state(answer="   "))
        assert result["validation_passed"] is False
        assert any("empty" in e.lower() for e in result["validator_errors"])

    async def test_echo_of_question(self):
        result = await Validator().process(_state(answer="Что такое сила?"))
        assert any("identical" in e for e in result["validator_errors"])

    async def test_prompt_markers(self):
        result = await Validator().process(_state(answer="Context: Сила — мера взаимодействия."))
        assert any("prompt markers" in e for e in result["validator_errors"])

    async def test_grounding_warning_on_hallucination(self):
        result = await Validator().process(_state(
            answer="Квантовая декогеренция измеряет гильбертово пространство.",
        ))
        assert any("hallucination" in e for e in result["validator_errors"])

    async def test_refusal_warning(self):
        result = await Validator().process(_state(
            answer="Контекст не содержит ответа на этот вопрос.",
        ))
        assert any("refuses" in w for w in result["validator_warnings"])

    async def test_repetition_warning(self):
        result = await Validator().process(_state(
            answer="Сила сила сила действует на тело.",
        ))
        assert any("Repeated word" in w for w in result["validator_warnings"])


class TestContainsWord:
    def test_whole_word(self):
        assert _contains_word("сила — мера", "сила")
        assert not _contains_word("сильный ветер", "сила")
