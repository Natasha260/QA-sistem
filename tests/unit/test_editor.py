# tests/unit/test_editor.py
import pytest
from unittest.mock import AsyncMock, patch
from backend.agents.editor import Editor


class MockEditorNoLLM(Editor):
    async def _call_llm(self, prompt):
        return "Отредактированный ответ"


def _state(**extra):
    base = {
        'question': 'Что такое сила?',
        'answer': 'Мера взаимодействия тел.',
        'critic_failed': False,
        'validator_errors': [],
        'validator_warnings': [],
        'critic_suggestions': [],
        'chunks_used': [],
    }
    base.update(extra)
    return base


@pytest.mark.asyncio
class TestEditorProcess:
    async def test_skip_critic_failed(self):
        editor = MockEditorNoLLM(engine="ollama")
        result = await editor.process(_state(critic_failed=True))
        assert result['editor_changed'] == False
        assert result['editor_failed'] == False

    async def test_skip_no_issues(self):
        editor = MockEditorNoLLM(engine="ollama")
        result = await editor.process(_state())
        assert result['editor_changed'] == False

    async def test_edit_with_issues(self):
        editor = MockEditorNoLLM(engine="ollama")
        result = await editor.process(_state(validator_errors=['Test error']))
        assert result['editor_changed'] == True
        assert result['answer'] == "Отредактированный ответ"

    async def test_empty_response(self):
        class EmptyEditor(Editor):
            async def _call_llm(self, prompt):
                return ""

        editor = EmptyEditor(engine="ollama")
        result = await editor.process(_state(validator_errors=['Test error']))
        assert result['editor_failed'] == True
        assert result['editor_changed'] == False
