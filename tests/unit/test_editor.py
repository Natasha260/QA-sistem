# tests/unit/test_editor.py
import pytest
from unittest.mock import AsyncMock, patch
from backend.agents.editor import Editor


class MockEditorNoLLM(Editor):
    async def _call_llm(self, prompt):
        return "Edited translation"


@pytest.mark.asyncio
class TestEditorProcess:
    async def test_skip_critic_failed(self):
        editor = MockEditorNoLLM(engine="ollama")
        state = {
            'source_text': 'Hello', 'source_lang': 'en', 'target_lang': 'ru',
            'translation': 'Привет', 'critic_failed': True
        }
        result = await editor.process(state)
        assert result['editor_changed'] == False
        assert result['editor_failed'] == False

    async def test_skip_no_issues(self):
        editor = MockEditorNoLLM(engine="ollama")
        state = {
            'source_text': 'Hello', 'source_lang': 'en', 'target_lang': 'ru',
            'translation': 'Привет', 'critic_failed': False,
            'validator_errors': [], 'validator_warnings': [], 'critic_suggestions': []
        }
        result = await editor.process(state)
        assert result['editor_changed'] == False

    async def test_edit_with_issues(self):
        editor = MockEditorNoLLM(engine="ollama")
        state = {
            'source_text': 'Hello', 'source_lang': 'en', 'target_lang': 'ru',
            'translation': 'Привет', 'critic_failed': False,
            'validator_errors': ['Test error'], 'validator_warnings': [], 'critic_suggestions': []
        }
        result = await editor.process(state)
        assert result['editor_changed'] == True
        assert result['translation'] == "Edited translation"

    async def test_empty_response(self):
        class EmptyEditor(Editor):
            async def _call_llm(self, prompt):
                return ""
        
        editor = EmptyEditor(engine="ollama")
        state = {
            'source_text': 'Hello', 'source_lang': 'en', 'target_lang': 'ru',
            'translation': 'Привет', 'critic_failed': False,
            'validator_errors': ['Test error'], 'validator_warnings': [], 'critic_suggestions': []
        }
        result = await editor.process(state)
        assert result['editor_failed'] == True
        assert result['editor_changed'] == False