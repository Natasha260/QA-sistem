# tests/unit/test_critic.py
import pytest
from unittest.mock import AsyncMock, patch
from backend.agents.critic import Critic


class TestCriticJSONExtraction:
    def test_extract_json_spans_simple(self):
        text = '{"adequacy": 5.0}'
        spans = Critic._extract_json_spans(text)
        assert len(spans) == 1
        assert spans[0] == '{"adequacy": 5.0}'

    def test_extract_json_spans_nested(self):
        text = '{"outer": {"inner": 1}}'
        spans = Critic._extract_json_spans(text)
        assert len(spans) == 1
        assert spans[0] == '{"outer": {"inner": 1}}'

    def test_extract_json_spans_multiple(self):
        text = '{"a": 1} {"b": 2}'
        spans = Critic._extract_json_spans(text)
        assert len(spans) == 2

    def test_extract_json_spans_with_braces_in_string(self):
        text = '{"reasoning": "test {braces} here"}'
        spans = Critic._extract_json_spans(text)
        assert len(spans) == 1

    def test_extract_json_spans_escaped_quotes(self):
        text = '{"reasoning": "test \\"quotes\\" here"}'
        spans = Critic._extract_json_spans(text)
        assert len(spans) == 1


class TestCriticParse:
    def test_parse_valid_json(self):
        critic = Critic(engine="ollama")
        text = '{"adequacy": 5.0, "fluency": 5.0, "terminology": 5.0, "overall_score": 5.0}'
        result = critic._parse_json_response(text)
        assert result['overall_score'] == 5.0

    def test_parse_fenced_json(self):
        critic = Critic(engine="ollama")
        text = '```json\n{"adequacy": 5.0, "fluency": 5.0, "terminology": 5.0, "overall_score": 5.0}\n```'
        result = critic._parse_json_response(text)
        assert result['overall_score'] == 5.0

    def test_parse_removes_failed_key(self):
        critic = Critic(engine="ollama")
        text = '{"adequacy": 5.0, "fluency": 5.0, "terminology": 5.0, "overall_score": 5.0, "failed": true}'
        result = critic._parse_json_response(text)
        assert "failed" not in result

    def test_fallback_extraction(self):
        critic = Critic(engine="ollama")
        text = 'SCORE: 7.5\nADEQUACY: 8.0'
        result = critic._fallback_extract(text)
        assert result['overall_score'] == 7.5
        assert result['adequacy'] == 8.0


class TestCriticFallback:
    def test_get_fallback_evaluation(self):
        critic = Critic(engine="ollama")
        result = critic._get_fallback_evaluation("Test error")
        assert result['overall_score'] is None
        assert result['failed'] == True
        assert "Test error" in result['reasoning']


@pytest.mark.asyncio
class TestCriticProcess:
    async def test_process_timeout(self):
        critic = Critic(engine="ollama")
        
        async def mock_call(prompt):
            import httpx
            raise httpx.TimeoutException("Timeout")
        
        critic._call_llm = mock_call
        state = {
            'source_text': 'Hello', 'source_lang': 'en', 'target_lang': 'ru',
            'translation': 'Привет', 'glossary_used': []
        }
        result = await critic.process(state)
        assert result['critic_failed'] == True
        assert result['critic_score'] is None