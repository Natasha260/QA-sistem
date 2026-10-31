# tests/unit/test_base_agent.py
import pytest
from unittest.mock import AsyncMock, patch, MagicMock
from backend.agents.base import BaseAgent, Agent
from backend.core.exceptions import ModelNotLoadedError


class TestBaseAgent:
    def test_base_agent_is_abstract(self):
        assert "process" in BaseAgent.__abstractmethods__

    def test_cannot_instantiate_base_agent(self):
        with pytest.raises(TypeError):
            BaseAgent()


class TestAgent:
    @pytest.fixture
    def agent(self):
        class ConcreteAgent(Agent):
            hf_model_setting = "HF_TEST_MODEL"
            ollama_model_setting = "TEST_MODEL"
            
            async def process(self, state):
                return state
        
        return ConcreteAgent(engine="ollama")

    def test_agent_is_abstract(self):
        assert "process" in Agent.__abstractmethods__

    def test_resolve_engine_auto_with_hf(self):
        class ConcreteAgent(Agent):
            hf_model_setting = "HF_TEST_MODEL"
            ollama_model_setting = "TEST_MODEL"
            async def process(self, state): return state
        
        with patch.object(ConcreteAgent, '_hf_model_name', return_value="test-model"):
            agent = ConcreteAgent(engine="auto")
            assert agent.engine == "huggingface"

    def test_resolve_engine_auto_without_hf(self):
        class ConcreteAgent(Agent):
            hf_model_setting = "HF_TEST_MODEL"
            ollama_model_setting = "TEST_MODEL"
            async def process(self, state): return state
        
        with patch.object(ConcreteAgent, '_hf_model_name', return_value=None):
            agent = ConcreteAgent(engine="auto")
            assert agent.engine == "ollama"

    def test_resolve_engine_huggingface_fallback(self):
        class ConcreteAgent(Agent):
            hf_model_setting = "HF_TEST_MODEL"
            ollama_model_setting = "TEST_MODEL"
            async def process(self, state): return state
        
        with patch.object(ConcreteAgent, '_hf_model_name', return_value=None):
            agent = ConcreteAgent(engine="huggingface")
            assert agent.engine == "ollama"

    def test_resolve_engine_ollama(self):
        class ConcreteAgent(Agent):
            hf_model_setting = "HF_TEST_MODEL"
            ollama_model_setting = "TEST_MODEL"
            async def process(self, state): return state
        
        agent = ConcreteAgent(engine="ollama")
        assert agent.engine == "ollama"

    def test_resolve_engine_unknown(self):
        class ConcreteAgent(Agent):
            hf_model_setting = "HF_TEST_MODEL"
            ollama_model_setting = "TEST_MODEL"
            async def process(self, state): return state
        
        agent = ConcreteAgent(engine="google")
        assert agent.engine == "ollama"

    @pytest.mark.asyncio
    async def test_call_llm_ollama_seed(self, agent):
        agent.engine = "ollama"
        agent.model_name = "test-model"
        
        mock_response = MagicMock()
        mock_response.json.return_value = {"response": "test"}
        mock_response.raise_for_status = MagicMock()
        
        with patch('backend.agents.base.get_ollama_client') as mock_client:
            mock_client.return_value.post = AsyncMock(return_value=mock_response)
            result = await agent._call_llm("test prompt")
            
            call_args = mock_client.return_value.post.call_args
            payload = call_args[1]['json']
            assert payload['options']['seed'] == 42

    @pytest.mark.asyncio
    async def test_call_llm_huggingface_not_loaded(self):
        class ConcreteAgent(Agent):
            hf_model_setting = "HF_TEST_MODEL"
            ollama_model_setting = "TEST_MODEL"
            async def process(self, state): return state
        
        agent = ConcreteAgent(engine="huggingface")
        agent.engine = "huggingface"
        agent.hf_llm = None
        
        with pytest.raises(ModelNotLoadedError):
            await agent._call_llm("test")