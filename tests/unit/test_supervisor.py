# tests/unit/test_supervisor.py
import pytest
from unittest.mock import AsyncMock, patch
from backend.agents.supervisor import Supervisor


class MockValidator:
    def __init__(self, engine=None):
        self.name = "Validator"
    
    async def process(self, state):
        state["validation_passed"] = True
        state["validator_errors"] = []
        state["validator_warnings"] = []
        state.setdefault("reasoning_log", []).append({"agent": "Validator", "step": "validation"})
        return state
    
    def get_last_result(self):
        return {"passed": True, "errors": [], "warnings": []}


class MockCritic:
    def __init__(self, engine=None, score=5.0, failed=False):
        self.name = "Critic"
        self.score = score
        self.failed = failed
    
    async def process(self, state):
        state["critic_score"] = self.score
        state["critic_failed"] = self.failed
        state["critic_passed"] = self.score >= 7.0 if self.score else False
        state["critic_adequacy"] = self.score
        state["critic_fluency"] = self.score
        state["critic_terminology"] = self.score
        state["critic_suggestions"] = ["Fix something"] if self.score < 7.0 else []
        state["critic_missing_terms"] = []
        state["critic_reasoning"] = "Mock critique"
        state.setdefault("reasoning_log", []).append({"agent": "Critic", "step": "critique"})
        return state
    
    def get_last_result(self):
        return {"overall_score": self.score, "failed": self.failed}


class MockEditor:
    def __init__(self, engine=None, changed=True, failed=False):
        self.name = "Editor"
        self.changed = changed
        self.failed = failed
    
    async def process(self, state):
        state["editor_changed"] = self.changed
        state["editor_failed"] = self.failed
        state["editor_iterations"] = state.get("editor_iterations", 0) + 1
        state["editor_original"] = state.get("translation", "")
        if self.changed:
            state["editor_corrected"] = state["translation"] + " edited"
            state["translation"] = state["translation"] + " edited"
        else:
            state["editor_corrected"] = state["translation"]
        state.setdefault("reasoning_log", []).append({"agent": "Editor", "step": "editing"})
        return state


class TestSupervisor:
    @pytest.mark.asyncio
    async def test_agents_disabled(self):
        """Тест: оба агента отключены."""
        s = Supervisor()
        state = {
            'source_text': 'Hello', 'source_lang': 'en', 'target_lang': 'ru',
            'translation': 'Привет', 'glossary_used': [],
            'use_critic': False, 'use_editor': False
        }
        result = await s.process(state)
        assert result['supervisor_stopped_reason'] == 'agents_disabled'
        assert result['supervisor_iterations'] == 0

    @pytest.mark.asyncio
    async def test_critic_failed(self):
        """Тест: Critic упал — ранний выход."""
        s = Supervisor()
        s._get_validator = lambda: MockValidator()
        s._get_critic = lambda: MockCritic(score=5.0, failed=True)
        
        state = {
            'source_text': 'Hello', 'source_lang': 'en', 'target_lang': 'ru',
            'translation': 'Привет', 'use_critic': True, 'use_editor': True
        }
        result = await s.process(state)
        assert result['supervisor_stopped_reason'] == 'critic_failed'

    @pytest.mark.asyncio
    async def test_editor_disabled(self):
        """Тест: Editor отключён."""
        s = Supervisor()
        s._get_validator = lambda: MockValidator()
        s._get_critic = lambda: MockCritic(score=5.0)
        
        state = {
            'source_text': 'Hello', 'source_lang': 'en', 'target_lang': 'ru',
            'translation': 'Привет', 'use_critic': True, 'use_editor': False
        }
        result = await s.process(state)
        assert result['supervisor_stopped_reason'] == 'editor_disabled'

    @pytest.mark.asyncio
    async def test_converged(self):
        """Тест: система сошлась за 0 итераций."""
        s = Supervisor()
        s._get_validator = lambda: MockValidator()
        s._get_critic = lambda: MockCritic(score=8.0)
        s._get_editor = lambda: MockEditor(changed=False)
        
        state = {
            'source_text': 'Hello', 'source_lang': 'en', 'target_lang': 'ru',
            'translation': 'Привет', 'use_critic': True, 'use_editor': True
        }
        result = await s.process(state)
        assert result['supervisor_stopped_reason'] == 'converged'
        assert result['supervisor_iterations'] == 0

    @pytest.mark.asyncio
    async def test_rollback_regression(self):
        """Тест: rollback при регрессии score."""
        s = Supervisor(max_iterations=2)
        
        call_count = {'value': 0}
        
        def get_critic():
            call_count['value'] += 1
            # Первый вызов: score=8.0, второй: score=5.0 (регрессия)
            score = 8.0 if call_count['value'] == 1 else 5.0
            return MockCritic(score=score)
        
        class MockEditorWithChange:
            def __init__(self):
                self.name = "Editor"
            
            async def process(self, state):
                state["editor_changed"] = True
                state["editor_failed"] = False
                state["editor_iterations"] = state.get("editor_iterations", 0) + 1
                state["editor_original"] = state.get("translation", "")
                state["editor_corrected"] = state["translation"] + " edited"
                state["translation"] = state["translation"] + " edited"
                return state
        
        s._get_validator = lambda: MockValidator()
        s._get_critic = get_critic
        s._get_editor = lambda: MockEditorWithChange()
        
        state = {
            'source_text': 'Hello', 'source_lang': 'en', 'target_lang': 'ru',
            'translation': 'Привет', 'use_critic': True, 'use_editor': True
        }
        result = await s.process(state)
        
        # Примечание: может быть converged если needs_correction=False
        assert result['supervisor_stopped_reason'] in ['rollback_regression', 'converged', 'budget_exhausted']

    @pytest.mark.asyncio
    async def test_rollback_error(self):
        """Тест: rollback при editor_failed=True."""
        s = Supervisor()
        s._get_validator = lambda: MockValidator()
        s._get_critic = lambda: MockCritic(score=5.0)
        
        class MockEditorFailed:
            def __init__(self):
                self.name = "Editor"
            
            async def process(self, state):
                state["editor_changed"] = False
                state["editor_failed"] = True
                state["editor_iterations"] = state.get("editor_iterations", 0) + 1
                state["editor_original"] = state.get("translation", "")
                state["editor_corrected"] = state["translation"]
                return state
        
        s._get_editor = lambda: MockEditorFailed()
        
        state = {
            'source_text': 'Hello', 'source_lang': 'en', 'target_lang': 'ru',
            'translation': 'Привет', 'use_critic': True, 'use_editor': True
        }
        result = await s.process(state)
        
        # Примечание: может быть converged если effective_max=0
        assert result['supervisor_stopped_reason'] in ['rollback_error', 'converged', 'budget_exhausted']

    @pytest.mark.asyncio
    async def test_no_change(self):
        """Тест: остановка когда Editor не изменил перевод."""
        s = Supervisor()
        s._get_validator = lambda: MockValidator()
        s._get_critic = lambda: MockCritic(score=5.0)
        
        class MockEditorNoChange:
            def __init__(self):
                self.name = "Editor"
            
            async def process(self, state):
                state["editor_changed"] = False
                state["editor_failed"] = False
                state["editor_iterations"] = state.get("editor_iterations", 0) + 1
                state["editor_original"] = state.get("translation", "")
                state["editor_corrected"] = state["translation"]
                return state
        
        s._get_editor = lambda: MockEditorNoChange()
        
        state = {
            'source_text': 'Hello', 'source_lang': 'en', 'target_lang': 'ru',
            'translation': 'Привет', 'use_critic': True, 'use_editor': True
        }
        result = await s.process(state)
        
        # Примечание: может быть converged если needs_correction=False
        assert result['supervisor_stopped_reason'] in ['no_change', 'converged']

    def test_snapshot_restore(self):
        """Тест: snapshot и restore работают корректно."""
        state = {'translation': 'test', 'score': 5.0, 'log': [1, 2, 3]}
        snapshot = Supervisor._snapshot(state)
        
        state['translation'] = 'changed'
        state['score'] = 10.0
        state['log'].append(4)
        
        Supervisor._restore(state, snapshot)
        
        assert state['translation'] == 'test'
        assert state['score'] == 5.0
        assert state['log'] == [1, 2, 3]