# test_supervisor_2.py
import asyncio
import logging
logging.basicConfig(level=logging.INFO)

from backend.agents.supervisor import Supervisor

async def main():
    s = Supervisor()
    state = {
        'source_text': 'Hello',
        'source_lang': 'en',
        'target_lang': 'ru',
        'translation': 'Привет',
        'use_critic': True,
        'use_editor': False
    }
    r = await s.process(state)
    print('stopped:', r.get('supervisor_stopped_reason'))
    print('critic_score:', r.get('critic_score'))

if __name__ == '__main__':
    asyncio.run(main())