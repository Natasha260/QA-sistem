# test_supervisor_3.py
import asyncio
import logging
logging.basicConfig(level=logging.INFO)

from backend.agents.supervisor import Supervisor

async def main():
    s = Supervisor()
    state = {
        'source_text': 'Hello world',
        'source_lang': 'en',
        'target_lang': 'ru',
        'translation': 'Привет мир',
        'use_critic': True,
        'use_editor': True
    }
    r = await s.process(state)
    print('stopped:', r.get('supervisor_stopped_reason'))
    print('iterations:', r.get('supervisor_iterations'))
    print('critic_score:', r.get('critic_score'))

if __name__ == '__main__':
    asyncio.run(main())