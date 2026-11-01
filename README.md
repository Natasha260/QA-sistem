# QA-система по университетским учебникам

**Вопросно-ответная система с гибридным поиском по учебникам и оценкой уверенности.**

## 📋 О проекте

Система отвечает на вопросы студентов по загруженным университетским
учебникам. Для каждого вопроса выполняется гибридный поиск релевантных
фрагментов текста (BM25 по чанкам + косинусный векторный поиск в Qdrant,
объединение через Reciprocal Rank Fusion). Контекст подставляется в
промпт локальной LLM (Ollama), а полученный ответ проверяется
мультиагентным конвейером:

- **Validator** — детерминированные rule-based проверки (пустой ответ,
  эхо вопроса, маркеры промпта, пересечение с контекстом, повторы);
- **Critic** — LLM-оценка по трём измерениям: релевантность,
  фактическая достоверность (faithfulness), полнота; итоговая оценка
  0–10 участвует в расчёте уверенности;
- **Editor** — правка ответа с опорой на контекст, если критик нашёл
  проблемы;
- **Supervisor** — оркестрация с адаптивным бюджетом итераций и
  rollback'ом неудачных правок.

**Оценка уверенности** — взвешенная комбинация оценки критика (вес 0.5),
релевантности найденного контекста (0.3) и результата валидации (0.2).
Ответы с уверенностью ниже порога попадают в Human-in-the-Loop очередь:
редактор подтверждает исправление, и оно сохраняется в базу
подтверждённых пар «вопрос → ответ», используемых как few-shot примеры.

## ✨ Возможности

- Гибридный RAG: BM25 (rank_bm25) + Qdrant (cosine) + RRF
- Чанкинг документов по заголовкам глав, абзацам и предложениям с перекрытием
- 8 ablation-режимов: `baseline`, `terminology`, `chunks_bm25`,
  `chunks_hybrid`, `agents_only`, `chunks_hybrid_critic`,
  `chunks_hybrid_editor`, `full`
- Локальный LLM (Ollama, qwen2.5:7b), опциональная HF-модель для критика/редактора
- Оценка уверенности (0–1) в каждом ответе
- Human-in-the-Loop: очередь неуверенных ответов, confirm/reject
- База подтверждённых QA-пар как few-shot примеры для похожих вопросов
- Кэш ответов в БД с подсчётом попаданий
- Метрики качества: Exact Match и token-F1 (SQuAD-стиль)
- Celery: переиндексация чанков, пакетный опрос, очистка кэша
- JWT auth с ролями user / editor / admin
- Streamlit UI (8 страниц)

## 🏗️ Архитектура

```
Client (Streamlit / HTTP)
        │
     Nginx :80 ──────────────► FastAPI (app :8000)
                                 │  ├── /health, /auth, /query*, /admin/*, /hitl/*
                                 │
                    ┌────────────┼─────────────┐
                    ▼            ▼             ▼
              PostgreSQL     Qdrant        Ollama (:11435→11434)
              (documents,    (chunks)      (qwen2.5:7b, ...)
               chunks,            │
               terminology,       │
               verified_qa,       │
               query_cache, users)│
                    ▲             │
                    └── Redis ◄───┴── Celery worker + beat
                        (broker)      (reindex chunks, batch query,
                                       cache cleanup)
```

Ключевые файлы:
- `backend/services/pipeline.py` — центральный `perform_qa`, 8 режимов, расчёт уверенности
- `backend/agents/` — Supervisor + Validator + Critic + Editor
- `backend/data/documents.py` — чанкинг и BM25-индекс учебников
- `backend/retrieval/` — BM25, векторный индекс чанков, RRF
- `backend/data/verified_qa.py` — подтверждённые пары «вопрос → ответ»
- `backend/hitl/` — ActiveLearningManager, FeedbackHandler
- `backend/eval/runner.py` — метрики EM / token-F1
- `backend/worker/` — Celery app + задачи
- `streamlit_app/` — UI (8 страниц)
- `infrastructure/` — Docker + Nginx

## 🚀 Быстрый старт

### Требования
- Docker 24+ и Docker Compose v2
- Python 3.12 + Poetry (для скриптов и тестов)
- 16 ГБ RAM (Ollama: qwen2.5:7b)

### Запуск

```bash
git clone <repo>
cd QA_sistem
python -m poetry install
cp infrastructure/.env.example infrastructure/.env
# Отредактируйте infrastructure/.env (POSTGRES_PASSWORD) и .env (JWT_SECRET_KEY)
docker compose -f infrastructure/docker-compose.yml up -d --build
```

Через 5–10 минут (скачивание моделей):
- API: http://localhost/docs (напрямую — http://localhost:8000/docs)
- Health: http://localhost/health

### Первый админ

```bash
python -m poetry run python -c "
import asyncio
from backend.core.database import init_db, close_db, get_db_session
from backend.utils.security import hash_password
from backend.db.models import User

async def m():
    await init_db()
    async with get_db_session() as s:
        s.add(User(
            email='admin@qa.local', username='admin',
            hashed_password=hash_password('Admin1234'),
            role='admin', is_active=True,
        ))
        await s.commit()
    await close_db()
asyncio.run(m())
"
```

### Загрузка данных

```bash
python -m poetry run python scripts/import/ingest_documents.py                    # учебники
python -m poetry run python scripts/import/add_terminology.py data/import/terminology.csv
python -m poetry run python scripts/import/add_qa_pairs.py data/import/verified_qa.json
```

Или через Streamlit-страницы «Учебники» и «Терминология» (роль admin).

Подробно про источники и критерии наполнения — `data/import/README.md`.

## 🧪 Тесты

```bash
python -m poetry run pytest tests/ -v
```

Сквозной e2e-тест требует живого стенда (Ollama): `RUNG_E2E=1 pytest tests/integration/test_end_to_end.py`.

## 🎯 Ablation-режимы

| Режим | Терминология | Поиск чанков | Critic | Editor |
|---|---|---|---|---|
| baseline | ✗ | — | ✗ | ✗ |
| terminology | ✓ | — | ✗ | ✗ |
| chunks_bm25 | ✓ | BM25 | ✗ | ✗ |
| chunks_hybrid | ✓ | BM25+векторы (RRF) | ✗ | ✗ |
| agents_only | ✗ | — | ✓ | ✓ |
| chunks_hybrid_critic | ✓ | BM25+векторы | ✓ | ✗ |
| chunks_hybrid_editor | ✓ | BM25+векторы | ✗ | ✓ |
| full | ✓ | BM25+векторы | ✓ | ✓ |

Выбор — через `EVAL_MODE` в `.env`, поле `eval_mode` в `/query` или селектор в Streamlit.

## 🖥️ Streamlit UI

Запуск отдельно (не в Docker):

```bash
python -m pip install -r streamlit_app/requirements.txt
streamlit run streamlit_app/app.py --server.port 8501
```

8 страниц: Вопрос-ответ, Пакетный запрос, Ансамбль, Учебники, Терминология, HITL, Оценка (EM/F1), Состояние.

## ⚠️ Известные ограничения

- **Streamlit не в Docker Compose** — dev-инструмент, запускается локально.
- **`beat_schedule` не настроен** — контейнер `beat` работает вхолостую; `clear_old_cache` запускается вручную.
- **`batch_query` не partial-fail-safe** — один упавший вопрос перезапускает весь batch.
- **`--concurrency=1`** — 1 задача одновременно.
- **`index_chunks` — полная пересборка** (drop + create коллекции Qdrant). Для больших корпусов нужен incremental upsert.
- **Кэш без TTL** — старые записи чистятся только задачей `clear_old_cache`.
- **Демонстрационные учебники** — синтетические конспекты; для реального качества загрузите лицензионно чистые тексты.

## 📚 Документация

- `data/import/README.md` — данные и критерии наполнения
- Методички этапов — репозиторий курса

## 👥 Авторы

Алексеева Оксана Павловна
