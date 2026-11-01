# streamlit_app/pages/8_System_Health.py
import streamlit as st

from streamlit_app.utils import api_client as api
from streamlit_app.utils.auth_guard import require_role

require_role("admin")

st.title("🩺 Состояние системы")
st.caption("Healthcheck сервисов: БД, Redis, Qdrant, Ollama. И статистика кэша.")

if st.button("Проверить", type="primary"):
    st.session_state["health_result"] = api.get_health()

health = st.session_state.get("health_result")
if health:
    st.caption(health.get("project", ""))
    icons = {"database": "🗄", "redis": "🧱", "qdrant": "🧭", "ollama": "🦙"}
    for service, info in health.get("services", {}).items():
        status = "✅" if info.get("healthy") else "❌"
        line = f"{icons.get(service, '⚙')} **{service}** {status}"
        if info.get("error"):
            line += f" — {info['error']}"
        st.write(line)

    model = health.get("model")
    if model:
        st.caption(f"LLM: {model} | ensemble: {', '.join(health.get('ensemble_models') or [])}")

st.divider()
st.subheader("Кэш запросов")
stats = api.get_cache_stats()
if stats:
    c1, c2 = st.columns(2)
    c1.metric("Записей", stats["total_entries"])
    c2.metric("Попаданий", stats["total_hits"])

eval_mode = st.text_input("Режим для очистки (пусто = всё)", key="clear_mode")
if st.button("Очистить кэш"):
    result = api.clear_cache(eval_mode or None)
    if result:
        st.success(f"Удалено записей: {result['deleted']}")
