# streamlit_app/pages/6_HITL.py
import streamlit as st

from streamlit_app.utils import api_client as api
from streamlit_app.utils.auth_guard import require_role

require_role("editor", "admin")

st.title("✍️ Human-in-the-Loop")
st.caption("Неуверенные ответы: редактор подтверждает исправление — оно попадает в базу verified QA.")

stats = api.get_uncertain_stats()
if stats:
    c1, c2, c3 = st.columns(3)
    c1.metric("Ожидают", stats["pending"])
    c2.metric("Подтверждены", stats["confirmed"])
    c3.metric("Отклонены", stats["rejected"])

st.divider()
st.subheader("Очередь")

data = api.get_uncertain(limit=20)
examples = (data or {}).get("examples") or []
for ex in examples:
    st.markdown(f"**Вопрос:** {ex['question']}")
    st.caption(
        f"draft answer: {ex.get('draft_answer', '')} | "
        f"critic: {ex.get('critic_score', 'N/A')}"
    )
    corrected = st.text_input("Исправленный ответ", key=f"fix_{ex['id']}")
    c1, c2 = st.columns(2)
    with c1:
        if st.button("✅ Подтвердить", key=f"ok_{ex['id']}"):
            if corrected.strip():
                api.confirm_uncertain(ex["id"], corrected)
                st.rerun()
            else:
                st.warning("Введите исправленный ответ.")
    with c2:
        if st.button("❌ Отклонить", key=f"rej_{ex['id']}"):
            api.reject_uncertain(ex["id"])
            st.rerun()
    st.divider()

if not examples:
    st.info("Очередь пуста.")

st.subheader("Сохранить исправление вручную")
q = st.text_input("Вопрос", key="manual_q")
a = st.text_area("Правильный ответ", key="manual_a", height=80)
if st.button("Сохранить"):
    if q.strip() and a.strip():
        api.save_correction(q, a)
    else:
        st.warning("Заполните оба поля.")
