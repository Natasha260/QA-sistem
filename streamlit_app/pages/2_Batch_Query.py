# streamlit_app/pages/2_Batch_Query.py
import streamlit as st

from streamlit_app.utils import api_client as api
from streamlit_app.utils.auth_guard import require_login
from streamlit_app.utils.subjects import SUBJECTS

require_login()

st.title("📦 Пакетный запрос")
st.caption("Ответы на список вопросов за один запрос к API.")

raw = st.text_area(
    "Вопросы (по одному на строку)", height=160, key="batch_questions",
)
c1, c2, c3 = st.columns(3)
with c1:
    subject = st.selectbox("Предмет", list(SUBJECTS.keys()), format_func=lambda x: SUBJECTS[x], key="batch_subject")
with c2:
    top_k = st.slider("top_k", 1, 10, 5, key="batch_top_k")
with c3:
    eval_mode = st.selectbox(
        "Режим (ablation)",
        ["", "baseline", "terminology", "chunks_bm25", "chunks_hybrid",
         "agents_only", "chunks_hybrid_critic", "chunks_hybrid_editor", "full"],
        format_func=lambda x: x or "(из настроек)",
        key="batch_mode",
    )

if st.button("Спросить пакетом", type="primary"):
    questions = [q.strip() for q in raw.splitlines() if q.strip()]
    if not questions:
        st.warning("Введите хотя бы один вопрос.")
    else:
        st.session_state["batch_result"] = api.query_batch(
            questions, top_k=top_k, subject=subject,
            eval_mode=eval_mode or None,
        )

result = st.session_state.get("batch_result")
if result:
    for answer, confidence in zip(result["answers"], result["confidences"]):
        st.markdown(f"**Ответ:** {answer}")
        if confidence is not None:
            st.progress(confidence, text=f"Уверенность: {confidence:.0%}")
        st.divider()
    details = result.get("details")
    if details:
        with st.expander("📋 Детали"):
            st.json(details)
