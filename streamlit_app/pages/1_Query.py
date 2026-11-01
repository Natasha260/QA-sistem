# streamlit_app/pages/1_Query.py
import streamlit as st

from streamlit_app.utils import api_client as api
from streamlit_app.utils.auth_guard import require_login
from streamlit_app.utils.subjects import SUBJECTS

require_login()

st.title("❓ Вопрос-ответ")
st.caption("Ответ на вопрос по учебникам с гибридным поиском и оценкой уверенности.")

question = st.text_area("Вопрос", height=100, key="single_question")
c1, c2, c3 = st.columns(3)
with c1:
    subject = st.selectbox("Предмет", list(SUBJECTS.keys()), format_func=lambda x: SUBJECTS[x], key="single_subject")
with c2:
    top_k = st.slider("Количество контекста (top_k)", 1, 10, 5)
with c3:
    eval_mode = st.selectbox(
        "Режим (ablation)",
        ["", "baseline", "terminology", "chunks_bm25", "chunks_hybrid",
         "agents_only", "chunks_hybrid_critic", "chunks_hybrid_editor", "full"],
        format_func=lambda x: x or "(из настроек)",
    )

if st.button("Ответить", type="primary"):
    if not question.strip():
        st.warning("Введите вопрос.")
    else:
        st.session_state["single_result"] = api.query(
            question, top_k=top_k, subject=subject,
            eval_mode=eval_mode or None,
        )

result = st.session_state.get("single_result")
if result:
    confidence = result.get("confidence")
    st.success(result["answer"])
    if confidence is not None:
        level = "высокая" if confidence >= 0.7 else "средняя" if confidence >= 0.5 else "низкая"
        st.progress(confidence, text=f"Уверенность: {confidence:.0%} ({level})")
    st.caption(
        f"model: {result.get('model_used', 'N/A')} | "
        f"critic: {result['critic_score'] if result.get('critic_score') is not None else 'N/A'}"
        + (" | ⚡ из кэша" if result.get("cached") else "")
    )

    chunks = result.get("chunks_used") or []
    if chunks:
        st.subheader("📚 Источники")
        for i, chunk in enumerate(chunks, 1):
            with st.expander(f"Чанк {i} — релевантность {chunk.get('score', 0):.3f}"):
                st.write(chunk.get("text", ""))

    reasoning_log = result.get("reasoning_log") or []
    with st.expander("📋 Детали ответа"):
        st.json({k: v for k, v in result.items() if k != "reasoning_log"})
    with st.expander(f"🧠 Лог агентов ({len(reasoning_log)} шагов)"):
        for i, step in enumerate(reasoning_log, 1):
            st.markdown(f"**Шаг {i}: {step.get('agent', step.get('stage', 'step'))}**")
            for field, value in step.items():
                if field in ("agent", "stage"):
                    continue
                if value in (None, [], {}, ""):
                    continue
                st.write(f"- **{field}**: {value}")
            st.divider()
