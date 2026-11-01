# streamlit_app/pages/7_Evaluation.py
import json

import streamlit as st

from streamlit_app.utils import api_client as api
from streamlit_app.utils.auth_guard import require_role

require_role("admin")

st.title("🧪 Оценка качества (EM / F1)")
st.caption("Метрики SQuAD-стиля: Exact Match и token-F1 по вопросам и эталонным ответам.")

uploaded = st.file_uploader("JSON с датасетом (questions/hypotheses/references)", type=["json"])
if uploaded and st.button("Оценить", type="primary"):
    try:
        data = json.loads(uploaded.read().decode("utf-8"))
        questions = data.get("questions", [])
        hypotheses = data.get("hypotheses", data.get("answers", []))
        references = data.get("references", data.get("gold", []))
        if not hypotheses or not references:
            st.warning("В файле должны быть списки hypotheses и references.")
        else:
            import requests
            result = api.api_request(
                "POST", "/query/batch", json={"questions": questions},
            ) if questions else None
            # Локальный подсчёт EM/F1 по готовым ответам:
            from backend.eval.runner import Evaluator
            ev = Evaluator()
            metrics = ev.evaluate(hypotheses, references)
            c1, c2, c3 = st.columns(3)
            c1.metric("Exact Match", f"{metrics['exact_match']:.1f}%")
            c2.metric("F1", f"{metrics['f1']:.1f}%")
            c3.metric("Пар", metrics["n"])
    except json.JSONDecodeError:
        st.error("Некорректный JSON")

st.divider()
st.subheader("Оценка через сервис (эталон vs системные ответы)")
st.caption("Пайплайн: задать вопросы системе через /query/batch, затем посчитать метрики.")

raw = st.text_area("Вопросы (по одному на строку)", key="eval_questions", height=120)
gold = st.text_area("Эталонные ответы (по одному на строку, в том же порядке)", key="eval_gold", height=120)

if st.button("Прогнать оценку"):
    questions = [q.strip() for q in raw.splitlines() if q.strip()]
    references = [g.strip() for g in gold.splitlines() if g.strip()]
    if not questions or len(questions) != len(references):
        st.warning("Списки должны быть непустыми и одинаковой длины.")
    else:
        batch = api.query_batch(questions)
        if batch:
            from backend.eval.runner import Evaluator
            metrics = Evaluator().evaluate(batch["answers"], references)
            c1, c2, c3 = st.columns(3)
            c1.metric("Exact Match", f"{metrics['exact_match']:.1f}%")
            c2.metric("F1", f"{metrics['f1']:.1f}%")
            c3.metric("Пар", metrics["n"])
            with st.expander("Ответы системы"):
                for q, a, g in zip(questions, batch["answers"], references):
                    st.markdown(f"**Q:** {q}")
                    st.markdown(f"- система: {a}")
                    st.markdown(f"- эталон: {g}")
                    st.divider()
