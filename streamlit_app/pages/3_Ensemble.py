# streamlit_app/pages/3_Ensemble.py
import streamlit as st

from streamlit_app.utils import api_client as api
from streamlit_app.utils.auth_guard import require_login
from streamlit_app.utils.subjects import SUBJECTS

require_login()

st.title("⚖️ Ансамбль моделей")
st.caption("Несколько моделей отвечают на вопрос; критик выбирает лучший вариант.")

question = st.text_area("Вопрос", height=100, key="ens_question")
subject = st.selectbox("Предмет", list(SUBJECTS.keys()), format_func=lambda x: SUBJECTS[x], key="ens_subject")

if st.button("Спросить ансамблем", type="primary"):
    if not question.strip():
        st.warning("Введите вопрос.")
    else:
        st.session_state["ens_result"] = api.query_ensemble(question, subject=subject)

result = st.session_state.get("ens_result")
if result:
    st.success(result["best_answer"])
    if result.get("confidence") is not None:
        st.progress(result["confidence"], text=f"Уверенность: {result['confidence']:.0%}")
    st.caption(f"best model: {result.get('best_model', 'N/A')}")

    st.subheader("Варианты моделей")
    for variant in result["variants"]:
        with st.expander(
            f"{variant['model']} — score {variant.get('critic_score', 'N/A')}"
        ):
            st.write(variant["answer"])
            if variant.get("critic_reasoning"):
                st.caption(variant["critic_reasoning"])
