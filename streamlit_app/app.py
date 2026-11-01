# streamlit_app/app.py
import re
import streamlit as st

from streamlit_app.utils import api_client as api

st.set_page_config(
    page_title="QA-система по учебникам",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.session_state.setdefault("access_token", None)
st.session_state.setdefault("user", None)


def _validate_password(password: str) -> str | None:
    """Mirror of the server-side password policy for instant feedback."""
    if len(password) < 8:
        return "Password must be at least 8 characters long."
    if len(password.encode("utf-8")) > 72:
        return "Password is too long (max 72 bytes)."
    if not re.search(r"\d", password):
        return "Password must contain at least one digit."
    if not re.search(r"[A-ZА-Я]", password):
        return "Password must contain at least one uppercase letter."
    return None


with st.sidebar:
    st.title("🎓 QA-система")
    st.caption("Ответы на вопросы по университетским учебникам")

    if st.session_state.user:
        user = st.session_state.user
        st.success(f"👤 {user.get('username', 'user')} ({user.get('role', 'user')})")
        if st.button("🚪 Logout", use_container_width=True):
            api.logout()
            st.rerun()
    else:
        tab_login, tab_register = st.tabs(["Login", "Register"])

        with tab_login:
            email = st.text_input("Email", key="login_email")
            password = st.text_input("Password", type="password", key="login_password")
            if st.button("Login", use_container_width=True):
                if api.login(email, password):
                    st.rerun()

        with tab_register:
            reg_email = st.text_input("Email", key="reg_email")
            reg_username = st.text_input("Username", key="reg_username")
            reg_full_name = st.text_input("Full name (optional)", key="reg_full_name")
            reg_password = st.text_input("Password", type="password", key="reg_password")
            error = _validate_password(reg_password) if reg_password else None
            if error:
                st.warning(error)
            if st.button("Register", use_container_width=True, disabled=bool(error)):
                if api.register(reg_email, reg_username, reg_password, reg_full_name):
                    st.rerun()

if not st.session_state.user:
    st.title("🎓 QA-система по университетским учебникам")
    st.markdown(
        "Вопросно-ответная система с гибридным поиском по учебникам "
        "(BM25 + Qdrant + RRF), мультиагентной проверкой ответа "
        "(Supervisor + Validator + Critic + Editor) и оценкой уверенности."
    )
    st.info("Войдите в sidebar, чтобы начать.")
    st.stop()

role = st.session_state.user.get("role", "user")
is_admin = role == "admin"
is_editor = role in ("admin", "editor")

st.title("🎓 QA-система — Dashboard")

st.subheader("Навигация")
cols = st.columns(2)
with cols[0]:
    st.page_link("pages/1_Query.py", label="❓ Вопрос-ответ", icon="❓")
    st.page_link("pages/2_Batch_Query.py", label="📦 Пакетный запрос", icon="📦")
    st.page_link("pages/3_Ensemble.py", label="⚖️ Ансамбль", icon="⚖️")
    if is_editor:
        st.page_link("pages/6_HITL.py", label="🧑‍🏫 Human-in-the-Loop", icon="🧑‍🏫")
with cols[1]:
    if is_admin:
        st.page_link("pages/4_Documents.py", label="📚 Учебники", icon="📚")
        st.page_link("pages/5_Terminology.py", label="📖 Терминология", icon="📖")
        st.page_link("pages/7_Evaluation.py", label="🧪 Оценка (EM/F1)", icon="🧪")
        st.page_link("pages/8_System_Health.py", label="🩺 Состояние", icon="🩺")

st.divider()

st.subheader("⚡ Быстрый вопрос")
quick_question = st.text_area(
    "Вопрос", height=80, key="quick_question",
    placeholder="Например: сформулируйте второй закон Ньютона",
)

quick_key = quick_question
if st.session_state.get("quick_key") != quick_key:
    st.session_state["quick_result"] = None
    st.session_state["quick_key"] = quick_key

if st.button("Ответить", type="primary"):
    if not quick_question.strip():
        st.warning("Введите вопрос.")
    else:
        result = api.query(quick_question)
        st.session_state["quick_result"] = result

quick_result = st.session_state.get("quick_result")
if quick_result:
    st.success(quick_result["answer"])
    if quick_result.get("confidence") is not None:
        st.progress(
            quick_result["confidence"],
            text=f"Уверенность: {quick_result['confidence']:.0%}",
        )
    st.caption(
        f"model: {quick_result.get('model_used', 'N/A')} | "
        f"critic: {quick_result.get('critic_score') if quick_result.get('critic_score') is not None else 'N/A'}"
        + (" | ⚡ из кэша" if quick_result.get("cached") else "")
    )
