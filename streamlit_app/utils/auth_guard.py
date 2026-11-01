# streamlit_app/utils/auth_guard.py
import streamlit as st


def current_user() -> dict:
    return st.session_state.get("user") or {}


def require_login() -> dict:
    user = current_user()
    if not user:
        st.error("🔒 Please log in to access this page.")
        st.page_link("app.py", label="→ Go to login", icon="🚀")
        st.stop()
    return user


def require_role(*allowed_roles: str) -> dict:
    user = require_login()
    role = user.get("role", "user")
    if role not in allowed_roles:
        st.error(
            f"🔒 This page requires one of the roles: "
            f"{', '.join(allowed_roles)}."
        )
        st.page_link("app.py", label="→ Go to dashboard", icon="🚀")
        st.stop()
    return user
