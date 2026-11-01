# streamlit_app/pages/5_Terminology.py
import streamlit as st

from streamlit_app.utils import api_client as api
from streamlit_app.utils.auth_guard import require_role
from streamlit_app.utils.subjects import SUBJECTS

require_role("admin")

st.title("📖 Терминология")
st.caption("Термины предметных областей: попадают в промпт, если встречаются в вопросе.")

st.subheader("Добавить термин")
c1, c2, c3 = st.columns(3)
with c1:
    term = st.text_input("Термин", key="term")
with c2:
    subject = st.selectbox("Предмет", list(SUBJECTS.keys()), format_func=lambda x: SUBJECTS[x], key="term_subject")
with c3:
    definition = st.text_input("Определение", key="definition")

if st.button("Добавить", type="primary"):
    if term and definition:
        api.add_terminology(term, definition, subject)
    else:
        st.warning("Заполните термин и определение.")

st.divider()
st.subheader("Импорт CSV (term,definition,subject)")
csv_file = st.file_uploader("CSV", type=["csv"], key="term_csv")
if csv_file and st.button("Импортировать CSV"):
    result = api.import_terminology_csv(csv_file)
    if result:
        st.success(f"Добавлено: {result['added']}, пропущено: {result['skipped']}")

st.divider()
st.subheader("Список терминов")

data = api.list_terminology(limit=300)
if data:
    for entry in data["entries"]:
        col1, col2, col3 = st.columns([3, 5, 1])
        with col1:
            st.write(f"**{entry['term']}** ({entry['subject']})")
        with col2:
            st.write(entry["definition"])
        with col3:
            if st.button("🗑", key=f"del_term_{entry['id']}"):
                api.delete_terminology(entry["id"])
                st.rerun()
    st.caption(f"Всего: {data['total']}")
else:
    st.info("Терминов пока нет.")
