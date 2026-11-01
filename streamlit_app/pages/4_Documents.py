# streamlit_app/pages/4_Documents.py
import streamlit as st

from streamlit_app.utils import api_client as api
from streamlit_app.utils.auth_guard import require_role

require_role("admin")

st.title("📚 Учебники")
st.caption("Загрузка учебников, чанкинг и индексация для гибридного поиска.")

st.subheader("Загрузить файл (.txt / .md / .csv)")
uploaded = st.file_uploader("Файл учебника", type=["txt", "md", "csv"])
title = st.text_input("Название", key="doc_title", placeholder="Физика. Механика")
description = st.text_input("Описание", key="doc_desc")

if uploaded is not None and st.button("Загрузить", type="primary"):
    result = api.import_document_file(
        uploaded, title=title or None, description=description or None,
    )
    if result:
        st.success(f"Загружено: {result.get('title')} ({result.get('chunks')} чанков)")

st.divider()
st.subheader("Загруженные документы")

data = api.list_documents(limit=200)
if data:
    for doc in data["documents"]:
        col1, col2, col3 = st.columns([4, 2, 1])
        with col1:
            st.write(f"**{doc['title']}**" + (f" — {doc['filename']}" if doc.get("filename") else ""))
            if doc.get("description"):
                st.caption(doc["description"])
        with col2:
            st.write(f"чанков: {doc['chunks']}")
        with col3:
            if st.button("🗑", key=f"del_{doc['id']}"):
                api.delete_document(doc["id"])
                st.rerun()

    with st.expander("📋 JSON"):
        st.json(data)
else:
    st.info("Документов пока нет.")

if st.button("🔄 Обновить"):
    st.rerun()
