import streamlit as st
import pandas as pd
import numpy as np

def show_embedding_preview(embedding, max_values=10):
    st.markdown("#### Esempio di embedding")
    if isinstance(embedding, np.ndarray):
        preview = embedding[:max_values]
    else:
        preview = embedding[0][:max_values]

    st.write(preview)

def show_document_details(doc_row):
    st.markdown("#### Dettagli documento")

    title = doc_row.get("id", "Documento")
    st.subheader(title)

    if "testo" in doc_row:
        st.markdown("**Testo:**")
        st.write(doc_row["testo"])

    cols = [c for c in doc_row.keys() if c not in ["testo"]]

    st.markdown("**Caratteristiche principali:**")
    st.dataframe(pd.DataFrame(doc_row[cols].items(), columns=["Feature", "Valore"]))
