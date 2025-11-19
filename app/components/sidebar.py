import streamlit as st

def render_sidebar():
    st.sidebar.title("Readability Navigator")

    st.sidebar.write("Impostazioni generali")

    theme = st.sidebar.selectbox(
        "Tema",
        ["Chiaro", "Scuro"],
        index=0
    )

    st.sidebar.markdown("---")
    st.sidebar.write("Navigazione tra le pagine")
    st.sidebar.write(
        "Usa il menu di sinistra di Streamlit per navigare tra le sezioni."
    )

    return {"theme": theme}
