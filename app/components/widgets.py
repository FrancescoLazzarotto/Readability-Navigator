import streamlit as st
import numpy as np

def user_form():
    st.markdown("#### Configura Utente")

    with st.form("user_form"):
        user_id = st.number_input("User ID", min_value=1, value=1)

        target = st.slider(
            "Target Readability",
            min_value=0,
            max_value=100,
            value=60
        )

        topic_vector = st.text_area(
            "Topic Vector (opzionale)",
            help="Se vuoto, verrà generato un vettore casuale."
        )

        submitted = st.form_submit_button("Genera Raccomandazioni")

    if topic_vector.strip() == "":
        
        topic_vec = list(np.random.rand(384))
    else:
        try:
            topic_vec = [float(x) for x in topic_vector.split(",")]
        except:
            st.error("Formato non valido per topic_vector.")
            topic_vec = None

    user = {
        "user_id": user_id,
        "target_readability": target,
        "topic_vector": topic_vec,
        "history": []
    }

    return submitted, user
