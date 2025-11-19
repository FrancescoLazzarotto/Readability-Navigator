import streamlit as st
import numpy as np
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from components.sidebar import render_sidebar
from components.layout import page_header, divider
from components.widgets import user_form
from main import main  

render_sidebar()

page_header("Readability Navigator", "Generatore di raccomandazioni")

submitted = user_form()

divider()
user = {
        "user_id": 10,
        "target_readability": 60,
        "topic_vector": list(np.random.rand(384)),
        "history": []
    }
if submitted:
    df = main(user)
    st.write(df)
    #st.dataframe(df)
    