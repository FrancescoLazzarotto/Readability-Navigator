import streamlit as st
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from components.sidebar import render_sidebar
from components.layout import page_header, divider
from components.widgets import user_form
from main import main  

render_sidebar()

page_header("Readability Navigator", "Generatore di raccomandazioni")

submitted, user = user_form()

divider()

if submitted:
    df = main()
    st.dataframe(df)