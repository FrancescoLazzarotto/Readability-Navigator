import streamlit as st

def section_title(text):
    st.markdown(f"### {text}")

def subsection(text):
    st.markdown(f"**{text}**")

def divider():
    st.markdown("---")

def page_header(title, subtitle=None):
    st.title(title)
    if subtitle:
        st.write(subtitle)
