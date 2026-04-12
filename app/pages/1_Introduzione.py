import streamlit as st
import plotly.graph_objects as go
import pandas as pd
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from components.layout import page_header, section_title, subsection, divider
from components.sidebar import render_sidebar

render_sidebar()

# Header
page_header("Readability Navigator", "Sistema di Raccomandazioni Personalizzate")

divider()

# Descrizione del Progetto
section_title(" Cos'è Readability Navigator?")
st.write("""
Readability Navigator è un sistema di raccomandazioni content-based che aiuta i lettori 
a scoprire contenuti adatti al loro livello di comprensione. Utilizzando algoritmi di machine learning 
e analisi del testo, il sistema suggerisce documenti che corrispondono alle preferenze 
e alle capacità di lettura corrispondeti al profilo utente.

Il progetto combina tecniche di **processamento del linguaggio naturale (NLP)**, **embedding di testo** 
e **algoritmi di ranking personalizzato** per cercare di offrire un'esperienza di lettura ottimale.
""")

divider()

# Pipeline del Progetto
section_title("Sviluppo del Progetto")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.write("**1️⃣ Ingestion**")
    st.write("Ricerca/Estrazione Dataset, caricamento dati testuali")

with col2:
    st.write("**2️⃣ Preprocessing**")
    st.write("Pulizia, tokenizzazione, normalizzazione, calcolo del Flesch Score")

with col3:
    st.write("**3️⃣ Embedding**")
    st.write("Conversione in vettori semantici dei documenti")
    
with col4:
    st.write("**4️⃣ Ranking**")
    st.write("Calcolo degli score di ranking sulla base del profilo utente")

divider()


# Metriche e Statistiche
section_title(" Caratteristiche Chiave")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(" Documenti Dataset", "567", delta="Corpus OneStop")
    st.caption("Testi con diversi livelli di leggibilità")

with col2:
    st.metric("Dimensione Embedding", "384", delta="Vettori Semantici")
    st.caption("Rappresentazione dello spazio semantico")

with col3:
    st.metric(" Algoritmi", "3+", delta="Similarity & Ranking")
    st.caption("Flesch, Cosine Similarity, Recommendation Engine")

divider()

# Tecnologie Utilizzate
section_title("Tecnologie Utilizzate")

tech_data = pd.DataFrame({
    "Tecnologia": ["Python", "Scikit-learn", "Pandas & NumPy", "NLTK", "Sentence-BERT (SBERT)", "Streamlit"],
    "Utilizzo": [
        "Linguaggio principale",
        "Algoritmi di ranking e similarity",
        "Manipolazione e analisi dati",
        "Preprocessing e tokenizzazione",
        "Embedding semantico (384 dimensioni)",
        "Dashboard interattivo"
    ]
})

st.dataframe(tech_data, use_container_width=True, hide_index=True)

divider()


