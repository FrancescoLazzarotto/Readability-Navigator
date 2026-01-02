import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from components.layout import page_header, section_title, subsection, divider
from components.sidebar import render_sidebar

render_sidebar()

# Header
page_header("Dataset e Preprocessing", "Esplorazione e preparazione dei dati")

divider()

# Carica il dataset
@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("data/processed/onestop_nltk_features.csv")
    except:
        return None

df = load_dataset()

# Dataset Overview
section_title(" Panoramica del Dataset")

if df is not None:
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric(" Numero di Documenti", len(df))
    with col2:
        st.metric(" Numero di Colonne", len(df.columns))
    with col3:
        st.metric(" Valori Mancanti", df.isnull().sum().sum())
    
 
 
    divider()
    
    # Flesch Score Distribution
    section_title(" Distribuzione dei Flesch Score")
    
    if 'flesch_score' in df.columns:
        fig = px.histogram(
            x=df['flesch_score'],
            nbins=30,
            labels={'x': 'Flesch Reading Ease Score', 'y': 'Numero di Documenti'},
            title='Distribuzione della Leggibilità',
            color_discrete_sequence=['#4ECDC4']
        )
        fig.add_vline(
            x=df['flesch_score'].mean(),
            line_dash="dash",
            line_color="red",
            annotation_text=f"Media: {df['flesch_score'].mean():.2f}",
            annotation_position="top right"
        )
        st.plotly_chart(fig, use_container_width=True)
        
        # Statistiche Flesch
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric(" Media", f"{df['flesch_score'].mean():.2f}")
        with col2:
            st.metric(" Mediana", f"{df['flesch_score'].median():.2f}")
        with col3:
            st.metric(" Min", f"{df['flesch_score'].min():.2f}")
        with col4:
            st.metric(" Max", f"{df['flesch_score'].max():.2f}")

    
    divider()
    
    
    
    # Preprocessing Steps
    section_title(" Passaggi di Preprocessing")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        ### 1️⃣ Tokenizzazione
        - Suddivisione del testo in token (parole)
        - Rimozione della punteggiatura
        - Conversione a minuscole
        
        ### 2️⃣ Normalizzazione
        - Rimozione di stop words
        - Lemmatizzazione (NLTK)
        - Stemming (se necessario)
        """)
    
    with col2:
        st.markdown("""
        ### 3️⃣ Estrazione Features
        - Numero di parole per documento
        - Numero di frasi
        - Lunghezza media delle parole
        - Calcolo del Flesch Score
        
        ### 4️⃣ Creazione Dataset Processato
        - Salvataggio in CSV
        - Generazione embedding vettoriali
        - Preparazione per il modello
        """)
    
    divider()
    # Anteprima Dati
    section_title(" Anteprima dei Dati")
    
    st.write("Primi 5 documenti del dataset:")
    if 'testo' in df.columns:
        display_cols = [col for col in df.columns if col != 'testo']
        st.dataframe(df[display_cols].head(), use_container_width=True)
    else:
        st.dataframe(df.head(), use_container_width=True)
    
    divider()
    # Tecniche Utilizzate
    

else:
    st.error(" Non è stato possibile caricare il dataset")
    st.info("Assicurarsi che il file sia presente in: `data/processed/onestop_nltk_features.csv`")
