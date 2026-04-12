import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from sklearn.decomposition import PCA
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from components.layout import page_header, section_title, subsection, divider
from components.sidebar import render_sidebar

render_sidebar()

page_header("Dataset, Preprocessing e Embedding", "Dal testo grezzo alla rappresentazione vettoriale")

divider()

@st.cache_data
def load_dataset():
    try:
        return pd.read_csv("data/processed/onestop_nltk_features.csv")
    except:
        return None

@st.cache_data
def load_embeddings():
    try:
        import pickle
        with open("src/features/doc_embedding.pickle", "rb") as f:
            return pickle.load(f)
    except:
        return None

df = load_dataset()
embeddings = load_embeddings()

if df is not None:
    # ============================================================
    # SEZIONE 1: DATASET
    # ============================================================
    section_title("Il Dataset - OneStop English")
    
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Articoli", len(df))
    with col2:
        st.metric("Lunghezza Media", f"{df['num_words'].mean():.0f} parole")
    with col3:
        st.metric("Lingua", "Inglese")
    
    st.markdown("""
    Questo dataset contiene articoli giornalistici in inglese provenienti da **OneStop English Corpus**, 
    una raccolta di testi appositamente semplificati per studenti di lingue.
    
    Gli articoli sono disponibili in 3 livelli di difficoltà:
    - **Elementary**: Testi semplici
    - **Intermediate**: Testi moderati
    - **Advanced**: Testi complessi
    
    https://github.com/nishkalavallabhi/OneStopEnglishCorpus\n
    https://www.kaggle.com/datasets/maunish/onestopenglishcorpus
    """)
    
    divider()
    
    # ============================================================
    # SEZIONE 2: PREPROCESSING
    # ============================================================
    section_title("Preprocessing")
    
    st.markdown("""
    Il preprocessing trasforma il testo grezzo in dati strutturati che il sistema può analizzare.
    """)
    
    divider()
    
    st.subheader("Fase 1: Pulizia del Testo")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Cosa viene fatto:**
        
        - Rimozione punteggiatura
        - Rimozione caratteri speciali
        - Conversione a minuscole
        """)
    
    with col2:
        st.markdown("""
        **Esempio:**
        
        Input: *"The Brain's Study..."*
        
        Output: *"the brains study"*
        """)
    
    divider()
    
    st.subheader("Fase 2: Normalizzazione")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Cosa viene fatto:**
        
        - Suddivisione in frasi
        - Suddivisione in parole
        - Conteggio sillabe per parola
        - Rimozione stop words (the, a, an, is...)
        """)
    
    with col2:
        st.markdown("""
        **Utilità:**
        
        Prepara il testo per il calcolo del Flesch Score e per l'estrazione di metriche significative che potrebbero essere usato per ulteriori implementazioni.
        """)
    
    divider()
    
    st.subheader("Fase 3: Calcolo Flesch Reading Ease Score")
    
    st.markdown("""
    Il Flesch Score è una metrica che misura quanto è leggibile un testo, basata sulla lunghezza delle parole e delle frasi.
    È uno dei metodi più classici e utilizzati per valutare la difficoltà di lettura di un documento.
    """)
    
    st.markdown("""
    **Formula di Flesch:**
    
    $$\\text{Flesch} = 206.835 - 1.015 \\times \\frac{\\text{parole}}{\\text{frasi}} - 84.6 \\times \\frac{\\text{sillabe}}{\\text{parole}}$$
    
    - **Primo termine**: Penalizza frasi lunghe (poche frasi rispetto alle parole)
    - **Secondo termine**: Penalizza parole lunghe (molte sillabe per parola)
    - **Intervallo**: 0-100 (100 = massima semplicità, 0 = massima difficoltà)
    """)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("""
        **Testi Semplici** (Score ALTO)
        - Parole brevi
        - Frasi corte
        - Concetti elementari
        
        Esempio: "Il gatto è nero."
        """)
    
    with col2:
        st.markdown("""
        **Testi Complessi** (Score BASSO)
        - Parole lunghe
        - Frasi complesse
        - Concetti astratti
        
        Esempio: "L'implementazione di algoritmi di deep learning per l'ottimizzazione..."
        """)
    
    divider()
    
    st.subheader("Distribuzione della Leggibilità nel Dataset")
    
    if 'flesch_score' in df.columns:
        fig_flesch = px.histogram(
            x=df['flesch_score'],
            nbins=30,
            color_discrete_sequence=['#4ECDC4'],
            labels={'x': 'Flesch Score', 'y': 'Numero di Articoli'},
        )
        fig_flesch.add_vline(
            x=df['flesch_score'].mean(),
            line_dash="dash",
            line_color="#FF6B6B",
            annotation_text=f"Media: {df['flesch_score'].mean():.1f}",
            annotation_position="top right"
        )
        fig_flesch.update_layout(
            title="",
            height=400,
            showlegend=False
        )
        st.plotly_chart(fig_flesch, use_container_width=True)
        
        st.markdown("**Interpretazione della scala Flesch:**")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("""
            **90-100**: Molto facile
            *Elementare*
            
            **70-90**: Facile
            *Conversazionale*
            """)
        with col2:
            st.markdown("""
            **50-70**: Moderato
            *Giornalistico*
            
            **30-50**: Difficile
            *Accademico*
            """)
        with col3:
            st.markdown("""
            **0-30**: Molto difficile
            *Specialistico*
            
            **Target**: [20-90]
            *Intervallo del progetto*
            """)
    
    divider()
    
    # ============================================================
    # SEZIONE 3: EMBEDDING
    # ============================================================
    section_title("Embedding Vettoriali - Rappresentare il Significato")
    
    st.markdown("""
    Dopo il preprocessing, ogni articolo viene trasformato in un **vettore di 384 numeri** . 
    
    """)
    
    divider()
        
    st.subheader(" Cos'è un Embedding?")
        
    st.markdown("""
        Un **embedding** è una rappresentazione numerica di un testo. Anziché memorizzare il testo come parole,
        lo trasformiamo in numeri che il computer può elaborare e confrontare.
        
        **Cos'è BERT?**
        
        BERT (Bidirectional Encoder Representations from Transformers) è una rete neurale addestrata su 
        miliardi di testi. Ha imparato a comprendere il significato delle parole guardando il contesto 
        (le parole prima e dopo). È "bidirezionale" perché analizza il testo sia da sinistra a destra 
        che da destra a sinistra.
        
        **Cos'è Sentence-BERT (SBERT)?**
        
        Sentence-BERT è una versione modificata di BERT specializzata nel rappresentare **interi articoli**, 
        non solo parole singole. Mentre BERT dà un numero per ogni parola, SBERT dà un numero per l'intero articolo.
        
        **Il Risultato: 384 Numeri**
        
        Quando SBERT elabora un articolo, produce 384 numeri. Questi numeri non hanno un significato diretto 
        (non rappresentano "tema", "lunghezza", ecc.), ma insieme catturano tutto il significato dell'articolo 
        in una forma che il computer può elaborare rapidamente.
        
        """)
    
    divider()
    
    st.subheader("Come Funziona")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Input:**
        
        Articolo pulito e normalizzato
        """)
    
    with col2:
        st.markdown("""
        **Processo:**
        
        Rete neurale SBERT
        """)
    
    with col3:
        st.markdown("""
        **Output:**
        
        384 numeri che rappresentano il significato
        """)
    
    divider()
    
      
    st.subheader("Il Processo di Embedding")
        
    st.markdown("""
        Un embedding è il risultato di un **processo di trasformazione** del testo:
        """)
    divider()
        
        # Diagramma minimalista e professionale
    col1, col2, col3, col4, col5, col6, col7 = st.columns([2, 0.5, 2, 0.5, 2, 0.5, 2])
        
    with col1:
            st.info("**Testo Grezzo**\n\nArticolo originale")
        
    with col2:
            st.write("**→**")
        
    with col3:
            st.success("**Normalizzazione**\n\nPulizia e preparazione")
        
    with col4:
            st.write("**→**")
        
    with col5:
            st.warning("**Rete Neurale**\n\nSBERT processa")
        
    with col6:
            st.write("**→**")
        
    with col7:
            st.error("**384 Numeri**\n\nEmbedding finale")
        
    divider()
        

        
       
  

else:
    st.error("Dataset non disponibile")
    st.info("Verifica che il file esista in: data/processed/onestop_nltk_features.csv")

