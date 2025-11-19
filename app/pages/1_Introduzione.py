import streamlit as st
import plotly.graph_objects as go
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from components.layout import page_header, section_title, subsection, divider
from components.sidebar import render_sidebar

render_sidebar()

# Header
page_header("📚 Readability Navigator", "Sistema Intelligente di Raccomandazioni Personalizzate")

divider()

# Descrizione del Progetto
section_title("📖 Cos'è Readability Navigator?")
st.write("""
Readability Navigator è un sistema avanzato di raccomandazioni personalizzate che aiuta i lettori 
a scoprire contenuti adatti al loro livello di comprensione. Utilizzando algoritmi di machine learning 
e analisi del testo, il sistema suggerisce documenti che corrispondono perfettamente alle preferenze 
e alle capacità di lettura di ogni utente.

Il progetto combina tecniche di **processamento del linguaggio naturale (NLP)**, **embedding di testo** 
e **algoritmi di ranking personalizzato** per offrire un'esperienza di lettura ottimale.
""")

divider()

# Pipeline del Progetto
section_title("🔄 Pipeline del Progetto")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.write("**1️⃣ Ingestion**")
    st.write("Caricamento e raccolta dei dati testuali")

with col2:
    st.write("**2️⃣ Preprocessing**")
    st.write("Pulizia, tokenizzazione e normalizzazione")

with col3:
    st.write("**3️⃣ Embedding**")
    st.write("Conversione in vettori semantici")

with col4:
    st.write("**4️⃣ Ranking**")
    st.write("Calcolo degli score personalizzati")

divider()


# Metriche e Statistiche
section_title("📊 Caratteristiche Chiave")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric("📁 Documenti Dataset", "567", delta="Corpus OneStop")
    st.caption("Testi con diversi livelli di leggibilità")

with col2:
    st.metric("🧠 Dimensione Embedding", "384", delta="Vettori Semantici")
    st.caption("Rappresentazione dello spazio semantico")

with col3:
    st.metric("⚙️ Algoritmi", "3+", delta="Similarity & Ranking")
    st.caption("Flesch, Cosine, Recommendation Engine")

divider()

# Caratteristiche Principali
section_title("✨ Funzionalità Principali")

tab1, tab2, tab3 = st.tabs(["Leggibilità", "Profilo Utente", "Raccomandazioni"])

with tab1:
    st.write("""
    **📈 Flesch Readability Score**: Analizza il livello di difficoltà
    
    - **Easy (Facile)**: 60-100
    - **Intermediate (Medio)**: 40-60  
    - **Hard (Difficile)**: 0-40
    """)

with tab2:
    st.write("""
    **👤 Personalizzazione Intelligente**:
    
    - Target di leggibilità desiderato
    - Vettore di interessi tematici
    - Cronologia documenti consultati
    - Preferenze di lettura personalizzate
    """)

with tab3:
    st.write("""
    **🎯 Sistema di Ranking Avanzato**:
    
    - **Somiglianza Tematica**: Cosine similarity
    - **Gap di Leggibilità**: Differenza dal target
    - **Penalità Dinamica**: Bonus/malus intelligente
    """)

divider()

# Tecnologie Utilizzate
section_title("🛠️ Stack Tecnologico")

col1, col2 = st.columns(2)

with col1:
    st.write("""
    **Backend:**
    - Python 3.10
    - Pandas & NumPy
    - Scikit-learn
    - NLTK
    """)

with col2:
    st.write("""
    **Frontend:**
    - Streamlit
    """)

divider()



st.divider()

# How It Works
st.markdown("## How It Works")
with st.expander("Click to learn more about the recommendation algorithm"):
    st.markdown("""
    ### Scoring Algorithm
    
    The recommendation score is calculated as:
    
    ```
    Score = η × Similarity - ζ × Penalized_Gap
    ```
    
    Where:
    - **η (eta)**: Weight for theme similarity
    - **ζ (zeta)**: Weight for readability gap penalty
    - **Similarity**: Cosine similarity between user topic vector and document embedding
    - **Penalized_Gap**: Readability gap (|target - document_score|) multiplied by penalty factor
    
    ### Penalty Function
    
    If document readability > target readability:
    ```
    Penalty = 1 + α (penalize difficult documents)
    ```
    
    Otherwise:
    ```
    Penalty = 1 (no additional penalty)
    ```
    
    This encourages recommendations of documents at or slightly below the target level.
    """)

