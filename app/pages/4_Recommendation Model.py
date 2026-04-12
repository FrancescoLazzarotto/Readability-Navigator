import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

from components.layout import page_header, section_title, subsection, divider
from components.sidebar import render_sidebar

render_sidebar()

# Header
page_header("Recommendation Model", "Sistema di ranking e raccomandazioni personalizzate")

divider()

# Model Overview
section_title(" Panoramica del Modello")

st.write("""
Il Recommendation Engine combina molteplici fattori per calcolare uno score personalizzato 
per ogni documento, tenendo conto delle preferenze dell'utente e delle caratteristiche 
del documento stesso.
""")

divider()

# Scoring Formula
section_title(" Formula di Scoring")

st.markdown("""
### La raccomandazione avviene tramite:

$$\\text{Score} = \\eta \\cdot \\text{Similarity} - \\zeta \\cdot \\text{Penalized Gap}$$

Dove:
- **η (eta)**: Peso della somiglianza tematica (0-1)
- **z (zeta)**: Peso della penalità di leggibilità (0-1)
- **Similarity**: Somiglianza coseno tra profilo utente e embedding documento
- **Penalized Gap**: Gap di leggibilità moltiplicato per fattore di penalità

**Gap di Leggibilità**: Differenza tra target dell'utente e leggibilità del documento. Gap = 0 → documento perfetto.

**Penalità Dinamica**: Se il documento è troppo difficile (> target), la penalità aumenta (1 + α). Se è facile, resta 1. 
""")

divider()

section_title(" Pipeline di Ranking")

col1, col2, col3, col4, col5 = st.columns(5)

with col1:
    st.markdown("**1. Input Utente**\n- Profilo utente\n- Target leggibilità\n- Vettore tematico")

with col2:
    st.markdown("**2. Filtraggio**\n- Catalogo disponibile\n- Esclusione cronologia\n- Filtro leggibilità")

with col3:
    st.markdown("**3. Scoring**\n- Similarità tematica\n- Gap leggibilità\n- Penalità dinamica")

with col4:
    st.markdown("**4. Ranking**\n- Ordinamento score\n- Selezione Top-K\n- Recupero documenti")

with col5:
    st.markdown("**5. Output**\n- Raccomandazioni\n- Score finali\n- Testo completo")



divider()

# Algoritmo Completo
section_title(" Algoritmo Completo")

st.code("""
def recommender(user, doc_id):
    # estrai parametri
    eta = config['eta']
    zeta = config['zeta']
    alpha = config['alpha']
    
    # calcola leggibilità
    flesch = get_flesch(doc_id)
    
    # calcola similarità tematica
    sim = theme_similarity(user, doc_id)
    
    # calcola gap di leggibilità
    gap, target, readability = gap_readability(user, flesch)
    
    # applica penalità dinamica
    penalty_score = penalty(target, readability, alpha)
    gap_penalized = gap * penalty_score
    
    # calcola score finale
    score = eta * sim - zeta * gap_penalized
    
    return score
""", language="python")

# Metriche di Valutazione
section_title(" Metriche di Valutazione")

st.write("""
Il sistema viene valutato con **NDCG@K** (Normalized Discounted Cumulative Gain),
una metrica che misura quanto bene il modello raccomanda elementi appropriati.
""")

st.markdown("""
### NDCG@K - Cosa è?

**L'idea**: Se consiglio un documento facile a chi ne legge uno di difficili, ho sbagliato.
NDCG misura quanto spesso consiglio l'item giusto.

**Formula**:
- Guardo i Top-5 elementi consigliati
- Controllo se hanno la leggibilità giusta (vicino al target dell'utente)
- Do un punteggio da 0 a 1
- **1.0** = sempre giusto | **0.5** = 50% giusti | **0.0** = sempre sbagliato

**Esempio**:
- Target utente = 60 (medio)
- Consiglio Top-5: [60, 62, 58, 75, 55]
- Sono 4/5 vicini al target → NDCG ≈ 0.80 (80%)
""")

divider()

st.markdown("### Risultati Attuali")

col1, col2 = st.columns(2)

with col1:
    st.metric("NDCG Medio", "0.742", "74.2%")
    st.write("""
    Su 2 utenti test:
    - User 1: NDCG = 0.75
    - User 2: NDCG = 0.73
    
    **Interpretazione**: In media, il sistema consiglia 
    la leggibilità giusta il 74% delle volte.
    """)

with col2:
    st.write("""
    ### Cosa Significa 0.74?
    
     - **Buono**: Non è casuale (0.5)
     - **Affidabile**: La maggior parte è corretta
     - **Non Perfetto**: C'è margine di miglioramento
    
    **Paragone**:
    - 0.90+ = Eccellente
    - 0.70-0.85 = Buono 
    - 0.50-0.70 = Accettabile
    - <0.50 = Pessimo
    """)

divider()

st.markdown("---")
st.markdown("Nota: I parametri possono essere regolati nel file di configurazione `conf/project.yaml`")
