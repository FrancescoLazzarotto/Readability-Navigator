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
### Equazione Principale

$$\\text{Score} = \\eta \\cdot \\text{Similarity} - \\zeta \\cdot \\text{Penalized Gap}$$

Dove:
- **η (eta)**: Peso della somiglianza tematica (0-1)
- **ζ (zeta)**: Peso della penalità di leggibilità (0-1)
- **Similarity**: Somiglianza coseno tra profilo utente e embedding documento
- **Penalized Gap**: Gap di leggibilità moltiplicato per fattore di penalità
""")

divider()

# Componenti del Modello
section_title(" Componenti del Modello")

tab1, tab2, tab3 = st.tabs(["Somiglianza Tematica", "Gap di Leggibilità", "Penalità Dinamica"])

with tab1:
    st.markdown("""
    ### Cosine Similarity
    
    Misura la somiglianza tra due vettori nel spazio embedding:
    
    $$\\text{Similarity} = \\frac{\\mathbf{u} \\cdot \\mathbf{v}}{\\|\\mathbf{u}\\| \\|\\mathbf{v}\\|}$$
    
    - Intervallo: [-1, 1] (tipicamente [0, 1])
    - 1 = identici, 0 = ortogonali, -1 = opposti
    - Misura il grado di sovrapposizione tematica
    
    **Vantaggi:**
    - Insensibile alla magnitudine dei vettori
    - Efficiente computazionalmente
    - Ottimo per spazi ad alta dimensionalità
    """)

with tab2:
    st.markdown("""
    ### Readability Gap
    
    Misura la differenza tra la leggibilità desiderata e quella effettiva:
    
    $$\\text{Gap} = |\\text{target\\_readability} - \\text{document\\_readability}|$$
    
    - Gap = 0: Leggibilità perfetta
    - Gap > 0: Documento non è al livello target
    
    **Interpretazione:**
    - Gap piccolo: Documento adatto al livello utente 
    - Gap grande: Documento troppo facile o difficile 
    """)

with tab3:
    st.markdown("""
    ### Penalty Function
    
    Penalizza i documenti troppo difficili:
    
    $$\\text{Penalty} = \\begin{cases} 
    1 + \\alpha & \\text{se readability} > \\text{target} \\\\
    1 & \\text{altrimenti}
    \\end{cases}$$
    
    - **α (alpha)**: Fattore di penalità (es. 0.5)
    - Documenti più difficili ricevono penalità maggiore
    - Incoraggia raccomandazioni al livello appropriato
    """)

divider()

# Pipeline del Modello
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

# Parametri del Modello
section_title(" Parametri Configurabili")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("""
    **η (Eta) - Peso Similarità**
    - Range: 0.0 - 1.0
    - Default: 0.6
    - Effetto: ↑ Enfatizza tematica
    """)

with col2:
    st.markdown("""
    **ζ (Zeta) - Peso Penalità**
    - Range: 0.0 - 1.0
    - Default: 0.4
    - Effetto: ↑ Enfatizza leggibilità
    """)

with col3:
    st.markdown("""
    **α (Alpha) - Fattore Penalità**
    - Range: 0.0 - 2.0
    - Default: 0.8
    - Effetto: ↑ Più severo su difficili
    """)

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

divider()

# Metriche di Valutazione
section_title(" Metriche di Valutazione")

tabs = st.tabs(["Precision@K"])

with tabs[0]:
    st.write("""
    **Precision@K**: Percentuale di raccomandazioni rilevanti nei Top-K
    
    - K=5, K=10, K=20 comuni
    - Misura: Quante raccomandazioni piacciono all'utente
    - Range: [0, 1], maggiore è meglio
    """)

divider()

st.markdown("---")
st.markdown("** Nota**: I parametri possono essere regolati nel file di configurazione `conf/project.yaml`")
