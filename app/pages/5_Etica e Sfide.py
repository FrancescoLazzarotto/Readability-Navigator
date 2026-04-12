import streamlit as st
import plotly.graph_objects as go
import plotly.express as px
import numpy as np
import pandas as pd
from PIL import Image

st.set_page_config(page_title="Fondamenti Cognitivi - Readability Navigator", layout="wide")

# ==================== HEADER ====================
st.title("Fondamenti Cognitivi di Readability Navigator")
st.markdown("**Dal modello teorico all'implementazione algoritmica**")




# ==================== TAB STRUCTURE ====================
tab1, tab2 = st.tabs([
    "1️⃣ Simbolico vs. Connessionista",
    "2️⃣ Etica e Sfide"
])

# ============================================================
# TAB 1: RAPPRESENTAZIONE DELLA CONOSCENZA
# ============================================================
with tab1:
    st.header("1. Rappresentazione della Conoscenza: Simbolica vs. Subsimbolica")
    st.markdown("""
    Le scienze cognitive distinguono storicamente tra due paradigmi fondamentali di come la mente rappresenta e elabora la conoscenza:
    - **IA Simbolica**: Regole logiche esplicite, manipolazione di simboli discreti
    - **IA Connessionista/Subsimbolica**: Rappresentazioni distribuite, vettori, significato emergente
    """)
    
    # =============== SEZIONE 1: FLESCH (SIMBOLICO) ===============
    st.divider()
    st.subheader("Il Lato Simbolico: Indice Flesch Reading Ease")
    
    col1, col2 = st.columns([1.2, 0.8])
    
    with col1:
        st.write("""
        ### Che cosa è il Flesch Score?
        
        L'indice Flesch è un **algoritmo simbolico deterministico**:
        
        **Caratteristiche**:
        -  **Manipola simboli discreti**: conta parole, conta sillabe, conta frasi
        -  **Applica regole logico-matematiche rigide**: è una formula deterministica
        -  **Indipendente dal supporto fisico**: funziona in carta come in software
        -  **Non apprende**: l'algoritmo rimane sempre uguale
        """)
    
    with col2:
        st.metric(label="Esempio di Calcolo", value="52.3", delta="Leggibilità Moderata")
        st.markdown("""
        **Input**:
        - Parole: 200
        - Frasi: 5
        - Sillabe: 280
        
        **Output**: 
        Score = 52.3
        
        """)
    
    
    # =============== SEZIONE 2: SBERT (CONNESSIONISTA) ===============
    st.divider()
    st.subheader("Il Lato Connessionista: SBERT Embeddings")
    
    col1, col2 = st.columns([1.2, 0.8])
    
    with col1:
        st.write("""
        ### Che cosa è SBERT (Sentence-BERT)?
        
        SBERT trasforma ogni testo in un **vettore a 384 dimensioni**:
        ```
        testo = "Il cervello è plastico e apprende"
        ↓ [Encoder SBERT]
        embedding = [0.12, -0.45, 0.87, ..., 0.23]  # 384 numeri
        ```
        
        **Caratteristiche**:
        -  **Rappresentazione distribuita**: il significato è sparso su 384 neuroni artificiali
        -  **Sub-simbolica**: non usa parole chiave, ma pattern numerici
        -  **Apprende da dati**: la rete è stata addestrata su milioni di testi
        -  **Black-box**: non sappiamo quale dimensione = quale significato
        """)
    
    with col2:
        st.metric(label="Dimensioni Embedding", value="384", delta="sub-simboliche")
        st.markdown("""
        **Similitudine Semantica**:
        - "Il cervello è plastico" 
          vs.
        - "Il cervello è flessibile"
        
        → **Cosine Similarity = 0.94**
        (Altamente simili 
        nel significato)
        """)
        
    # =============== SEZIONE 3: L'APPROCCIO IBRIDO ===============
    st.divider()
    st.subheader("Readability Navigator: Un Sistema Ibrido")
    
    st.markdown("""
    Il progetto integra quindi i due paradigmi in modo complementare:
    """)
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        **Difficoltà: (Flesch - Simbolico)**
        
        Input: Struttura sintattica
        
        Flesch = 52.3 → "Moderatamente difficile"
        
        Logica: Regole formali
        
        Output: Numero [0-100]
        """)
    
    with col2:
        st.markdown("""
        **Interesse: (SBERT - Connessionista)**
        
        Input: Significato semantico
        
        Embedding = [0.12, -0.45, ...] → Cosine Sim = 0.87
        
        Logica: Pattern distribuito
        
        Output: Numero [0-1]
        """)
    
    with col3:
        st.markdown("""
        **Combinazione: Scoring Ibrido (Flesch × SBERT)**
        
        Input: Difficoltà + Interesse
        
        Score = Flesch × Similarity → Ranking
        
        Logica: Due paradigmi uniti
        
        Output: Migliori Raccomandazioni per utente 
        """)
    
    
    
    # =============== SEZIONE 4: CONCLUSIONE ===============
    st.divider()
    st.subheader(" In sintesi")
    
    comparison_data = pd.DataFrame({
        "Aspetto": [
            "Paradigma",
            "Come rappresenta la conoscenza",
            "Struttura dell'informazione",
            "Apprende?",
            "Interpretabilità",
            
        ],
        "Flesch Score": [
            "Funzionalismo Computazionale",
            "Regole logico-matematiche esplicite",
            "Simboli discreti (parole, frasi)",
            " No (algoritmo fisso)",
            " Completamente trasparente",
            
        ],
        "Embedding": [
            "Reti Neurali Artificiali",
            "Pattern distribuiti su vettori",
            "Numeri continui (384 dimensioni)",
            " Sì (addestramento su dati)",
            " Black-box (opaco)",
            
        ]
    })
    
    st.dataframe(comparison_data, use_container_width=True)


# ============================================================
# TAB 2: ETICA E SFIDE
# ============================================================
with tab2:
    st.header("Etica e Sfide")
    
    st.markdown("""
    Una sistema di raccomandazioni personalizzate solleva importanti questioni etiche e pratiche
    che non possono essere ignorate, soprattutto quando rivolto a utenti con esigenze specifiche.
    """)
    
    # =============== SEZIONE 1: BLACK BOX ===============
    st.subheader("1. La Black Box: Trasparenza vs. Precisione")
    
    st.markdown("""
    SBERT è straordinariamente efficace nel comprendere il significato semantico, ma è un sistema opaco.
    
    **Il compromesso:**
    - SBERT è preciso ma non spiegabile (384 dimensioni nascoste)
    - Flesch è trasparente ma cattura solo aspetti sintattici superficiali
    
    Gli utenti ricevono raccomandazioni buone, ma non sanno il perché dietro ogni scelta.
    """)
    
    st.divider()
    
    # =============== SEZIONE 2: CONTROLLO DELL'UTENTE ===============
    st.subheader("2. Il Problema del Controllo")
    
    st.markdown("""
    Quando il sistema aggiorna il target_readability in base al feedback, l'utente **non può rifiutare la modifica**.
    
    **Conseguenze:**
    - **Errori permanenti**: Un errore di valutazione influenza il profilo per sempre
    - **Nessuna anteprima**: Non vedi cosa cambierà prima che accada
    - **No reversibilità**: Non puoi annullare un aggiornamento sbagliato
    - **Perdita di agency**: Particolarmente critico per utenti con DSA che hanno bisogno di controllo
    """)
    
    st.divider()
    
    # =============== SEZIONE 3: LIMITAZIONI ===============
    st.subheader("3. Limitazioni Attuali")
    
    limitations = pd.DataFrame({
        "Aspetto": ["Aggiornamento", "Spiegazioni", "Profili", "Accessibilità", "Validazione"],
        "Stato Attuale": ["Automatico, non revocabile", "Black box SBERT", "Non modificabili", "Limitata", "No test su DSA"],
    })
    st.dataframe(limitations, use_container_width=True, hide_index=True)
    
    st.divider()
    
    # =============== SEZIONE 4: MIGLIORAMENTI ===============
    st.subheader("4. Percorso di miglioramento")
    
    st.markdown("""
    Readability Navigator nasce con l'obiettivo specifico di supportare le persone con Disturbi Specifici 
    dell'Apprendimento (DSA), affrontando barriere concrete nella lettura. Tuttavia, **il progetto nella sua 
    forma attuale non è ancora pronto per un uso reale con utenti DSA**: necessita di miglioramenti significativi 
    per essere eticamente responsabile e veramente inclusivo.
    """)
    
    st.markdown("**Controllo dell'Utente**")
    st.markdown("""
    L'utente deve avere pieno controllo sul proprio profilo. Implementare un meccanismo che permetta 
    di rifiutare modifiche proposte dal sistema, visualizzare un'anteprima prima dell'aggiornamento, 
    e resettare il profilo in qualunque momento sono azioni fondamentali per rispettare l'autonomia dell'utente.
    """)
    
    st.markdown("**Trasparenza delle Decisioni**")
    st.markdown("""
    Mentre SBERT rimane un sistema opaco, è possibile fornire spiegazioni parziali ma utili: 
    "Questo documento parla di temi simili a quelli che leggi" oppure "Questo testo ha parole lunghe 
    e frasi complesse". Anche una trasparenza imperfetta è meglio di nessuna.
    """)
    
    st.markdown("**Accessibilità Adattata**")
    st.markdown("""
    Font configurabile, supporto per lettura ad alta voce, e adattamenti visuali specifici per dislessia 
    non sono "nice to have" ma requisiti fondamentali. Un sistema pensato per DSA deve essere effettivamente 
    accessibile a chi ne ha bisogno.
    """)
    
    st.divider()
    
 

