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

# Header
page_header("Features ed Embedding", "Estrazione di caratteristiche e rappresentazione vettoriale")

divider()

# Carica il dataset
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
    except Exception as e:
        st.warning(f"Non è stato possibile caricare gli embedding: {e}")
        return None

df = load_dataset()
embeddings = load_embeddings()

section_title("Features Estratte")

if df is not None:
    st.write("""
    Le features sono caratteristiche numeriche estratte dal testo per quantificare 
    proprietà linguistiche e strutturali di ogni documento.
    """)
    
    divider()
    
    # Feature Analysis
    section_title("Analisi delle Features")
    
    # Seleziona solo colonne numeriche e non testuali
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    
    if numeric_cols:
        st.write(f"**Numero di features numeriche: {len(numeric_cols)}**")
        
        # Statistiche features
        stats_df = pd.DataFrame({
            "Feature": numeric_cols,
            "Media": [df[col].mean() for col in numeric_cols],
            "Mediana": [df[col].median() for col in numeric_cols],
            "Min": [df[col].min() for col in numeric_cols],
            "Max": [df[col].max() for col in numeric_cols],
            "Std Dev": [df[col].std() for col in numeric_cols]
        })
        
        st.dataframe(stats_df.round(4))
        
        divider()
        
        # Visualizzazione Features
        section_title("Distribuzione delle Features Principali")
        
        if 'flesch_score' in numeric_cols:
            col1, col2 = st.columns(2)
            
            with col1:
                fig_flesch = px.histogram(
                    x=df['flesch_score'],
                    nbins=30,
                    color_discrete_sequence=['#FF6B6B'],
                    labels={'x': 'Flesch Score', 'y': 'Frequenza'},
                    title='Distribuzione del Flesch Score'
                )
                st.plotly_chart(fig_flesch, )
            
            with col2:
                if 'num_words' in df.columns or any('word' in col.lower() for col in numeric_cols):
                    word_col = 'num_words' if 'num_words' in df.columns else [col for col in numeric_cols if 'word' in col.lower()][0]
                    fig_words = px.histogram(
                        x=df[word_col],
                        nbins=30,
                        color_discrete_sequence=['#4ECDC4'],
                        labels={'x': 'Numero di Parole', 'y': 'Frequenza'},
                        title='Distribuzione della Lunghezza dei Documenti'
                    )
                    st.plotly_chart(fig_words, )
    
    divider()
    
    # Embedding Information
    section_title(" Embedding Vettoriali")
    
    st.write("""
    Gli embedding sono rappresentazioni vettoriali dense del testo che catturano 
    il significato semantico e il contesto dei documenti.
    """)
    
    if embeddings is not None:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Numero di Embedding", len(embeddings))
        with col2:
            dim_str = str(embeddings[0].shape[0]) if len(embeddings) > 0 else "N/A"
            st.write(f"** Dimensione Embedding:** {dim_str}")
        with col3:
            mem_str = f"{embeddings.nbytes / 1024**2:.2f} MB" if hasattr(embeddings, 'nbytes') else "N/A"
            st.write(f"** Memoria:** {mem_str}")
        
        divider()
        
        # PCA Visualization
        section_title(" Visualizzazione PCA degli Embedding")
        
        st.write("Riduzione dei 384 embedding a 2 dimensioni usando PCA per la visualizzazione:")
        
        try:
            pca = PCA(n_components=2)
            embeddings_2d = pca.fit_transform(embeddings)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Scatter plot base
                fig_scatter = go.Figure()
                fig_scatter.add_trace(go.Scatter(
                    x=embeddings_2d[:, 0],
                    y=embeddings_2d[:, 1],
                    mode='markers',
                    marker=dict(
                        size=6,
                        color=df['flesch_score'] if 'flesch_score' in df.columns else 'blue',
                        colorscale='Viridis',
                        showscale=True,
                        colorbar=dict(title="Flesch Score")
                    ),
                    text=df['id'] if 'id' in df.columns else "Documento",
                    hoverinfo='text'
                ))
                fig_scatter.update_layout(
                    title="Embedding Space Visualization (PCA)",
                    xaxis_title="Componente Principale 1",
                    yaxis_title="Componente Principale 2",
                    height=500
                )
                st.plotly_chart(fig_scatter, )
            
            with col2:
                # Varianza spiegata
                explained_variance = pca.explained_variance_ratio_
                fig_variance = go.Figure()
                fig_variance.add_trace(go.Bar(
                    x=['PC1', 'PC2'],
                    y=explained_variance,
                    marker_color=['#4ECDC4', '#FF6B6B']
                ))
                fig_variance.update_layout(
                    title="Varianza Spiegata dalle Componenti",
                    yaxis_title="Percentuale di Varianza",
                    height=500
                )
                st.plotly_chart(fig_variance, )
                
                st.info(f"""
                **Varianza Spiegata:**
                - PC1: {explained_variance[0]*100:.2f}%
                - PC2: {explained_variance[1]*100:.2f}%
                - Totale: {sum(explained_variance)*100:.2f}%
                """)
        
        except Exception as e:
            st.error(f"Errore nella visualizzazione PCA: {e}")
    
    else:
        st.warning(" Gli embedding non sono disponibili")
    
    divider()
    
    # Tecniche di Embedding
    section_title("Tecniche di Embedding")
    
    tabs = st.tabs(["Word Embeddings", "Sentence Embeddings", "TF-IDF"])
    
    with tabs[0]:
        st.write("""
        **Word2Vec / GloVe**
        
        - Ogni parola è rappresentata da un vettore denso
        - Cattura relazioni semantiche tra parole
        - Dimensioni tipiche: 100-300
        - Vantaggi: Veloce, interpretabile
        """)
    
    with tabs[1]:
        st.write("""
        **Sentence Embeddings (BERT / Sentence-BERT)**
        
        - Rappresentazione dell'intero documento/frase
        - Cattura il significato del contesto
        - Dimensioni tipiche: 384-768
        - Vantaggi: Semantica ricca, performance migliore
        """)
    
    with tabs[2]:
        st.write("""
        **TF-IDF (Term Frequency - Inverse Document Frequency)**
        
        - Rappresentazione sparsa basata su frequenze
        - Peso inversamente proporzionale alla frequenza nel corpus
        - Buono per problemi di similarità
        - Vantaggi: Interpretabile, efficiente
        """)
    
    divider()
    
    # Feature Correlation
    section_title(" Correlazione tra Features")
    
    if len(numeric_cols) > 1:
        correlation_matrix = df[numeric_cols].corr()
        
        fig_corr = go.Figure(data=go.Heatmap(
            z=correlation_matrix.values,
            x=correlation_matrix.columns,
            y=correlation_matrix.columns,
            colorscale='RdBu',
            zmid=0
        ))
        fig_corr.update_layout(
            title="Matrice di Correlazione delle Features",
            height=600
        )
        st.plotly_chart(fig_corr, )

else:
    st.error(" Non è stato possibile caricare il dataset")
