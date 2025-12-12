import streamlit as st
import numpy as np
import json
import os
import sys
import os
import json
import streamlit as st

APP_DIR = os.path.dirname(os.path.abspath(__file__))   # prog/app
PROJECT_DIR = os.path.dirname(APP_DIR)                 # prog
json_path = os.path.join(PROJECT_DIR, "user.json")     # prog/file.json
json_path = os.path.normpath(json_path)

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

#st.write("Percorso JSON:", json_path)  


with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)
from components.sidebar import render_sidebar
from components.layout import page_header, divider, section_title
from main import main 

render_sidebar()

page_header("Readability Navigator", "Generatore di raccomandazioni personalizzate")

divider()

section_title("Seleziona Modalità Utente")



user_mode = st.radio(
    "Scegli come procedere:",
    ["Crea Nuovo Utente", "Usa Utente Esistente"],
    horizontal=True
)

divider()

if user_mode == "Crea Nuovo Utente":
    section_title("Crea Nuovo Profilo Utente")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        new_user_id = st.number_input(
            "ID Utente",
            min_value=1,
            value=st.session_state.get("last_user_id", 1),
            step=1
        )
    
    with col2:
        target_readability = st.slider(
            "Target Readability",
            min_value=0,
            max_value=100,
            value=60,
            step=5
        )
    
    with col3:
        st.write("")
        st.write("")
        generate_btn = st.button("Genera Raccomandazioni", key="generate_new")
    
    divider()
    
    st.write("Seleziona Argomento")
    
    available_topics = [
        "None","Amazon","Amsterdam","Arctic","Banksy","Brazil","Climate Change","Copyright",
        "Crowdfunding","Denmark","Everest","Exercise","Facebook","False Memory"
    ]
    
    selected_topic_raw = st.selectbox(
        "Scegli un argomento",
        available_topics
    )

    if selected_topic_raw == "None": 
        selected_topic = None
    else: 
        selected_topic = selected_topic_raw
    
    
    if generate_btn: 
        np.random.seed(new_user_id)
        topic_vector = list(np.random.rand(384))
        
        user = {
        "user_id": new_user_id,
        "target_readability": target_readability,
        "topic_vector": topic_vector,
        "history": [],
        "profile_path": f"user{new_user_id}.json"
        }

        st.session_state.last_user_id = new_user_id
        
        divider()
        section_title("Raccomandazioni per Utente #" + str(new_user_id))
        
        try:
            df = main(user, selected_topic)
            if df is not None and len(df) > 0:
                st.success(f"Trovate {len(df)} raccomandazioni!")
                st.dataframe(df, use_container_width=True)
                
                with st.expander("Visualizza Dettagli Completi"):
                    for idx, row in df.iterrows():
                        st.markdown(f"### {row['title']}")
                        st.write(f"Score: {row['score']:.4f}")
                        st.write(f"Testo:\n{row['testo'][:500]}...")
                        st.divider()
            else:
                st.warning("Nessuna raccomandazione disponibile con questi parametri")
                st.dataframe(df)
        except Exception as e:
            st.error(f"Errore nel caricamento: {str(e)}")




else:
    section_title("Usa Profilo Utente Esistente")
    
    existing_users = []

# Cerca i profili nella cartella principale del progetto
    for file in os.listdir(PROJECT_DIR):
        if file.startswith("user") and file.endswith("son"):
            path = os.path.join(PROJECT_DIR, file)
            try:
                uid = int(file[4:-5])
                existing_users.append((uid, path))
            except:
                pass

    if existing_users:
        existing_users.sort(key=lambda x: x[0])
        
        col1, col2 = st.columns([2, 1])
        
        with col1:
            selected_user_id = st.selectbox(
                "Seleziona Utente Esistente",
                [uid for uid, _ in existing_users],
                format_func=lambda x: f"Utente #{x}"
            )
        
        with col2:
            st.write("")
            st.write("")
            load_btn = st.button("Carica Profilo", key="load_existing")
        
        divider()
        
        selected_file = [path for uid, path in existing_users if uid == selected_user_id][0]
        
        try:
            
            user_profile = data
            if user_profile is not None:
                    print("Dati ricevuti")
            else: 
                    print("Dati non ricevuti")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("User ID", user_profile.get("user_id", "N/A"))
            with col2:
                st.metric("Target Readability", user_profile.get("target_readability", "N/A"))
            with col3:
                st.metric("Documenti Visti", len(user_profile.get("history", [])))

            divider()
            st.write("Seleziona Argomento")

            available_topics = [
                "Amazon","Amsterdam","Arctic","Banksy","Brazil","Climate Change","Copyright",
                "Crowdfunding","Denmark","Everest","Exercise","Facebook","False Memory"
            ]

            selected_topic_raw = st.selectbox(
                "Scegli un argomento",
                available_topics
            )

            if selected_topic_raw == "None": 
                selected_topic = None
            else:
                selected_topic_raw

            if load_btn:
                section_title("Raccomandazioni per Utente #" + str(selected_user_id))
                
                try:
                    df = main(user_profile, selected_topic)
                    if df is not None and len(df) > 0:
                        st.success(f"Trovate {len(df)} raccomandazioni!")
                        st.dataframe(df, use_container_width=True)
                        
                        with st.expander("Visualizza Dettagli Completi"):
                            for idx, row in df.iterrows():
                                st.markdown(f"### {row['title']}")
                                st.write(f"Score: {row['score']:.4f}")
                                st.write(f"Testo:\n{row['testo'][:500]}...")
                                st.divider()
                    else:
                        st.warning("Nessuna raccomandazione disponibile per questo utente")
                except Exception as e:
                    st.error(f"Errore: {str(e)}")
        
        except Exception as e:
            st.error(f"Errore nel caricamento del profilo: {str(e)}")
    
    else:
        st.info("Nessun profilo utente esistente. Crea un nuovo utente per iniziare!")
