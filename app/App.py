import streamlit as st
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from components.sidebar import render_sidebar
from components.layout import page_header, divider, section_title
from main import main 
from src.user.model_user import build_user_model, load_user_model, update_user_model, save_user_json
from utils.io_utils import load_yaml

config = load_yaml()
rel_path_users = config['paths']['user_json']
users_path = os.path.join(PROJECT_ROOT, rel_path_users)
os.makedirs(users_path, exist_ok=True)
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
            value=0,
            step=5
        )
    
    with col3:
        st.write("")
        st.write("")
        generate_btn = st.button("Genera Raccomandazioni", key="generate_new")
    
    divider()

    if generate_btn: 
        user = build_user_model(user_id = new_user_id, default_readability=target_readability, save=True)

        st.session_state.last_user_id = new_user_id
        
        divider()
        section_title("Raccomandazioni per Utente #" + str(new_user_id))
        
        try:
            df = main(user)
            if df is not None and len(df) > 0:
                
                st.success(f"Trovate {len(df)} raccomandazioni!")
                st.write("Scegli quale raccomandazione leggere: ")
                st.dataframe(df, use_container_width=True)
                
                
                
                
                with st.expander("Visualizza Dettagli Completi"):
                    for idx, row in df.iterrows():
                        st.markdown(f"### {row['title']}")
                        st.write(f"Score: {row['score']:.4f}")
                        st.write(f"Testo:\n{row['testo']}")
                        st.divider()
            else:
                st.warning("Nessuna raccomandazione disponibile con questi parametri")
                st.dataframe(df)
        except Exception as e:
            st.error(f"Errore nel caricamento: {str(e)}")




else:
    section_title("Usa Profilo Utente Esistente")
    
    existing_users = []
    
    for file in os.listdir(users_path):
        if file.startswith("user") and file.endswith("son"):
            path = os.path.join(users_path, file)
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
            user_profile = load_user_model(f"user{selected_user_id}.json", users_path)
            if user_profile is not None:
                st.success(f"Profilo caricato: Utente #{selected_user_id}")
            
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric("User ID", user_profile.get("user_id", "N/A"))
            with col2:
                st.metric("Target Readability", user_profile.get("target_readability", "N/A"))
            with col3:
                st.metric("Documenti Visti", len(user_profile.get("history", [])))

            divider()
            

            if load_btn:
                section_title("Raccomandazioni per Utente #" + str(selected_user_id))
                
                try:
                    df = main(user_profile)
                    if df is not None and len(df) > 0:
                        st.success(f"Trovate {len(df)} raccomandazioni!")
                        st.dataframe(df, use_container_width=True)
                        
                        with st.expander("Visualizza Dettagli Completi"):
                            for idx, row in df.iterrows():
                                st.markdown(f"### {row['title']}")
                                st.write(f"Score: {row['score']:.4f}")
                                st.write(f"Testo:\n{row['testo']}")
                                st.divider()
                    else:
                        st.warning("Nessuna raccomandazione disponibile per questo utente")
                except Exception as e:
                    st.error(f"Errore: {str(e)}")
        
        except Exception as e:
            st.error(f"Errore nel caricamento del profilo: {str(e)}")
    
    else:
        st.info("Nessun profilo utente esistente. Crea un nuovo utente per iniziare!")
        print("Percorso calcolato da Python:", users_path)
        print("Percorso calcolato da Python:", PROJECT_ROOT)