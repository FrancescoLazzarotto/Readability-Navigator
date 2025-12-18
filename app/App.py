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

if "selected_doc" not in st.session_state:
    st.session_state.selected_doc = None

if "current_user" not in st.session_state:
    st.session_state.current_user = None



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
        st.session_state.current_user = user 
        st.session_state.last_user_id = new_user_id
        st.session_state.show_recommendations_new = True
        
    if st.session_state.get("show_recommendations_new", False):
        divider()
        section_title("Raccomandazioni per Utente #" + str(st.session_state.current_user['user_id']))
        
        try:
            df = st.session_state.get("recommendations_df")
            if df is None:
                df = main(st.session_state.current_user)
                st.session_state.recommendations_df = df
            
            if df is not None and len(df) > 0:
                
                st.success(f"Trovate {len(df)} raccomandazioni!")
                st.write("Scegli quale raccomandazione leggere: ")
                st.dataframe(df, use_container_width=True)
                
                section_title("Seleziona un testo da leggere")

                doc_titles = df["title"].tolist()

                selected_title = st.selectbox(
                    "Scegli un testo da leggere",
                    options=doc_titles,
                    key="doc_selector_new"
                )

                col1, col2 = st.columns([1, 3])
                with col1:
                    if st.button("Apri testo", key="open_btn_new"):
                        matching_docs = df[df["title"] == selected_title]
                        if len(matching_docs) > 0:
                            st.session_state.selected_doc = matching_docs.iloc[0].to_dict()
                        else:
                            st.error("Documento non trovato")

                if st.session_state.selected_doc is not None:
                    divider()
                    section_title("Testo selezionato")
                    doc = st.session_state.selected_doc

                    st.markdown(f"### {doc['title']}")
                    st.markdown(f"**Score:** {doc['score']:.4f}")
                    st.markdown(f"**Flesch Score {doc['flesch_score']:.4f}")
                    st.divider()
                    st.write(doc["testo"])

                    difficulty = st.radio(
                        "Quanto hai trovato difficile questo testo?",
                        options=[1, 2, 3, 4, 5],
                        format_func=lambda x: {
                            1: "Molto facile",
                            2: "Facile",
                            3: "Adeguato",
                            4: "Difficile",
                            5: "Molto difficile"
                        }[x],
                        key="difficulty_radio_new"
                    )

                    col1, col2 = st.columns([1, 3])
                    with col1:
                        if st.button("Conferma valutazione", key="confirm_btn_new"):
                            try:
                                doc_id = str(doc['title'])
                                doc_readability = float(doc.get('flesch_score', 60))
                                difficulty_val = int(difficulty)
                                update_user_model(st.session_state.current_user, doc_id, doc_readability, difficulty_val)
                                st.success("Feedback registrato e profilo aggiornato!")
                                st.session_state.selected_doc = None
                                st.rerun()
                            except Exception as e:
                                st.error(f"Errore nell'aggiornamento del profilo: {str(e)}")

                """
                with st.expander("Visualizza Dettagli Completi"):
                    for idx, row in df.iterrows():
                        st.markdown(f"### {row['title']}")
                        st.write(f"Score: {row['score']:.4f}")
                        st.write(f"Testo:\n{row['testo']}")
                        st.divider() """
            else:
                st.warning("Nessuna raccomandazione disponibile con questi parametri")
                st.dataframe(df)
        except Exception as e:
            st.error(f"Errore nel caricamento: {str(e)}")




else:
    section_title("Usa Profilo Utente Esistente")
    
    existing_users = []
    
    for file in os.listdir(users_path):
        if file.startswith("user") and file.endswith(".json"):
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
            if load_btn:
                st.session_state.show_recommendations_existing = True
        
        divider()
        
        selected_file = [path for uid, path in existing_users if uid == selected_user_id][0]
        
        try:
            user_profile = load_user_model(f"user{selected_user_id}.json", users_path)
            st.session_state.current_user = user_profile 
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
            
            if st.session_state.get("show_recommendations_existing", False):
                section_title("Raccomandazioni per Utente #" + str(selected_user_id))
                
                try:
                    df = st.session_state.get("recommendations_df_existing")
                    if df is None:
                        df = main(user_profile)
                        st.session_state.recommendations_df_existing = df
                    
                    if df is not None and len(df) > 0:
                        st.success(f"Trovate {len(df)} raccomandazioni!")
                        st.write("Scegli quale raccomandazione leggere: ")
                        st.dataframe(df, use_container_width=True)
                        
                        section_title("Seleziona un testo da leggere")

                        doc_titles = df["title"].tolist()

                        selected_title = st.selectbox(
                            "Scegli un testo da leggere",
                            options=doc_titles,
                            key="doc_selector_existing"
                        )

                        col1, col2 = st.columns([1, 3])
                        with col1:
                            if st.button("Apri testo", key="open_btn_existing"):
                                matching_docs = df[df["title"] == selected_title]
                                if len(matching_docs) > 0:
                                    st.session_state.selected_doc_existing = matching_docs.iloc[0].to_dict()
                                else:
                                    st.error("Documento non trovato")

                        if st.session_state.get("selected_doc_existing") is not None:
                            divider()
                            section_title("Testo selezionato")
                            doc = st.session_state.selected_doc_existing

                            st.markdown(f"### {doc['title']}")
                            st.markdown(f"**Score:** {doc['score']:.4f}")
                            st.divider()
                            st.write(doc["testo"])

                            difficulty = st.radio(
                                "Quanto hai trovato difficile questo testo?",
                                options=[1, 2, 3, 4, 5],
                                format_func=lambda x: {
                                    1: "Molto facile",
                                    2: "Facile",
                                    3: "Adeguato",
                                    4: "Difficile",
                                    5: "Molto difficile"
                                }[x],
                                key="difficulty_radio_existing"
                            )

                            col1, col2 = st.columns([1, 3])
                            with col1:
                                if st.button("Conferma valutazione", key="confirm_btn_existing"):
                                    try:
                                        doc_id = str(doc['title'])
                                        doc_readability = float(doc.get('flesch_score', 60))
                                        difficulty_val = int(difficulty)
                                        update_user_model(st.session_state.current_user, doc_id, doc_readability, difficulty_val)
                                        st.success("Feedback registrato e profilo aggiornato!")
                                        st.session_state.selected_doc_existing = None
                                        st.rerun()
                                    except Exception as e:
                                        st.error(f"Errore nell'aggiornamento del profilo: {str(e)}")
                        
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