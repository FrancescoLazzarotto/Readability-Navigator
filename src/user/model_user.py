import numpy as np
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
from utils.io_utils import load_json, load_yaml, save_json, load_pickle
config = load_yaml() 

rel_path_users = config['paths']['user_json']
users_path = os.path.join(PROJECT_ROOT, rel_path_users)
rel_emb = config['paths']['embeddings_pickle']
emb_path = os.path.join(PROJECT_ROOT, rel_emb)
emb = load_pickle(emb_path)

def save_user_json(user, user_id):
    """Salva un profilo utente nel file JSON
    
    Args:
        user (dict): dizionario con i dati dell'utente
        user_id (int): identificativo dell'utente
    """
    os.makedirs(users_path, exist_ok=True)
    file_name = f"user{user_id}.json"
    path = os.path.join(users_path, file_name)
    save_json(user, path)
    
    
    

def initialize_topic_vector(embedding):
    """Calcola il topic vector iniziale come centroide del corpus
    
    Args:
        embedding (np. ndarray): matrice N x 384 degli embedding
    
    Returns: 
        np.ndarray: vettore 1 x 384 
    """
    emb = np.array(embedding)
    norm_emb = emb / np.linalg.norm(emb, axis=1, keepdims=True)
    mean = np.mean(norm_emb, axis=0)
    norm_mean = mean / np.linalg.norm(mean)
    np.save("topic_vector_init.npy", norm_mean)
    

initialize_topic_vector(emb)



def build_user_model(user_id, *, topic_vector_default=None, default_readability=60, save=True):
    """Crea un nuovo profilo utente e lo salva
    
    Args:
        user_id (int): identificativo univoco dell'utente
        topic_vector_init (np.ndarray): vettore iniziale 1 x 384
        default_readability (int): target readability preferito (default 60)
        save (bool): se True, salva il profilo nel JSON (default True)
    
    Returns:
        dict: dizionario con i dati dell'utente
    """
    
    if topic_vector_default is None:
        topic_vector_default = np.load("topic_vector_init.npy")
    
    
    user = {
        "user_id": user_id,
        "topic_vector": topic_vector_default.tolist(),
        "target_readability": default_readability,
        "history": []
    }
    
    if save:
        save_user_json(user, user_id)
    
    return user



def load_user_model(name, path):
    """Carica un profilo utente da file JSON
    
    Args:
        name (str): nome del file in formato userid.json
        path (str): percorso della directory contenente i file JSON
    
    Returns:
        dict: dizionario con i dati dell'utente
    """
    path_user = os.path.join(path, name)
    
    if not os.path.exists(path_user):
        raise FileNotFoundError(f"Profilo utente non trovato: {path_user}")
    
    user_json = load_json(path_user)
    
    user = {
        "user_id": user_json['user_id'],
        "topic_vector": user_json['topic_vector'],
        "target_readability": user_json['target_readability'],
        "history": user_json.get('history', []),
    }
    
    return user

def update_user_model(user, doc_id, doc_embedding, alpha=0.3):
    """
    Aggiorna il profilo utente quando legge un documento
    
    Args:
        user (dict): profilo utente
        doc_id (str): ID del documento letto
        doc_embedding (list): embedding del documento
        alpha (float): peso del nuovo documento (0.3 = 30% nuovo, 70% passato)
    
    Returns:
        dict: profilo utente aggiornato
    """
    if doc_id not in user["history"]:
        user["history"].append(doc_id)
    
    old_vector = np.array(user["topic_vector"])
    new_embedding = np.array(doc_embedding)
    
    new_embedding = new_embedding / np.linealg.norm(new_embedding)
    
    updated_vector = ((1 - alpha) * old_vector) + (alpha * new_embedding)
    
    updated_vector = updated_vector / np.linalg.norm(updated_vector)
    user["topic_vector"] = updated_vector.tolist()
    
    save_user_json(user, user["user_id"])
    
    return user












    
    



