import numpy as np
import os
import sys

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
from utils.io_utils import load_json, load_yaml, save_json, load_pickle
from src.features.embeddings import get_document_embedding
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



def difficulty_to_alpha(difficulty):
    """Mappamento della difficultà dell'utente espressa in feedback (1-5) con valori alpha
    
    Args:
        difficulty(int): difficoltà espressa dall'utente
        
    Returns:
        float: alpha corrispondente alla difficoltà espressa
    """
    mapping = {
        1: 0.1,
        2: 0.2,
        3: 0.4,
        4: 0.2,
        5: 0.1
    }
    return mapping.get(difficulty, 0.1)


def update_topic_vector(user, doc_embedding, difficulty):
    """Aggiornamento topic_vector dell'user model
    
    Args: 
        user (dict): user model dell'utente
        doc_embedding (list[list[float]]): embedding del documento specifico 
        difficulty (int): difficoltà di feedback espressa dall'utente convertita tramite la funzione in alpha
        
    Returns:
        list[list[float]]: vettore aggiornato in base al peso 
    """
    alpha = difficulty_to_alpha(difficulty)
    
    old_vector = np.array(user["topic_vector"])
    new_embedding = np.array(doc_embedding)
    
    new_embedding = new_embedding / np.linalg.norm(new_embedding)
    updated_vector = ((1- alpha) * old_vector) + (alpha * new_embedding)
    updated_vector = updated_vector / np.linalg.norm(updated_vector)
    
    return updated_vector
    

def update_target_readability(old_target, doc_readability, difficulty, learning_rate = 0.1):
    """Aggiornamento target readaibiity dell'user model
        -negativo troppo facile positivo troppo difficile 
    
    Args:
        old_target(int): target readability dell'user model
        doc_readability(int): leggibilità del documento
        difficulty(int): difficoltà espressa dall'utente (1-5)
        learning_rate(float): parametro che controlla la dimensione del passo per la modalità di aggiornamento
    
    Returns:
        int: target readability dell'utente aggiornato
    """
    error = difficulty - 3 
    shift = learning_rate * error * (old_target - doc_readability)
    
    new_target = old_target - shift
    
    return new_target


def update_history(user, doc_id):
    if doc_id not in user["history"]:
        user["history"].append(doc_id)
        
        
    
def update_user_model(user, doc_id, doc_readability, difficulty):
    """Aggiorna il profilo utente quando legge un documento - richiama le funzioni per aggiornare:
        -topic_vector 
        -target_readability
        -history
    
    Args:
        user (dict): user model dell'utente
        doc_id (int or str): identificativo del documento appena letto 
        doc_embedding (list[list[float]]): embedding del documento appena letto
        doc_readability (int or float): leggibilità del documento
        difficulty (int): difficoltà espressa dall'utente (1-5)
    
    Returns:
        dict: user model aggiornato - history, topic_vector, target_readability
        
    """
    update_history(user, doc_id)
    
    doc_embedding = get_document_embedding(doc_id)
    new_vector = update_topic_vector(user, doc_embedding, difficulty)
    new_target = update_target_readability(user['target_readability'], doc_readability, difficulty)
    
    user['topic_vector'] = new_vector.tolist()
    user['target_readability'] = new_target
    
    save_user_json(user, user["user_id"])
    
    return user














    
    



