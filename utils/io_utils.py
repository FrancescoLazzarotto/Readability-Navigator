import json
import pandas as pd
import os
import pickle
import yaml
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

def load_yaml(path=None):
    """Caricamento file di configurazione yaml.

    Args:
        path (str or None): path opzionale, se non dato prende quello di default.
    
    Returns:
        dict: contenuto file yaml parsed
    """
    
    if path is None:
        path = os.path.join(ROOT_DIR, 'conf', 'project.yaml')
        
    if not os.path.exists(path):
        raise FileNotFoundError(f"File di configurazione non trovato: {path}")

    with open(path, 'r', encoding='utf-8') as file:
        return yaml.safe_load(file)


def load_json(path):
    """Caricamento file json
    
    Args:
        path (str): stringa path del file json
    
    Returns:
        object: oggetto json parsed
    """
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
    

def save_json(data, path):
    """Salvataggio file json
    
    Args:
        data (dict): dati da salvare nel file json 
        path (str): stringa path dove salvare il file json
    
    Returns:
        None  
    
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)
        

def load_csv(path):
    """Caricamento file csv
    
    Args:
        path (str): stringa path da dove caricare il file
    
    Returns:
        pd: lettura csv con pandas
    """
    return pd.read_csv(path, encoding="utf-8")


def save_csv(df, path):
    """Salvataggio csv
    
    Args:
        df (pandas.DataFrame): oggetto dataframe pandas da salvare in formato csv
        path (str): path in cui salvare il file csv
        
    Returns:
        None
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")
    

def save_pickle(name_file, data):
    """Salvataggio file pickle
    
    Args:
        name_file (str): nome con cui salvare il file pickle
        data (object): dati da salvare nel file pickle
    
    Returns:
        None
    """
    with open(name_file, "wb") as pkl:
        pickle.dump(data, pkl)



def load_pickle(name_file):
    """Caricamento file pickle
    
    Args:
        name_file (str): nome del file pickle da caricare
        
    Returns:
        object: dati caricati dal file pickle
    """
    
    with open(name_file, 'rb') as pkl:
        data = pickle.load(pkl)
    return data
        
def find(name, path):
    """Trovare un file in una directory
    
    Args:
        name (str): nome del file da trovare
        path (str): path in cui cercare
        
    Returns:
        str: path preciso del file trovato
    """
    for root, dirs, files in os.walk(path):
        if name in files:
            return os.path.join(root, name)