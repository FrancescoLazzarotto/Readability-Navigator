# utils/data_loader.py
import pandas as pd
import os
import sys 
from functools import lru_cache

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils.io_utils import load_yaml, load_pickle


@lru_cache(maxsize=1)
def load_features_df():
    """Carica DataFrame features (cached - carica solo una volta)"""
    config = load_yaml()
    data_rel = config['paths']['features_csv']
    data_path = os.path.join(PROJECT_ROOT, data_rel)
    
    if not os.path.exists(data_path):
        raise FileNotFoundError(f" file non trovato: {data_path}")
    
    print(f" Caricamento dataset da: {data_path}")
    return pd.read_csv(data_path)




@lru_cache(maxsize=1)
def load_embedding():
    config = load_yaml()
    emb_rel = config['paths']['embeddings_pickle']
    emb_path = os.path.join(PROJECT_ROOT, emb_rel)
    
    if not os.path.exists(emb_path):
        raise FileNotFoundError(f"file non trovato {emb_path}")
    return load_pickle(emb_path)

