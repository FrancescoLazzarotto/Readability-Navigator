import pandas as pd 
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
import umap 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer
from bertopic import BERTopic
import matplotlib.pyplot as plt 
from collections import Counter
import sys
import os 
from umap import UMAP
from hdbscan import HDBSCAN
from loguru import logger


CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)

from utils.io_utils import load_csv, save_pickle, load_pickle, load_yaml
from utils.data_loader import load_features_df, load_embedding

config = load_yaml() 
rel_config = config['paths']['features_csv']
path = os.path.join(PROJECT_ROOT, rel_config)


#path = r"C:\Users\checc\OneDrive\Desktop\Readability-Navigator\data\processed\onestop_nltk_features.csv"

try: 
    
    dataframe = load_csv(path)
except FileNotFoundError:
    raise FileNotFoundError(f"File non trovato nel path {path}")

sentences = dataframe['testo']


def model_embedding():
    """Creazione modello
    
    Returns: 
        object: modello SBERT 
    """
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model 


model = model_embedding()

def sentences_embedding(sentences, model):
    """Generazione vettorizzazione del testo 
    
    Args: 
        sentences(str): testo da vettorizzare
        model (obj): modello SBERT per fare l'embedding
    
    Returns:
        list[list[float]]: embedding dei testi sotto forma di liste di vettori  
    
    """
    
    embedding = model.encode(
        sentences,
        truncate_dim = 512
                            )
    return embedding


#embedding = sentences_embedding(sentences, model)
#save_pickle('doc_embedding.pickle', embedding)


def get_document_embedding(doc_id):
    """Estrae l'embedding di un documento specifico
    
    Args:
        doc_id (str): ID del documento (es: "Amazon-Ele_easy")
    
    Returns:
        list: embedding del documento specifico
    """
    df = load_features_df()
    emb = load_embedding()
    
    idx_array = df.index[df["id"] == doc_id].tolist()
    
    if len(idx_array) == 0:
        raise ValueError(f"documento non trovato: {doc_id}")
    
    idx = idx_array[0]
    return emb[idx].tolist()


