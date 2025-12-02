import pandas as pd 
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
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




path = r"C:\Users\checc\OneDrive\Desktop\readability-navigator\data\processed\onestop_nltk_features.csv"
PATH_OUTPUT_DIR = "data_artifacts/"
try: 
    dataframe = load_csv(path)
except FileNotFoundError:
    raise FileNotFoundError(f"File non trovato nel path {path}")

sentences = dataframe['testo']


def model_embedding():
    model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    return model 

model = model_embedding()

def sentences_embedding(sentences, model):
    
    embedding = model.encode(
        sentences,
        truncate_dim = 512
                            )
    return embedding

#def topic_embedding(topic_text, model):
    #embedding = model.encode([topic_text], truncate_dim=512)
    #return embedding[0].tolist()


#embedding = sentences_embedding(sentences, model)
#save_pickle('doc_embedding.pickle', embedding)





def topic_model(df, emb):
    text = df["testo"]
    logger.info("UMAP")
    umap_model = UMAP(n_neighbors=15, n_components=5, min_dist=0.0)

    logger.info("HDBSCAN")
    hdbscan_model = HDBSCAN(min_cluster_size=10)
    vectorizer_model = CountVectorizer(stop_words="english")
    
    logger.info("BERTopic final")
    model = BERTopic(umap_model=umap_model,
                     hdbscan_model=hdbscan_model,
                     vectorizer_model=vectorizer_model)
    topics = model.fit_transform(text, emb)

    return topics, model
    


def visualize_document_data(df, model, emb):
    return model.visualize_document_datamap(df, embeddings=emb)


    

config = load_yaml() 
rel_emb = config['paths']['embeddings_pickle']
emb_path = os.path.join(PROJECT_ROOT, rel_emb)
emb = load_pickle(emb_path)

rel_df_path = config['paths']['features_csv'] 
df_path = os.path.join(PROJECT_ROOT, rel_df_path) 
df = load_csv(df_path) 

topics, model = topic_model(df, emb)
df.head()
classes = df['titolo']
text = df['testo']
topic_class = model.topics_per_class(text, classes)
df.head()
#model.visualize_topics_per_class(topic_class)