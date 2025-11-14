import pandas as pd 
import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.cluster import KMeans
from collections import Counter
from utils.io_utils import load_data, save_pickle, load_pickle


path = r"C:\Users\checc\OneDrive\Desktop\readability-navigator\data\processed\onestop_nltk_features.csv"
try: 
    dataframe = load_data(path)
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

def similarity_embedding(embedding, model):     
    similarity = model.similarity(embedding, embedding)
    return similarity

def clustering(df, embedding, k = 10, random_state = 42):
    kmeans = KMeans(n_clusters=k, random_state = random_state)
    labels = kmeans.fit_predict(embedding)
    df["topic_cluster"] = labels
    return df, kmeans

def cluster_label(df, k):
    description = {}
    
    for clusters_id in range(k):
        text = df[df["topic_cluster"]] == clusters_id["testo"]
        words = " ".join(text).lower().split()
        common = []
        for index, word in Counter(words).most_common(10):
            description[clusters_id] = common 
    
    return description 
        

embedding = sentences_embedding(sentences, model)
similarity = similarity_embedding(embedding, model)
save_pickle('doc_embedding.pickle', embedding)
save_pickle('similarity_embedding.pickle', similarity)



