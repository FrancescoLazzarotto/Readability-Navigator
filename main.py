import sys
import os
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.recommender.recommender_engine import RecommenderEngine
from utils.io_utils import load_csv, load_pickle, load_yaml
import numpy as np 
import pandas as pd



def load_utils():
    """Caricamento utils: file di configurazione, dataframe, emebedding
    
    Returns:
        dict: dizionario di configurazione
        pandas.DataFrame: dati
        list[list[float]]: lista di vettori (embedding) dei testi 
    """
    config = load_yaml()

    rel_csv_path = config['paths']['features_csv']
    rel_pickle_path = config['paths']['embeddings_pickle']

    csv_path = os.path.join(ROOT, rel_csv_path)
    pickle_path = os.path.join(ROOT, rel_pickle_path)

    df = load_csv(csv_path)
    embedding = load_pickle(pickle_path)

    return config, df, embedding



def main(user):
    """Main, richiamo degli utils e creazione dell'oggetto reccomender, ranking dei documento    
    
    Args: 
        user(dict): dizionario contenente dei dati degli utenti
    
    Returns:
        list[List]
    """
    config, df, embedding = load_utils()
    
    engine = RecommenderEngine(
        df=df,
        embedding=embedding,
        config=config,
        user_id=user['user_id'],
        profile_path= None
    )
    
    rank = engine.rank_to_df(user)
    
    return rank


if __name__ == "__main__":
    user = {
        "user_id": 10,
        "target_readability": 60,
        "topic_vector": list(np.random.rand(384)),
        "history": []
    }
    main(user)

