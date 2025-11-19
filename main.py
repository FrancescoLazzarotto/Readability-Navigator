import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.recommender.recommender_engine import RecommenderEngine
from utils.io_utils import load_csv, load_pickle, load_yamal
import numpy as np 
import pandas as pd


def load_utils():
    config = load_yamal()

    # Percorsi assoluti
    csv_path = os.path.join(ROOT, "data", "processed", "onestop_nltk_features.csv")
    pickle_file = os.path.join(ROOT, "src", "features", "doc_embedding.pickle")

    df = load_csv(csv_path)
    embedding = load_pickle(pickle_file)

    return config, df, embedding


def main():
    config, df, embedding = load_utils()
    user = {
        "user_id": 10,
        "target_readability": 60,
        "topic_vector": list(np.random.rand(384)),
        "history": []
    }
    doc_id = "Amazon-Ele_easy"

    engine = RecommenderEngine(
        df=df,
        embedding=embedding,
        config=config,
        user_id=user['user_id'],
        profile_path="user10.json"
    )
    

    #score = engine.recommender(user, doc_id)
    #rank = engine.rank_top_k(user)
    rank = engine.rank_to_df(user)
    
    #print("Score:", score)
    #print("Top K Recommendations:", rank)
    #rank.to_csv('test3.csv', index=True)
    return rank
    
    
if __name__ == "__main__":
    main()


