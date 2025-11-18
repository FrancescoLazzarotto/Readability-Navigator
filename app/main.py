import sys
import os
ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, ROOT)

from src.recommender.recommender_engine import RecommenderEngine
from utils.io_utils import load_csv, load_pickle, load_yamal
import numpy as np 



def load_utils():
    config = load_yamal()
    path = "data/processed/onestop_nltk_features.csv"
    df = load_csv(path)

    pickle_file = "src/features/similarity_embedding.picle"
    embedding = load_pickle(pickle_file)

    return config, df, embedding
_,df,_ = load_utils()
df.head() 

def main():
    config, df, embedding = load_utils()
    user = {
        "user_id": 10,
        "target_readability": 60,
        "topic_vector": list(np.random.rand(384)),
        "history": []
    }
    doc_id = 1

    engine = RecommenderEngine(
        df=df,
        embedding=embedding,
        config=config,
        user_id=user['user_id'],
        profile_path="user10.json"
    )
    
    score = engine.recommender(user, doc_id)
    rank = engine.rank_top_k(user)
    
    print("Score:", score)
    print("Top K Recommendations:", rank)


#if __name__ == "__main__":
 #   main()
