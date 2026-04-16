import sys
import os
import argparse
ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from src.recommender.recommender_engine import RecommenderEngine
from utils.io_utils import load_csv, load_pickle, load_yaml
from src.user.model_user import build_user_model, load_user_model
import numpy as np 
import pandas as pd



def load_utils():
    config = load_yaml()

    rel_csv_path = config['paths']['features_csv']
    rel_pickle_path = config['paths']['embeddings_pickle']

    csv_path = os.path.join(ROOT, rel_csv_path)
    pickle_path = os.path.join(ROOT, rel_pickle_path)

    df = load_csv(csv_path)
    embedding = load_pickle(pickle_path)

    return config, df, embedding



def main(user):
    
    if user is None:
        user = {
            "user_id": 1,
            "target_readability": 60,
            "topic_vector": list(np.random.rand(384)),
            "history": []
        }
    
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


def _load_or_create_user(user_id: int):
    config = load_yaml()
    users_path = os.path.join(ROOT, config["paths"]["user_json"])
    filename = f"user{user_id}.json"

    try:
        return load_user_model(filename, users_path)
    except FileNotFoundError:
        return build_user_model(user_id=user_id, default_readability=60, save=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="CLI checkpoint for Readability Navigator"
    )
    parser.add_argument(
        "--user-id",
        type=int,
        default=1,
        help="User profile ID to use for recommendation",
    )
    args = parser.parse_args()

    user = _load_or_create_user(args.user_id)
    ranked_df = main(user)

    if ranked_df.empty:
        print("No recommendations found for the selected user.")
    else:
        print(ranked_df[["title", "score", "flesch_score"]].to_string(index=False))




