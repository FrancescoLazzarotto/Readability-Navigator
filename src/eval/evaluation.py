import numpy as np
from sklearn.metrics import ndcg_score
import os, sys
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, "..", ".."))
sys.path.insert(0, PROJECT_ROOT)
from src.recommender.recommender_engine import RecommenderEngine
from utils.io_utils import load_yaml

class RecommenderEvaluation:
    """
    Valutazione offline del recommender tramite NDCG@K
    basata sulla distanza tra flesch score e target di leggibilità utente
    """

    def __init__(self, k):
        self.k = k
        self.ndcg_history = []

    @staticmethod
    def compute_relevance_from_flesch(flesch_scores, target):
        flesch_scores = np.array(flesch_scores)
        distances = np.abs(flesch_scores - target)
        max_dist = distances.max()
        if max_dist == 0:
            return np.ones_like(distances)
        return 1 - distances / max_dist

    def ndcg_at_k(self, flesch_scores, pred_scores, target):
        relevance = self.compute_relevance_from_flesch(flesch_scores, target)
        y_true = relevance.reshape(1, -1)
        y_score = np.array(pred_scores).reshape(1, -1)
        ndcg_value = ndcg_score(y_true, y_score, k=self.k)
        self.ndcg_history.append(ndcg_value)
        return ndcg_value

    def evaluate_users(self, recommender, users):
        """
        Valuta NDCG su una lista di utenti reali.
        
        Args:
            recommender (RecommenderEngine): il motore di raccomandazione
            users (list[dict]): lista dei profili utenti caricati da JSON
            
        Returns:
            float: NDCG medio su tutti gli utenti
        """
        self.ndcg_history = []

        for user in users:
            titles, scores, testi, flesch_values = recommender.rank_top_k(user)

            
            self.ndcg_at_k(
                flesch_scores=flesch_values,
                pred_scores=scores,
                target=user['target_readability']
            )

        return np.mean(self.ndcg_history)


# ==========================
# Esecuzione standalone
# ==========================
if __name__ == "__main__":
    import json
    import os
    import pandas as pd
    from utils.data_loader import load_features_df, load_embedding
    config = load_yaml()
    # =====================================================
    # Configurazione base
    # =====================================================
    configuration = {
        "tol": config['tol'],       # tolleranza readability
        "eta": config['eta'],      # peso similarità tematica
        "zeta": config['zeta'],     # peso gap leggibilità
        "alpha": config['alpha'],    # penalità over-readability
        "k": config['k']           # Top-K raccomandazioni
    }

    # =====================================================
    # Caricamento dati reali
    # =====================================================
    df = load_features_df()
    embedding = load_embedding()  
    rel_profile_path = config['paths']['user_json']
    profile_path = os.path.join(PROJECT_ROOT, rel_profile_path)
    user_ids = [f.replace(".json","") for f in os.listdir(profile_path) if f.endswith(".json")]

    
    users = []
    for uid in user_ids:
        with open(os.path.join(profile_path, f"{uid}.json"), "r") as f:
            users.append(json.load(f))

    # =====================================================
    # Istanzio il recommender
    # =====================================================
    recommender = RecommenderEngine(df, embedding, configuration, user_id=None, profile_path=profile_path)

    # =====================================================
    # Valutazione
    # =====================================================
    evaluator = RecommenderEvaluation(k=configuration['k'])
    final_ndcg = evaluator.evaluate_users(recommender, users)

    print("NDCG medio su tutti gli utenti:", round(final_ndcg, 4))
    print("Valori NDCG per ogni utente:", evaluator.ndcg_history)
