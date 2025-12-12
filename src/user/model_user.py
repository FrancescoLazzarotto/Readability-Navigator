import numpy as np
import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, ROOT)


def user_model(user_id, chosen_cluster, df, embedding, default_readability = 60):
    
    selected_embeddings  = []
    
    for cluster in chosen_cluster:
        id = df[df["topic_cluster"] == cluster].index
        emb = np.mean(embedding[id], axis= 0)
        selected_embeddings.append(emb)
    
    topic_vector = np.mean(selected_embeddings, axis=0)
     
    profile = {
        "user_id": user_id,
        "topic_vector": topic_vector.tolist(),
        "target_readability": default_readability,
        #"chosen_topics": chosen_cluster,
        "history": []
    }
    
    return profile

profile = {
    "user_id": 200,
    "topic_vector": list(np.random.rand(384)),
    "target_readability": 50,
    "history": []
}






    
    



