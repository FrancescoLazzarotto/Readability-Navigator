import numpy as np 

def user_model(user_id, chosen_cluster, df, embedding, default_readability = 60):
    
    embedding  = []
    
    for cluster in chosen_cluster:
        id = df[df["topic_cluster"] == cluster].index
        emb = np.mean(embedding[id], axis= 0)
        embedding.append(emb)
    
    topic_vector = np.mean(embedding, axis=0)
     
    profile = {
        "user_id": user_id,
        "topic_vector": topic_vector.tolist(),
        "target_readability": default_readability,
        "chosen_topics": chosen_cluster,
        "history": {}
    }
    
    return profile


