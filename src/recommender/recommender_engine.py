import os
import pandas as pd
import numpy as np 
from user.model_user import user_model
from utils.io_utils import load_yamal, load_json, save_json, load_pickle
from sklearn.metrics.pairwise import cosine_similarity
from utils.config_loader import load_config 



class RecommenderEngine():
    def __init__(self, df, embedding, config, user_id, profile_path, flusch_ease):
        self.df = df
        self.embedding = embedding
        #self.config = config
        self.user_id = user_id
        self.profile_path = profile_path
        self.flusch = flusch_ease
    
    def yamal(self):
        config = load_config()
        return config 

    def profile(self):
        if os.path.exists(self.profile_path):
            return load_json(self.profile_path)
        chosen_cluster = self.df["topic_cluster"].unique().tolist()[:3]
        profile = user_model(self.user_id, chosen_cluster, self.df, self.embedding)
        save_json(profile, self.profile_path)
        return profile

    def catalog(self, profile):
        tol = self.config["tol"]
        target = profile["target_readability"]
        history = set(profile["history"])
        df = self.df[~self.df["id"].isin(history)]
        df = df[np.abs(df["flesch_score"] - target) <= tol]
        return df
    
    def get_document(self, doc_id):
        idx = self.df.index[self.df["id"] == doc_id][0]
        testo = self.df.loc[idx, "testo"]
        emb = self.embedding[idx]
        return testo, emb

    def get_flesch(self, doc_id):
         idx = self.df.index[self.df["id"] == doc_id[0]]
         flesch = self.df.loc[idx, "flesch_score"]
         return flesch
         
    def gap_readability(self, user, flesch):
        target_readability = user['target_readability']
        readability = flesch 
        gap = abs(target_readability - readability) 
        return gap, target_readability, readability 
    
    def penality(self):
        if self.target_readability < self.readability:
            score_penality = True
        elif self.target_readability > self.readability:
            score_penality = False 
        return score_penality
        
    def theme_similarity(self, user, doc_id):
        topic_vector = np.array(user['topic_vector']).reshape(1, -1)
        _, emb = self.get_document(doc_id)
        emb = emb.reshape(1, -1)
        sim_score = cosine_similarity(topic_vector, emb)[0][0]
        return sim_score
        
    def recommender(self, user, doc_id, flesch):
        config = self.yamal()
        sim = self.theme_similarity(user, doc_id)
        gap = self.gap_readability5(user, flesch)
        k = config['k']
        eta = config['eta']
        zeta = config['zeta']
        score = eta * sim - zeta  * gap
        return score         


        

    

    
    
    
