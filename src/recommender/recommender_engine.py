import os
import pandas as pd
import numpy as np 
from src.user.model_user import user_model
from utils.io_utils import load_json
from sklearn.metrics.pairwise import cosine_similarity



class RecommenderEngine():
    """Classe per il motore di raccomandazione dei contenuti

    questa classe gestisce i dati, gli embeddings dei documenti e il profilo utente
    per generare raccomandazioni personalizzate
    """
    def __init__(self, df, embedding, config, user_id, profile_path):
        """Inizializza il motore di raccomandazione con i dati e le configurazioni

        Args:
            df (pd.DataFrame): DataFrame dei contenuti
            embedding (object): embeddings dei documenti
            config (dict): parametri di configurazione
            user_id (str): identificativo dell'utente corrente
            profile_path (str): percorso dei profili utente
        """
        self.df = df
        self.embedding = embedding
        self.config = config
        self.user_id = user_id
        self.profile_path = profile_path

        
    
    def profile(self):
        """Carica il file json relativo a un utente se trovato
        
        Returns:
            Dict or None: dati dell'utente se esiste oppure None    
        """
        if self.profile_path and os.path.exists(self.profile_path):
                return load_json(self.profile_path)

        return None
    
    
      

    def catalog(self, profile):
        """Creazione catalogo utente.
        Filtra i contenuti già visti dall'utente e seleziona quelli
        con punteggio di leggibilità vicino al target dell'utente.
        
        Args:
            profile (dict): dati utente
            selected_topic (str, optional): topic selezionato dall'utente
        
        Returns:
            pd.DataFrame: catalogo filtrato dei contenuti
        """
        tol = self.config["tol"]
        target = profile["target_readability"]
        history = set(profile["history"])
        df = self.df[~self.df["id"].isin(history)]
        df = df[np.abs(df["flesch_score"] - target) <= tol]
          
        return df

    
    
    
    
    def get_document(self, doc_id):
        """Prendere testo ed embedding di un testo dato il suo id
        
        Args:
            doc_id (int): numero identificatore del documento
        
        Returns:
            tuple [str, list[list[float]]]: 
                -testo del documento
                -embedding del testo sottoforma di lista di vettori
        """
        idx = self.df.index[self.df["id"] == doc_id][0]
        testo = self.df.loc[idx, "testo"]
        emb = self.embedding[idx]
        return testo, emb
    

    def get_flesch(self, doc_id):
        """Prendere il punteggio flesch di un testo dato il suo id
        
        Args:
            doc_id (int): numero identificatore del documento
            
        Returns:
            float: punteggio ease reading di un documento
        
        Raises:
            ValueError: se il valore dell'id del documento passato come parametro non coincide/esiste nel file
        """
        idx = self.df.index[self.df["id"] == doc_id]
        if len(idx) == 0:
            raise ValueError("Documento non trovato")
        return float(self.df.loc[idx[0], "flesch_score"])
    

    def gap_readability(self, user, flesch):
        """Calcolo gap di leggibilità fra quella dell'utente e punteggio flesch del testo
        
        Args:
            user(dict): dizionario contenente dati dell'utente fra cui target readability
            flesch (float): il punteggio flesch ease reading del testo
            
        Returns:
            tuple[float, float, float]:
                -gap fra target readability e flesch score
                -target readability
                -flesch score
        """
        target = user['target_readability']
        gap = abs(target - flesch)
        return gap, target, flesch


    def penalty(self, target, readability, alpha):
        """Calcolo penalità da applicare al testo in base alla leggibilità
            la penalità aumenta di `alpha` se la leggibilità del testo supera il target dell'utente
        
        Args:
            target(float): target readability dell'utente
            readability(float): readability del testo
            alpha(float): fattore di penalità da aggiungere se il testo è troppo leggibile
        
        Returns:
            float: fattore di penalità da applicare al testo (1 o 1 + alpha)
        """
        if readability > target:
            return 1 + alpha
        else:
            return 1
        
        
    def theme_similarity(self, user, doc_id):
        """
        Calcola la similarità tematica tra un utente e un documento

        La similarità viene calcolata come coseno tra il vettore tematico
        dell'utente e l'embedding del documento

        Args:
            user (dict): dizionario contenente i dati dell'utente, tra cui 
                        'topic_vector', il vettore tematico dell'utente
            doc_id (int): identificativo  del documento da confrontare

        Returns:
            float: punteggio di similarità tematica compreso tra -1 e 1
                valori più alti indicano maggiore similarità
        """
        topic_vector = np.array(user['topic_vector']).reshape(1, -1)
        _, emb = self.get_document(doc_id)
        emb = emb.reshape(1, -1)
        sim_score = cosine_similarity(topic_vector, emb)[0][0]
        return sim_score
    
        
    def recommender(self, user, doc_id):
        """Calcola il punteggio di raccomandazione di un documento per un utente.

            Il punteggio combina la similarità tematica con l'adeguatezza della
            leggibilità rispetto al target dell'utente, applicando una penalità
            se necessario

        Args:
            user (dict): dizionario contenente i dati dell'utente
            doc_id (int or str): identificativo del documento da valutare.

        Returns:
            float: punteggio di raccomandazione del documento per l'utente.
                   valori più alti indicano contenuti più raccomandati
        """
        config = self.config
        eta = config['eta']
        zeta = config['zeta']
        alpha = config['alpha']
        
        flesch = self.get_flesch(doc_id)
        sim = self.theme_similarity(user, doc_id)
        gap, target, readability = self.gap_readability(user, flesch)
        
        penalty_score = self.penalty(target, readability, alpha)
        gap_penalized = gap * penalty_score
        
        score = eta * sim - zeta  * gap_penalized
        return score         


    def rank_top_k(self, user):
        """Raccomandare e classificare i top k documenti 
        
        Args:
            user(dict): dizionario contenente i dati dell'utente
            
        Returns:
            tuple[list[str], list[float], list[str]]: 
            - lista degli ID dei documenti migliori
            - lista dei punteggi di raccomandazione corrispondenti, arrotondati a 6 decimali
            - lista dei testi dei documenti raccomandati
        """
        config = self.config
        k = config['k']
        profile = user
        catalog = self.catalog(profile)
        
        scores = []
        titles = []
        testi = []

        
        for doc_id in catalog['id'].tolist():
            score = self.recommender(user, doc_id)
            scores.append((doc_id, score))
                               
        scores.sort(key=lambda x: x[1], reverse=True)
        top_scores = scores[:k]
        titles = [item[0] for item in top_scores]
        scores_only = [item[1] for item in top_scores]
        scores_only = np.round(scores_only, 6)
        
        for doc_id in titles:
            testo,_ = self.get_document(doc_id)
            testi.append(testo)
        
        
        return titles, scores_only, testi
    
    
    
    def rank_to_df(self, user):
        """Convertire la classificazione dei documenti raccomandati in un pandas.DataFrame
        
        Args:
            user(dict): dizionario contenente i dati dell'utente

        Returns:
            pandas.DataFrame:
                -title(str): id dei documenti       
                -score(float): punteggio di raccomandazione
                -testo(str): testo dei documenti
        """
        df = pd.DataFrame(dict(
            title = list(),
            score = list(),
            testo = list()
        ))
        
        title, score, testo = self.rank_top_k(user)
        
        df['title'] = title
        df['score'] = score
        df['testo'] = testo
        
        return df
    
        
        
        
        
        

        

    

    
    
    
