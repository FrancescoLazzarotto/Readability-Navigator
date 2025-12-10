import nltk
import pandas as pd
import os 
from nltk.tokenize import sent_tokenize, word_tokenize

input_df = "./ingest/data/interim/onestop_texts.csv"
dataframe = pd.read_csv(input_df)
dataframe.head()


def count_syllables(word):
    """Conteggio sillabe in una parola
    
    Args: 
        word (str): parola di cui conteggiare le sillabe
    
    Returns:
        int: numero sillabe
    """
    vowels = "aeiouy"
    word = word.lower().strip()
    syllables = 0
    prev_char_was_vowel = False

    for char in word:
        if char in vowels:
            if not prev_char_was_vowel:
                syllables += 1
                prev_char_was_vowel = True
        else:
            prev_char_was_vowel = False

    if word.endswith("e") and syllables > 1:
        syllables -= 1
    if syllables == 0:
        syllables = 1

    return syllables

def flesch_ease_reading(num_words, num_sentences, num_syllables):
    """Calcolo flesch ease reading score di un testo
    
    Args:
        num_words (int): numero di parole nel testo
        num_sentences (int): numero di frasi nel testo
        num_syllables (int): numero di sillabe nel testo
    
    Returns:
        float: punteggio flesch ease reading arrotondato a due decimali 
                (valore più alto indica una maggiore facilità di lettura)
    
    """
    if num_sentences == 0 or num_words == 0:
        return 0
    res = 206.835 - 1.015 * (num_words / num_sentences) - 84.6 * (num_syllables / num_words)
    return round(res, 2)


def preprocessing(text):
    """Calcola statistiche di base e punteggio di leggibilità su un testo

    Args:
        text (str): testo da analizzare

    Returns:
        pd.Series: Serie Pandas contenente le seguenti informazioni:
            num_sentences (int): numero di frasi nel testo
            num_words (int): numero di parole alfabetiche nel testo
            avg_sentence_length (float): lunghezza media delle frasi in numero di parole
            avg_word_length (float): lunghezza media delle parole
            long_words (list of str): lista delle parole con più di 6 lettere
            perc_long_words (float): percentuale di parole lunghe sul totale
            flesch_score (float): punteggio Flesch Reading Ease del testo
    """
    sentences = sent_tokenize(text)
    words = word_tokenize(text)
    
    word_alphabetic = []
    for word in words:
        if word.isalpha():
            word_alphabetic.append(word)
            
    num_sentences = len(sentences)
    num_words = len(word_alphabetic)
           
    if num_sentences > 0:
        avg_sentence_length = num_words / num_sentences
    else:
        avg_sentence_length = 0
        
    total_length = 0
    for word in word_alphabetic:
            total_length += len(word)
        
    if num_words > 0:
        avg_word_length = total_length / num_words
    else:
        avg_word_length = 0
        
    
    long_words = []
    for word in word_alphabetic:
        if len(word) > 6:
            long_words.append(word)
             
    if num_words > 0:
        perc_long_words = (len(long_words) / num_words) * 100
    else: 
        perc_long_words = 0
    
    num_syllables = 0
    for word in word_alphabetic:
        num_syllables += count_syllables(word)
    
    flesch_score = flesch_ease_reading(num_words, num_sentences, num_syllables)
    
    return pd.Series({
        "num_sentences": num_sentences,
        "num_words": num_words,
        "avg_sentence_lenght": avg_sentence_length,
        "avg_word_lenght": avg_word_length,
        "long_words": long_words,
        "perc_long_words": perc_long_words,
        "flesch_score": flesch_score
    })

"""
text_length (numero di parole)
– avg_sentence_length
– avg_word_length
– lexical_density
"""
    
"""
features = dataframe["testo"].apply(preprocessing)
dataframe = pd.concat([dataframe, features], axis=1)

os.makedirs("data/processed", exist_ok=True)
dataframe.to_csv("data/processed/onestop_nltk_features.csv", index=False, encoding="utf-8")


dataframe.head()

"""


