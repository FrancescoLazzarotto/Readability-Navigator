from utils.io_utils import load_pickle

path = (r"C:\Users\checc\OneDrive\Desktop\Intelligenza Artificiale\Esercizi\modeltree.pkcls")

try:
    emb = load_pickle(path)
    print("ok")
except:
    FileNotFoundError
    
    