import json
import pandas as pd
import os
import pickle
import yaml


def load_yamal():
    with open('C:\Users\checc\Readability-Navigator\conf\project.yaml', 'r') as file:
        return yaml.safe_load(file)

def load_json(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)

def save_json(data, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4, ensure_ascii=False)

def load_csv(path):
    return pd.read_csv(path, encoding="utf-8")

def save_csv(df, path):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    df.to_csv(path, index=False, encoding="utf-8")

def save_pickle(name_file, data):
    with open(name_file, "wb") as pkl:
        pickle.dump(data, pkl)

def load_pickle(name_file):
    with open(name_file, 'rb') as pkl:
        data = pickle.load(pkl)
    return data
        
