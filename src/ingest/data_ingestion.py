import os
import pandas as pd


levels = {
    "Ele-Txt": "easy",
    "Int-Txt": "medium",
    "Adv-Txt": "hard"
}

base_dir = r"C:\Users\checc\OneDrive\Desktop\readability-navigator\data\raw\Texts-SeparatedByReadingLevel"


articles = []

for level_code, level_name in levels.items():
    folder = os.path.join(base_dir, level_code)
    if not os.path.exists(folder):
        print(f" Cartella mancante: {folder}")
        continue

    for file in os.listdir(folder):
        if file.endswith(".txt"):
            title = file.replace(".txt", "").replace("_", " ").title()
            with open(os.path.join(folder, file), "r", encoding="utf-8") as f:
                text = f.read().strip()

            articles.append({
                "id": f"{title}_{level_name}",
                "titolo": title,
                "livello": level_name,
                "testo": text,
                "lingua": "en"
            })


df = pd.DataFrame(articles)

os.makedirs("./data/interim", exist_ok=True)
output_path = "data/interim/onestop_texts.csv"
df.to_csv(output_path, index=False, encoding="utf-8")

print(f" Salvati {len(df)} articoli dal OneStopEnglish Corpus")
print(f"File: {output_path}")
print(df.head())
