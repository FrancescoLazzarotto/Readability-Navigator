# Readability-Navigator-Adaptive-Text-Recommender-System-for-Students-with-Learning-Disabilities
An intelligent recommender system that personalizes reading materials for students with dyslexia or learning difficulties. It automatically measures text readability, matches topics with user interests, and gradually adapts reading difficulty to optimize cognitive load and learning progress

# 🧠 Readability-Navigator  
### Adaptive Text Recommender System for Students with Learning Disabilities

---

## 📖 Overview

**Readability-Navigator** is an academic project developed at the **University of Turin** for the course *“Web Sicuro e Personalizzato”*.  
The goal is to support students with **dyslexia or reading difficulties (DSA)** through a **personalized recommender system** that suggests texts matching both their **interests** and **reading ability**.

Instead of simplifying text, the system **selects the next best document** whose topic and difficulty are optimal for the user's cognitive level.  
It monitors progress, avoids overload, and adapts automatically to help users improve reading comprehension step by step.

---

## 🎯 Objectives

- Estimate **text readability** using linguistic metrics (e.g., Flesch, Gulpease).  
- Build a **user profile** that includes reading level and interests.  
- Recommend the “right text” balancing:
  - **Semantic similarity** between text and interests  
  - **Readability distance** from the user’s current level  
- Collect **feedback** (time, completion, difficulty) to adjust the next recommendation.  
- Provide an **accessible interface** (font for dyslexia, high contrast, TTS option).

---

## 🧱 System Architecture

User → Profile (topic_vector, readability_target)
→ Retrieve texts by topic similarity
→ Filter by readability tolerance
→ Score = η·semantic_similarity − ζ·|readability − target|
→ Rank and recommend Top-K
→ Collect feedback (time, completion)
→ Update user profile (new target + updated interests)

yaml
Copia codice

---

## ⚙️ Technologies Used

| Component | Tool |
|------------|------|
| Language | Python 3.10+ |
| NLP & Readability | `textstat`, `spacy`, `nltk` |
| Semantic Embeddings | `sentence-transformers` (SBERT) |
| Similarity & Ranking | `scikit-learn`, cosine similarity |
| Interface (optional) | `Streamlit` or `Flask` |
| Evaluation | `numpy`, `pandas`, `matplotlib` |

---

## 🧩 Project Structure

readability-navigator/
│
├── data/ # datasets: raw, cleaned, processed
│ ├── raw/
│ ├── interim/
│ └── processed/
│
├── src/ # project code
│ ├── ingest/ # data collection and parsing
│ ├── features/ # readability metrics & embeddings
│ ├── catalog/ # text database
│ ├── users/ # user modeling
│ ├── recommender/ # recommendation logic
│ └── feedback/ # feedback processing
│
├── notebooks/ # analysis and experiments
├── conf/ # configuration files (YAML)
├── outputs/ # results, figures, logs
├── app/ # demo web interface
│
├── README.md
├── requirements.txt
└── LICENSE

less
Copia codice

---

## 🧮 Datasets

| Dataset | Description | Source |
|----------|--------------|--------|
| **Simple English Wikipedia** | Simplified articles | [simple.wikipedia.org](https://simple.wikipedia.org) |
| **Standard Wikipedia** | Reference full-text articles | [wikipedia.org](https://wikipedia.org) |
| **ASSET** | Sentence-level simplifications | [HuggingFace](https://huggingface.co/datasets/asset) |
| **OneStopEnglish** | Texts at multiple difficulty levels | [GitHub](https://github.com/nishkalavallabhi/OneStopEnglishCorpus) |

---

## 🧠 How It Works

1. **Data preprocessing**  
   Clean and tokenize texts, remove markup, and calculate readability scores.

2. **Feature extraction**  
   Compute text readability (e.g., Flesch, Gulpease) and semantic embeddings (SBERT).

3. **User modeling**  
   - Collect user interests and reading test.  
   - Estimate initial readability target.  
   - Store as `user_profile.json` with embedding and target.

4. **Recommendation**  
   Retrieve top-N texts by semantic similarity,  
   filter those near the target difficulty,  
   and compute:
   \[
   score(u,d) = η·sim_{topic}(u,d) − ζ·|readability(d) − target_u|
   \]
   Then recommend the Top-K.

5. **Feedback loop**  
   Measure reading time, completion, or skips.  
   Adjust target difficulty:
   - +Δ if easy  
   - −Δ if difficult or abandoned.  
   Update interests over time.

6. **Evaluation**  
   Assess with metrics: NDCG@k, Target-Deviation@k, and Completion Rate.

---

## 📊 Evaluation Metrics

| Metric | Purpose |
|---------|----------|
| **NDCG@k** | Measures ranking quality |
| **Target-Deviation@k** | Measures how close recommendations match user level |
| **Completion Rate** | Engagement & suitability |
| **Calibration** | Balance between topic and difficulty |
| **Novelty** | Diversity across sessions |

---

## 🧰 Installation

```bash
# clone repository
git clone https://github.com/<your-username>/readability-navigator.git
cd readability-navigator

# create environment
python -m venv venv
source venv/bin/activate  # or venv\Scripts\activate on Windows

# install dependencies
pip install -r requirements.txt
🚀 Run the Demo
bash
Copia codice
# start local web demo
streamlit run app/demo.py
Demo features:

Choose a topic (e.g., science, history, technology)

Get 3–5 recommended texts

Rate them (“Too Easy”, “OK”, “Too Hard”)

System adapts next suggestions automatically

📈 Example Results (Pilot Test)
Metric	Value	Comment
Precision@3	0.82	Good relevance
Target-Deviation	±4.0	Close to user reading level
Completion Rate	0.77	Strong engagement
Avg. Level Growth	+5 points	Shows cognitive improvement

🧩 Future Work
Reinforcement Learning for adaptive difficulty tuning.

Multilingual support (Italian and English).

Integration with eye-tracking for cognitive feedback.

Explainable recommendations (“why this text”).

Personalized interface (TTS, spacing, font adjustments).

👥 Authors
Francesco Lazzarotto
Department of Computer Science – University of Turin
Course: Web Sicuro e Personalizzato
Academic Year: 2025

🧠 Keywords
recommender-system · machine-learning · AI · nlp ·
education · accessibility · dyslexia · python · learning

📜 License
Released under the MIT License.
See the LICENSE file for details.
