# Readability Navigator

Readability Navigator is a personalized text recommendation project that suggests the next best document based on user interests and reading difficulty.

The system combines:
- symbolic readability metrics (Flesch Reading Ease)
- semantic embeddings (SBERT, 384 dimensions)
- iterative user-profile updates driven by feedback

The goal is not to simplify text automatically, but to select the most suitable next text for each user.

Recommendations balance:
- semantic relevance to user interests
- distance from the user readability target

## How It Works

Pipeline:
1. Load engineered features and document embeddings.
2. Load or create a user profile (topic_vector, target_readability, history).
3. Build a candidate catalog:
- remove already read documents
- keep documents within readability tolerance
4. Compute hybrid score:

$$
score = \eta \cdot similarity - \zeta \cdot gap_{penalized}
$$

The readability gap is dynamically penalized when a text is above the user target.

5. Rank documents and return Top-K.
6. Collect difficulty feedback (1-5) and update:
- reading history
- topic vector
- target readability

## Repository Structure

- app/: Streamlit dashboard and presentation pages
- src/recommender/: ranking and recommendation engine
- src/user/: user profile creation and update logic
- src/features/: preprocessing and embeddings
- src/eval/: offline evaluation (NDCG)
- utils/: loading and I/O utilities
- conf/project.yaml: core parameters and paths
- data/: processed datasets and user JSON profiles

## Quick Start

Prerequisites:
- Python 3.10+
- pip

Install dependencies:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

Download required NLTK resource:

```bash
python -c "import nltk; nltk.download('punkt')"
```

Run the Streamlit app from project root:

```bash
streamlit run app/App.py
```

## Developer Testing Setup

The requirements file includes both runtime and testing dependencies.

Recommended validation workflow:
1. Smoke check on processed data:

```bash
python src/test/test.py
```

2. Offline recommender evaluation:

```bash
python src/eval/evaluation.py
```

3. Run unit/integration test suite (when tests are added/extended):

```bash
pytest -q
```

## Run From Python

Minimal example using main.py:

```python
from main import main

user = {
    "user_id": 1,
    "target_readability": 60,
    "topic_vector": [0.0] * 384,
    "history": []
}

ranked_df = main(user)
print(ranked_df.head())
```

## Data Assets

Primary dataset used in this repository: OneStopEnglish (processed version).

Expected local assets:
- data/interim/onestop_texts.csv
- data/processed/onestop_nltk_features.csv
- src/features/doc_embedding.pickle

## Configuration Notes

- Main model parameters are in conf/project.yaml.
- User profiles are saved in data/user/json_file/.
- Some scripts in src/ingest and src/features are intended for experimentation in addition to app runtime.

## Author

Francesco Lazzarotto


