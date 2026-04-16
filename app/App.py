import os
import sys

import pandas as pd
import plotly.express as px
import streamlit as st

CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(CURRENT_DIR, ".."))
sys.path.insert(0, PROJECT_ROOT)

from main import main
from src.user.model_user import build_user_model, load_user_model, update_user_model
from utils.data_loader import load_features_df
from utils.io_utils import load_yaml


st.set_page_config(
    page_title="Readability Navigator Demo",
    page_icon="📚",
    layout="wide",
)


def apply_styles() -> None:
    st.markdown(
        """
        <style>
        @import url('https://fonts.googleapis.com/css2?family=Space+Grotesk:wght@400;500;700&family=Fraunces:opsz,wght@9..144,500;9..144,700&display=swap');

        :root {
            --ink: #1f2933;
            --muted: #52606d;
            --sand: #f9f5ee;
            --card: #fffdf8;
            --teal: #0f766e;
            --coral: #e76f51;
            --border: #e8dece;
        }

        html, body, [class*="css"] {
            font-family: 'Space Grotesk', sans-serif;
            color: var(--ink);
        }

        .stApp {
            background:
                radial-gradient(circle at 15% 15%, rgba(231, 111, 81, 0.12), transparent 40%),
                radial-gradient(circle at 85% 0%, rgba(15, 118, 110, 0.13), transparent 34%),
                linear-gradient(180deg, #f7efe1 0%, #f5f3ef 50%, #f8f7f4 100%);
        }

        .block-container {
            padding-top: 1.6rem;
            padding-bottom: 2.3rem;
            max-width: 1260px;
        }

        h1, h2, h3 {
            font-family: 'Fraunces', serif;
            color: #152238;
            letter-spacing: 0.2px;
        }

        .hero {
            border: 1px solid var(--border);
            background: linear-gradient(120deg, rgba(255,255,255,0.82), rgba(255,247,236,0.92));
            border-radius: 22px;
            padding: 1.3rem 1.4rem;
            box-shadow: 0 10px 24px rgba(30, 41, 59, 0.08);
            margin-bottom: 0.8rem;
        }

        .hero p {
            margin: 0.1rem 0;
            color: var(--muted);
            font-size: 1.02rem;
        }

        .mini-card {
            border: 1px solid var(--border);
            background: var(--card);
            border-radius: 16px;
            padding: 0.75rem 0.9rem;
            margin-bottom: 0.6rem;
            box-shadow: 0 4px 14px rgba(15, 23, 42, 0.05);
        }

        .mini-title {
            font-size: 0.82rem;
            color: var(--muted);
            margin-bottom: 0.18rem;
            text-transform: uppercase;
            letter-spacing: 0.05em;
        }

        .mini-value {
            font-size: 1.1rem;
            font-weight: 700;
            color: var(--ink);
        }

        .rec-card {
            border: 1px solid var(--border);
            background: linear-gradient(180deg, #fffdfa, #fffaf1);
            border-radius: 16px;
            padding: 0.9rem;
            min-height: 140px;
            box-shadow: 0 8px 18px rgba(15, 23, 42, 0.06);
        }

        .rec-rank {
            display: inline-block;
            font-size: 0.72rem;
            font-weight: 700;
            color: white;
            background: linear-gradient(90deg, var(--teal), #14b8a6);
            border-radius: 999px;
            padding: 0.15rem 0.5rem;
            margin-bottom: 0.5rem;
        }

        .rec-title {
            font-weight: 700;
            color: #1b2a41;
            margin-bottom: 0.35rem;
            line-height: 1.25;
        }

        .rec-meta {
            font-size: 0.88rem;
            color: var(--muted);
            margin: 0;
        }

        [data-testid="stSidebar"] {
            background: linear-gradient(180deg, #fff9ef 0%, #fffdf8 100%);
            border-right: 1px solid #eddcc8;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def init_state() -> None:
    st.session_state.setdefault("current_user", None)
    st.session_state.setdefault("last_user_id", 1)
    st.session_state.setdefault("recommendations_df", None)
    st.session_state.setdefault("selected_doc", None)


@st.cache_data(show_spinner=False)
def get_dataset() -> pd.DataFrame:
    dataset = load_features_df().copy()
    dataset["id"] = dataset["id"].astype(str)
    return dataset


def get_existing_user_ids(users_path: str) -> list[int]:
    user_ids = []
    for filename in os.listdir(users_path):
        if not (filename.startswith("user") and filename.endswith(".json")):
            continue
        try:
            user_ids.append(int(filename[4:-5]))
        except ValueError:
            continue
    return sorted(user_ids)


def render_sidebar(current_user: dict | None, dataset: pd.DataFrame) -> None:
    with st.sidebar:
        st.markdown("## Readability Navigator")
        st.caption("Live demo for user modeling and text recommendation")

        st.markdown("### Demo Checklist")
        st.checkbox("User profile ready", value=current_user is not None, disabled=True)
        st.checkbox("Dataset loaded", value=len(dataset) > 0, disabled=True)
        st.checkbox(
            "Recommendations generated",
            value=st.session_state.recommendations_df is not None,
            disabled=True,
        )

        st.markdown("### Session")
        if current_user is None:
            st.info("Activate a user profile in the User tab.")
        else:
            st.success(f"Active user: #{current_user['user_id']}")
            st.caption(f"Readability target: {current_user['target_readability']:.1f}")
            st.caption(f"History length: {len(current_user.get('history', []))}")


def render_hero() -> None:
    st.markdown(
        """
        <div class="hero">
            <h1 style="margin:0 0 0.4rem 0;">Readability Navigator</h1>
            <p>Adaptive recommendation demo that balances semantic relevance and cognitive accessibility.</p>
            <p style="font-size:0.95rem;">Use the tabs below to manage user profiles, inspect the dataset, and run recommendations.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )


def render_user_tab(users_path: str) -> None:
    st.subheader("User Profile")
    left, right = st.columns([1.25, 1], gap="large")

    with left:
        mode = st.radio(
            "Profile mode",
            ["Create new user", "Load existing user"],
            horizontal=True,
        )

        if mode == "Create new user":
            col1, col2 = st.columns(2)
            with col1:
                new_user_id = st.number_input(
                    "User ID",
                    min_value=1,
                    value=int(st.session_state.last_user_id),
                    step=1,
                )
            with col2:
                target_readability = st.slider(
                    "Target readability (Flesch)",
                    min_value=20,
                    max_value=90,
                    value=60,
                    step=1,
                )

            if st.button("Activate new profile", use_container_width=True):
                user = build_user_model(
                    user_id=int(new_user_id),
                    default_readability=float(target_readability),
                    save=True,
                )
                st.session_state.current_user = user
                st.session_state.last_user_id = int(new_user_id)
                st.session_state.recommendations_df = None
                st.session_state.selected_doc = None
                st.success(f"User #{new_user_id} is now active.")

        else:
            existing_ids = get_existing_user_ids(users_path)
            if not existing_ids:
                st.warning("No user profiles found. Create one first.")
            else:
                selected_user_id = st.selectbox("Existing users", existing_ids)
                if st.button("Load selected profile", use_container_width=True):
                    user = load_user_model(f"user{selected_user_id}.json", users_path)
                    st.session_state.current_user = user
                    st.session_state.recommendations_df = None
                    st.session_state.selected_doc = None
                    st.success(f"User #{selected_user_id} loaded.")

    with right:
        user = st.session_state.current_user
        st.markdown("### Active Profile Snapshot")
        if user is None:
            st.info("No active user. Use the controls on the left to activate one.")
        else:
            history_count = len(user.get("history", []))
            target = float(user.get("target_readability", 60))

            st.markdown(
                f"""
                <div class="mini-card">
                    <div class="mini-title">User ID</div>
                    <div class="mini-value">#{user['user_id']}</div>
                </div>
                <div class="mini-card">
                    <div class="mini-title">Target Readability</div>
                    <div class="mini-value">{target:.1f}</div>
                </div>
                <div class="mini-card">
                    <div class="mini-title">Read Documents</div>
                    <div class="mini-value">{history_count}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )

            st.caption("Readability target range: 20 to 90")
            st.progress(int(((target - 20) / 70) * 100))


def render_data_tab(dataset: pd.DataFrame) -> None:
    st.subheader("Dataset Overview")

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Documents", f"{len(dataset):,}")
    m2.metric("Avg Flesch", f"{dataset['flesch_score'].mean():.2f}")
    m3.metric("Min Flesch", f"{dataset['flesch_score'].min():.2f}")
    m4.metric("Max Flesch", f"{dataset['flesch_score'].max():.2f}")

    c1, c2 = st.columns([1.35, 1], gap="large")

    with c1:
        fig = px.histogram(
            dataset,
            x="flesch_score",
            nbins=36,
            title="Readability Distribution",
            color_discrete_sequence=["#0f766e"],
        )
        fig.update_layout(
            margin=dict(l=10, r=10, t=42, b=10),
            plot_bgcolor="rgba(255,255,255,0.55)",
            paper_bgcolor="rgba(0,0,0,0)",
            title_font=dict(size=18),
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        if "livello" in dataset.columns:
            level_counts = (
                dataset["livello"]
                .astype(str)
                .value_counts()
                .reset_index()
                .rename(columns={"index": "level", "livello": "count"})
            )
            level_chart = px.bar(
                level_counts,
                x="level",
                y="count",
                title="Documents by Level",
                color_discrete_sequence=["#e76f51"],
            )
            level_chart.update_layout(
                margin=dict(l=10, r=10, t=42, b=10),
                plot_bgcolor="rgba(255,255,255,0.55)",
                paper_bgcolor="rgba(0,0,0,0)",
            )
            st.plotly_chart(level_chart, use_container_width=True)
        else:
            st.info("Column 'livello' not available in this dataset.")

    st.markdown("### Sample Records")
    rows_to_show = st.slider("Rows to display", 5, 50, 12)
    available_columns = [
        col for col in ["id", "titolo", "livello", "flesch_score"] if col in dataset.columns
    ]
    st.dataframe(dataset[available_columns].head(rows_to_show), use_container_width=True)


def render_top_recommendation_cards(recommendations: pd.DataFrame) -> None:
    top = recommendations.head(3)
    if top.empty:
        return

    cols = st.columns(3)
    for i, (_, row) in enumerate(top.iterrows()):
        with cols[i]:
            st.markdown(
                f"""
                <div class="rec-card">
                    <div class="rec-rank">Top {i + 1}</div>
                    <div class="rec-title">{row['title']}</div>
                    <p class="rec-meta">Score: {float(row['score']):.4f}</p>
                    <p class="rec-meta">Flesch: {float(row['flesch_score']):.2f}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_recommendation_tab() -> None:
    st.subheader("Recommendation Demo")

    user = st.session_state.current_user
    if user is None:
        st.warning("Activate a user profile first in the User tab.")
        return

    col_action, col_info = st.columns([1, 1.2], gap="large")
    with col_action:
        if st.button("Generate Recommendations", use_container_width=True):
            with st.spinner("Computing top recommendations..."):
                st.session_state.recommendations_df = main(user)
                st.session_state.selected_doc = None

    with col_info:
        st.info(
            "This ranking combines semantic similarity with readability alignment "
            "using the hybrid scoring function from the paper."
        )

    recommendations = st.session_state.recommendations_df
    if recommendations is None:
        st.caption("No recommendations generated yet.")
        return

    if recommendations.empty:
        st.warning("No recommendations available for the current profile.")
        return

    st.success(f"Generated {len(recommendations)} recommendations.")
    render_top_recommendation_cards(recommendations)

    st.markdown("### Full Ranking")
    display_cols = [
        col
        for col in [
            "title",
            "score",
            "flesch_score",
            "semantic_similarity",
            "normalized_gap",
            "penalty",
        ]
        if col in recommendations.columns
    ]
    st.dataframe(recommendations[display_cols], use_container_width=True)

    st.markdown("### Read and Give Feedback")
    selected_title = st.selectbox("Select a recommended document", recommendations["title"].tolist())

    if st.button("Open Document", use_container_width=False):
        row = recommendations[recommendations["title"] == selected_title].iloc[0]
        st.session_state.selected_doc = row.to_dict()

    selected_doc = st.session_state.selected_doc
    if selected_doc is None:
        return

    read_col, feedback_col = st.columns([2.2, 1], gap="large")
    with read_col:
        st.markdown(f"#### {selected_doc['title']}")
        st.caption(
            f"Score {float(selected_doc['score']):.4f} • "
            f"Flesch {float(selected_doc['flesch_score']):.2f}"
        )
        with st.expander("Open reading text", expanded=True):
            st.write(selected_doc["testo"])

    with feedback_col:
        difficulty = st.radio(
            "How difficult was this text?",
            options=[1, 2, 3, 4, 5],
            format_func=lambda x: {
                1: "Very easy",
                2: "Easy",
                3: "Balanced",
                4: "Hard",
                5: "Very hard",
            }[x],
        )

        if st.button("Save Feedback", use_container_width=True):
            try:
                update_user_model(
                    st.session_state.current_user,
                    str(selected_doc["title"]),
                    float(selected_doc.get("flesch_score", 60.0)),
                    int(difficulty),
                )
                st.session_state.recommendations_df = None
                st.session_state.selected_doc = None
                st.success("Feedback saved and user profile updated.")
                st.rerun()
            except Exception as exc:
                st.error(f"Could not save feedback: {exc}")


def run() -> None:
    apply_styles()
    init_state()

    config = load_yaml()
    users_path = os.path.join(PROJECT_ROOT, config["paths"]["user_json"])
    os.makedirs(users_path, exist_ok=True)

    dataset = get_dataset()
    render_sidebar(st.session_state.current_user, dataset)
    render_hero()

    tab_user, tab_data, tab_reco = st.tabs(
        ["User", "Data", "Recommendations"]
    )

    with tab_user:
        render_user_tab(users_path)
    with tab_data:
        render_data_tab(dataset)
    with tab_reco:
        render_recommendation_tab()


run()