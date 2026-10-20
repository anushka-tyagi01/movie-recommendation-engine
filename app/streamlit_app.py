"""Demo app.   streamlit run app/streamlit_app.py   (run from the repository root)"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd
import streamlit as st

from src.recsys import AGE_LABELS, POP_PENALTY, Recommender

ARTIFACTS = ROOT / "artifacts"
st.set_page_config(page_title="Movie Recommendation Engine", page_icon="🎬", layout="wide")

if not (ARTIFACTS / "model.npz").exists():
    st.error("Model files not found. Build them once with:")
    st.code("python -m src.build_artifacts --data data/raw/ml-1m --out artifacts")
    st.stop()


@st.cache_resource(show_spinner="Loading the model ...")
def load_model() -> Recommender:
    return Recommender(ARTIFACTS)


rec = load_model()
by_popularity = [int(rec.item_ids[i]) for i in sorted(range(rec.n_items), key=lambda i: -rec.n_pop[i])]   # popular movies first in the pickers
title_of = {int(m): rec.titles[i] for i, m in enumerate(rec.item_ids)}

st.title("🎬 Movie Recommendation Engine")
st.caption("MovieLens 1M (ratings from 2000 to 2003, so no recent films). Every recommendation comes with a reason.")
tab_me, tab_like, tab_how = st.tabs(["Recommend for me", "Like this movie, but…", "How it works"])

# ------------------------------------------------------------------------------------------------ tab 1
with tab_me:
    left, right = st.columns([1, 2.2], gap="large")
    with left:
        st.subheader("About you")
        age_label = st.selectbox("Age group", ["Prefer not to say"] + list(AGE_LABELS.values()), key="age")
        gender_label = st.radio("Gender", ["Prefer not to say", "Female", "Male"], horizontal=True, key="gender")
        liked = st.multiselect("Movies you liked", by_popularity, format_func=title_of.get, key="liked",
                               placeholder="Type to search, e.g. Fargo")
        disliked = st.multiselect("Movies you disliked", by_popularity, format_func=title_of.get, key="disliked",
                                  placeholder="Optional")
        gamma = st.slider("Push away from over-popular movies", 0.0, 1.25, POP_PENALTY, 0.25, key="gamma",
                          help="0 = the raw model, which mostly shows the most-rated films. Higher = more variety.")
    with right:
        st.subheader("Your recommendations")
        age_code = {v: k for k, v in AGE_LABELS.items()}.get(age_label)
        gender = {"Female": "F", "Male": "M"}.get(gender_label)
        ratings = {**{m: 5 for m in liked}, **{m: 1 for m in disliked if m not in liked}}
        out = rec.recommend(ratings, gender, age_code, n=10, gamma=gamma)
        st.caption(rec.recommendation_mode(len(ratings)))
        show = out.assign(year=out.year.astype("Int64").astype(str))[["title", "genres", "why"]]
        show.columns = ["Movie", "Genres", "Why"]
        st.dataframe(show, hide_index=True, width="stretch", height=390,
                     column_config={"Movie": st.column_config.TextColumn(width="medium"), "Genres": st.column_config.TextColumn(width="medium"),
                                    "Why": st.column_config.TextColumn(width="large")})
        if not ratings:
            st.info("Pick a few movies you liked and the list will switch from 'popular with people like you' to personal picks.")

# ------------------------------------------------------------------------------------------------ tab 2
with tab_like:
    c1, c2 = st.columns([1, 2.2], gap="large")
    with c1:
        st.subheader("Start from a movie")
        anchor = st.selectbox("Movie", by_popularity, format_func=title_of.get, key="anchor",
                              index=by_popularity.index(rec.search("Matrix, The", 1)[0][0]))
        text = st.text_input("…but", value="lighter and more recent", key="modifier",
                             help="Try: lighter, darker, funnier, scarier, newer, older, more obscure, family, more comedy, sci-fi")
        st.caption("Examples: *lighter* · *darker and older* · *funnier and more obscure* · *more romance*")
    with c2:
        st.subheader("Results")
        df, understood, ignored = rec.similar(anchor, text, n=10)
        if understood:
            st.success("Understood: " + "; ".join(understood))
        elif text.strip():
            st.warning("I did not recognise any modifier, so this is plain 'people who liked it also liked'.")
        if ignored and understood:
            st.caption("Ignored words: " + ", ".join(ignored))
        show = df.assign(year=df.year.astype("Int64").astype(str))[["title", "genres", "audience similarity rank"]]
        show.columns = ["Movie", "Genres", "Audience-similarity rank (1 = closest)"]
        st.dataframe(show, hide_index=True, width="stretch", height=390,
                     column_config={"Movie": st.column_config.TextColumn(width="medium"), "Genres": st.column_config.TextColumn(width="medium")})

# ------------------------------------------------------------------------------------------------ tab 3
with tab_how:
    st.markdown("""
**Models.** Item-based nearest neighbours (people who liked A also liked B) and an implicit-feedback ALS written from
scratch, combined as `z(kNN) + 0.3 × z(ALS)`. A popularity penalty keeps the list from collapsing onto the most-rated films.

**New users.** With no ratings the list is what people of your age and gender group like. As you add ratings, that prior
fades with weight `1 / (1 + k / 5)` and the collaborative models take over (they overtake demographics at about 10 ratings).

**Why this movie?** The reason shown is the liked movies that contribute most to the item-kNN score. In a test, removing the
cited movie pushed a recommendation out of the top 10 about 5 times more often than removing a random liked movie.

**Like X but…** The nearest neighbours of a movie are re-ranked with plain keyword modifiers (genre- and year-based). It is
rule-based and says which words it understood. It does not call an LLM.

**Honest limits.** The data stops in 2003. Genres are the only content signal, so "lighter" means more comedy and animation and
less horror, thriller and war, not a judgement about tone. Quality figures are in the project README and notebook.
""")
