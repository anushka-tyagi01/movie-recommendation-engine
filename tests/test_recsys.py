"""Tests on a tiny synthetic MovieLens-format dataset (no real data needed).   pytest -q"""
import numpy as np
import pandas as pd
import pytest

from src.recsys import Recommender, build_artifacts


@pytest.fixture(scope="module")
def rec(tmp_path_factory):
    rng = np.random.default_rng(0)
    root = tmp_path_factory.mktemp("ml")
    genres = ["Comedy", "Drama", "Horror|Thriller", "Animation|Children's", "Action|Sci-Fi", "Romance"]
    movies = [f"{i}::Movie {i} ({1980 + i % 20})::{genres[i % len(genres)]}" for i in range(1, 61)]
    users = [f"{u}::{'F' if u % 2 else 'M'}::{[1, 18, 25, 35][u % 4]}::{u % 20}::12345" for u in range(1, 81)]
    rows = []
    for u in range(1, 81):
        for m in rng.choice(np.arange(1, 61), 25, replace=False):
            rows.append(f"{u}::{m}::{rng.integers(1, 6)}::{960000000 + int(rng.integers(0, 10**6))}")
    (root / "movies.dat").write_text("\n".join(movies), encoding="utf-8")
    (root / "users.dat").write_text("\n".join(users), encoding="utf-8")
    (root / "ratings.dat").write_text("\n".join(rows), encoding="utf-8")
    build_artifacts(root, root / "art", verbose=False)
    return Recommender(root / "art")


def test_seen_movies_are_never_recommended(rec):
    ratings = {1: 5, 2: 5, 3: 1, 4: 3}
    out = rec.recommend(ratings, "F", 25, n=10)
    assert len(out) == 10 and not set(out.movie_id) & set(ratings)


def test_works_with_zero_ratings_and_without_demographics(rec):
    assert len(rec.recommend({}, None, None, n=5)) == 5
    assert (rec.recommend({}, None, None, n=5).why == "Popular overall").all()


def test_deterministic(rec):
    a = rec.recommend({1: 5, 7: 4}, "M", 18).movie_id.tolist()
    assert a == rec.recommend({1: 5, 7: 4}, "M", 18).movie_id.tolist()


def test_unknown_movie_ids_are_ignored(rec):
    assert len(rec.recommend({999999: 5}, "F", 25, n=3)) == 3


def test_lighter_makes_results_lighter_on_average(rec):
    light = rec.lightness()
    plain, lighter = [], []
    for mid in rec.item_ids[:30]:
        plain.append(light[[rec.col[m] for m in rec.similar(mid, "", n=5)[0].movie_id]].mean())
        lighter.append(light[[rec.col[m] for m in rec.similar(mid, "lighter", n=5)[0].movie_id]].mean())
    assert np.mean(lighter) > np.mean(plain)


def test_modifier_parser_reports_unknown_words(rec):
    _, labels, ignored = rec.parse_modifiers("lighter banana")
    assert labels and ignored == ["banana"]
