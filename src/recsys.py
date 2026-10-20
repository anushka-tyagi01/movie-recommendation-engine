"""Movie recommendation engine: model building and serving logic.

The final warm-user model is  hybrid = z(item-kNN) + 0.3 * z(ALS)  minus a popularity penalty.
New users get a demographic prior that fades as ratings arrive, and a "like X but ..." mode
re-ranks the nearest neighbours of a movie with simple, transparent text modifiers.
All numbers (0.3, 1.0, tau = 5, 64 factors, ...) were chosen on the validation split in the notebook.
"""
from __future__ import annotations

import io
import re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

SEED = 42
POS_THRESHOLD = 4                    # rating >= 4 means "liked"
ALS_FACTORS, ALS_REG, ALS_ALPHA, ALS_ALPHA_NEG, ALS_ITERS = 64, 100.0, 15.0, 5.0, 15
HYBRID_ALS_WEIGHT = 0.3              # weight of ALS next to item-kNN (tuned on validation)
PRIOR_TAU = 5.0                      # demographic prior weight = 1 / (1 + k / tau) after k ratings
POP_PENALTY = 1.0                    # popularity penalty gamma (faded in as k / (k + tau))

AGE_LABELS = {1: "Under 18", 18: "18-24", 25: "25-34", 35: "35-44", 45: "45-49", 50: "50-55", 56: "56+"}
GENRES = ["Action", "Adventure", "Animation", "Children's", "Comedy", "Crime", "Documentary", "Drama", "Fantasy",
          "Film-Noir", "Horror", "Musical", "Mystery", "Romance", "Sci-Fi", "Thriller", "War", "Western"]


# ----------------------------------------------------------------------------------------------- data
def read_dat(path: Path, names):
    """Read a '::'-separated MovieLens file (UTF-8 first, latin-1 as a fallback)."""
    raw = Path(path).read_bytes()
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    return pd.read_csv(io.StringIO(text), sep="::", engine="python", names=names)


def load_ml1m(data_dir):
    data_dir = Path(data_dir)
    ratings = read_dat(data_dir / "ratings.dat", ["user_id", "movie_id", "rating", "timestamp"])
    movies = read_dat(data_dir / "movies.dat", ["movie_id", "title", "genres"])
    users = read_dat(data_dir / "users.dat", ["user_id", "gender", "age_code", "occupation_code", "zip"])
    movies["year"] = movies["title"].str.extract(r"\((\d{4})\)\s*$")[0].astype("Int64")
    return ratings, movies, users


def training_part(ratings, val_frac=0.10, test_frac=0.20):
    """Each user's oldest 70% of ratings: exactly the notebook's per-user chronological training split."""
    work = ratings[["user_id", "movie_id", "rating", "timestamp"]].copy()
    work["_order"] = np.arange(len(work))
    work = work.sort_values(["user_id", "timestamp", "_order"]).drop(columns="_order")
    g = work.groupby("user_id")
    pos, size = g.cumcount().to_numpy(), g["movie_id"].transform("size").to_numpy()
    n_test = np.maximum(1, (test_frac * size + 0.5).astype(int))
    n_val = np.maximum(1, (val_frac * size + 0.5).astype(int))
    return work[pos < size - n_test - n_val]


# ----------------------------------------------------------------------------------------------- training
def item_similarity(R, alpha=0.5):
    """Cosine-style similarity between movies from a (users x movies) 0/1 sparse matrix."""
    co = (R.T @ R).toarray().astype(np.float64)
    n = np.diag(co) + 1e-9
    sim = co / (np.power(n, alpha)[:, None] * np.power(n, 1 - alpha)[None, :])
    np.fill_diagonal(sim, 0)
    return sim.astype(np.float32)


def als_implicit(conf_minus_1, target, n_users, n_items, factors=ALS_FACTORS, reg=ALS_REG, iters=ALS_ITERS, seed=SEED):
    """Implicit ALS (Hu, Koren & Volinsky 2008), written from scratch."""
    rng = np.random.default_rng(seed)
    X = rng.normal(0, 0.01, (n_users, factors))
    Y = rng.normal(0, 0.01, (n_items, factors))
    by_user = (conf_minus_1.tocsr(), target.tocsr())
    by_item = (conf_minus_1.T.tocsr(), target.T.tocsr())
    ridge = reg * np.eye(factors)

    def solve(Cm, Tg, F):
        FtF = F.T @ F + ridge
        rhs = Tg @ F
        out = np.empty((Cm.shape[0], factors))
        for r in range(Cm.shape[0]):
            s, e = Cm.indptr[r], Cm.indptr[r + 1]
            Fr = F[Cm.indices[s:e]]
            out[r] = np.linalg.solve(FtF + (Fr.T * Cm.data[s:e]) @ Fr, rhs[r])
        return out

    for _ in range(iters):
        X = solve(*by_user, Y)
        Y = solve(*by_item, X)
    return X, Y


def build_artifacts(data_dir, out_dir, train_only=False, verbose=True):
    """Fit item-kNN + ALS on the ratings and save everything the app needs to `out_dir`.

    train_only=True fits on each user's training split only (used to check that the app reproduces the notebook).
    """
    log = print if verbose else (lambda *a, **k: None)
    ratings, movies, users = load_ml1m(data_dir)
    user_ids = np.sort(ratings.user_id.unique())
    item_ids = np.sort(ratings.movie_id.unique())                   # movies rated at least once in the full data
    used = training_part(ratings) if train_only else ratings
    u_pos = pd.Series(np.arange(len(user_ids)), index=user_ids)
    i_pos = pd.Series(np.arange(len(item_ids)), index=item_ids)
    ur, ic = u_pos.loc[used.user_id].to_numpy(), i_pos.loc[used.movie_id].to_numpy()
    rt = used.rating.to_numpy()
    n_users, n_items = len(user_ids), len(item_ids)
    shape = (n_users, n_items)
    log(f"{len(used):,} ratings, {n_users:,} users, {n_items:,} movies")

    R_all = csr_matrix((np.ones(len(used), dtype=np.float32), (ur, ic)), shape=shape)
    liked = rt >= POS_THRESHOLD
    R_lik = csr_matrix((np.ones(liked.sum(), dtype=np.float32), (ur[liked], ic[liked])), shape=shape)

    log("item-kNN similarity ...")
    sim = item_similarity(R_all)

    log("ALS (about 30 seconds) ...")
    w = np.where(rt >= 4, ALS_ALPHA, np.where(rt <= 2, ALS_ALPHA_NEG, 0.0))
    keep = w > 0
    conf = csr_matrix((w[keep], (ur[keep], ic[keep])), shape=shape)
    target = csr_matrix(((1 + ALS_ALPHA) * np.ones(liked.sum()), (ur[liked], ic[liked])), shape=shape)
    _, Y = als_implicit(conf, target, n_users, n_items)

    n_pop = np.asarray(R_all.sum(axis=0)).ravel()
    log_pop = np.log(n_pop + 1.0)
    log_pop_z = (log_pop - log_pop.mean()) / log_pop.std()

    u_info = users.set_index("user_id").loc[user_ids]
    segment = (u_info.gender + "_" + u_info.age_code.astype(str)).to_numpy()
    seg_names = np.unique(segment)
    seg_share = np.vstack([np.asarray(R_lik[np.flatnonzero(segment == s)].mean(axis=0)).ravel() for s in seg_names])
    global_share = np.asarray(R_lik.mean(axis=0)).ravel()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / "model.npz", sim=sim, Y=Y.astype(np.float32), item_ids=item_ids, n_pop=n_pop,
             log_pop_z=log_pop_z.astype(np.float32), seg_names=seg_names.astype(str), seg_share=seg_share.astype(np.float32),
             global_share=global_share.astype(np.float32))
    meta = movies.set_index("movie_id").loc[item_ids].reset_index()[["movie_id", "title", "year", "genres"]]
    meta.to_csv(out_dir / "movies.csv", index=False)
    log(f"saved model.npz and movies.csv to {out_dir}")


# ----------------------------------------------------------------------------------------------- serving
def _z(v, unseen):
    """Standardise scores over the movies the user has not seen, so different models can be added."""
    v = np.asarray(v, dtype=np.float64)
    mu, sd = v[unseen].mean(), v[unseen].std() + 1e-9
    return (v - mu) / sd


# words that map to a direction in "movie space"; each returns a per-movie bonus in [-1, 1]
_LIGHT = {"Comedy": 1, "Animation": 1, "Children's": 1, "Musical": 1, "Romance": 0.5,
          "Horror": -1, "Thriller": -1, "War": -1, "Crime": -1, "Drama": -0.5, "Film-Noir": -0.5, "Mystery": -0.5}
_GENRE_WORDS = {"action": ["Action"], "adventure": ["Adventure"], "animation": ["Animation"], "animated": ["Animation"],
                "kids": ["Children's"], "comedy": ["Comedy"], "crime": ["Crime"], "documentary": ["Documentary"],
                "drama": ["Drama"], "fantasy": ["Fantasy"], "noir": ["Film-Noir"], "horror": ["Horror"],
                "musical": ["Musical"], "mystery": ["Mystery"], "romance": ["Romance"], "romantic": ["Romance"],
                "scifi": ["Sci-Fi"], "sci-fi": ["Sci-Fi"], "science fiction": ["Sci-Fi"], "thriller": ["Thriller"],
                "war": ["War"], "western": ["Western"], "funnier": ["Comedy"], "funny": ["Comedy"],
                "scarier": ["Horror", "Thriller"], "scary": ["Horror", "Thriller"], "creepier": ["Horror", "Thriller"]}
_STOP = {"but", "with", "a", "an", "more", "less", "the", "and", "like", "than", "something", "movie", "movies", "please",
         "i", "want", "it", "is", "to", "me", "some", "of", "that", "this", "just", "bit", "little", "very", "film", "films"}


class Recommender:
    def __init__(self, artifact_dir):
        d = Path(artifact_dir)
        z = np.load(d / "model.npz", allow_pickle=False)
        self.sim = z["sim"]
        self.Y = z["Y"].astype(np.float64)
        self.item_ids = z["item_ids"]
        self.n_pop = z["n_pop"]
        self.log_pop_z = z["log_pop_z"].astype(np.float64)
        self.seg_names = list(z["seg_names"])
        self.seg_share = z["seg_share"]
        self.global_share = z["global_share"]
        self.movies = pd.read_csv(d / "movies.csv").set_index("movie_id")
        self.col = {int(m): i for i, m in enumerate(self.item_ids)}
        self.n_items = len(self.item_ids)
        self.YtY = self.Y.T @ self.Y + ALS_REG * np.eye(self.Y.shape[1])
        self.titles = [self.movies.loc[m, "title"] for m in self.item_ids]
        genre_lists = self.movies.loc[self.item_ids, "genres"].str.split("|")
        self.genre_mat = np.array([[g in gl for g in GENRES] for gl in genre_lists], dtype=np.float32)
        self.years = self.movies.loc[self.item_ids, "year"].to_numpy(dtype=float)

    # ---- helpers
    def search(self, query, n=10):
        q = query.lower().strip()
        hits = [i for i, t in enumerate(self.titles) if q in t.lower()]
        hits.sort(key=lambda i: -self.n_pop[i])
        return [(int(self.item_ids[i]), self.titles[i]) for i in hits[:n]]

    def _prior_vector(self, gender, age_code):
        key = f"{gender}_{age_code}"
        if key in self.seg_names:
            return self.seg_share[self.seg_names.index(key)]
        return self.global_share

    def _fold_in(self, liked, disliked):
        """Solve a taste vector for a (possibly new) user with the movie vectors held fixed."""
        cols = np.array(list(liked) + list(disliked), dtype=int)
        if len(cols) == 0:
            return None
        conf = np.array([ALS_ALPHA] * len(liked) + [ALS_ALPHA_NEG] * len(disliked))
        tgt = np.array([1 + ALS_ALPHA] * len(liked) + [0.0] * len(disliked))
        F = self.Y[cols]
        return np.linalg.solve(self.YtY + (F.T * conf) @ F, F.T @ tgt)

    @staticmethod
    def _split_ratings(cols):
        liked = [c for c, r in cols.items() if r >= POS_THRESHOLD]
        disliked = [c for c, r in cols.items() if r <= 2]
        return liked, disliked

    def _because(self, liked, j):
        if not liked:
            return None
        contrib = self.sim[liked, j]
        order = np.argsort(-contrib)[:2]
        names = [self.titles[liked[o]] for o in order if contrib[o] > 0]
        return "Because you liked " + " and ".join(names) if names else None

    # ---- mode 1: recommendations for a user described by a few ratings
    def recommend(self, ratings, gender=None, age_code=None, n=10, gamma=POP_PENALTY, exclude_ids=()):
        """ratings: {movie_id: stars}. Works from zero ratings (demographic prior) up to a full history."""
        cols = {self.col[int(m)]: r for m, r in ratings.items() if int(m) in self.col}
        liked, disliked = self._split_ratings(cols)
        k = len(cols)
        seen = np.zeros(self.n_items, dtype=bool)
        seen[list(cols)] = True
        for m in exclude_ids:
            if int(m) in self.col:
                seen[self.col[int(m)]] = True
        unseen = ~seen

        s_knn = self.sim[liked].sum(axis=0) if liked else np.zeros(self.n_items)
        x = self._fold_in(liked, disliked)
        s_als = self.Y @ x if x is not None else np.zeros(self.n_items)
        hybrid = _z(s_knn, unseen) + HYBRID_ALS_WEIGHT * _z(s_als, unseen)
        prior = _z(self._prior_vector(gender, age_code), unseen) / (1 + k / PRIOR_TAU)
        penalty = gamma * k / (k + PRIOR_TAU) * self.log_pop_z
        score = hybrid + prior - penalty
        score[seen] = -np.inf
        top = np.argsort(-score, kind="stable")[:n]

        rows = []
        for j in top:
            because = self._because(liked, j)
            if because is None:
                because = "Popular with people of your age and gender" if f"{gender}_{age_code}" in self.seg_names else "Popular overall"
            rows.append({"movie_id": int(self.item_ids[j]), "title": self.titles[j], "year": self.years[j],
                         "genres": self.movies.loc[self.item_ids[j], "genres"], "score": float(score[j]), "why": because})
        return pd.DataFrame(rows)

    def recommendation_mode(self, k):
        """Plain-words description of what drives the list for a user with k ratings."""
        w = 1 / (1 + k / PRIOR_TAU)
        if k == 0:
            return "No ratings yet: using what people in your age and gender group like."
        return (f"{k} rating(s) known: collaborative filtering counts fully, the demographic prior has weight {w:.2f} "
                f"and the popularity penalty is {POP_PENALTY * k / (k + PRIOR_TAU):.2f}.")

    # ---- mode 2: "like X but ..."
    def lightness(self):
        """Per-movie 'lightness' in [-1, 1] from genres (comedy/animation up, horror/thriller/war down)."""
        w = np.array([_LIGHT.get(g, 0.0) for g in GENRES], dtype=np.float32)
        n_g = np.maximum(self.genre_mat.sum(axis=1), 1)
        return (self.genre_mat @ w) / n_g

    def _genre_bonus(self, names):
        idx = [GENRES.index(g) for g in names]
        return (self.genre_mat[:, idx].sum(axis=1) > 0).astype(np.float64)

    def parse_modifiers(self, text):
        """Turn free text such as 'lighter and more recent' into per-movie bonuses. Returns (bonus, labels, ignored words)."""
        t = " " + re.sub(r"[^a-z\- ]", " ", text.lower()) + " "
        bonus, labels, used = np.zeros(self.n_items), [], []
        def add(pattern, label, vec):
            nonlocal bonus, t
            if re.search(pattern, t):
                bonus = bonus + vec; labels.append(label); t = re.sub(pattern, " ", t)
        light = self.lightness().astype(np.float64)
        add(r"\b(lighter|light|feel-good|feel good|happier|less dark|less serious|more fun|fun)\b", "lighter (comedy, animation, musical up; horror, thriller, war, crime down)", light)
        add(r"\b(darker|dark|grittier|bleaker|more serious|serious)\b", "darker (the reverse of lighter)", -light)
        yr = np.clip((np.nan_to_num(self.years, nan=1990) - 1985) / 15, -1, 1)
        add(r"\b(newer|more recent|recent|modern)\b", "newer", yr)
        add(r"\b(older|classic|classics|vintage|old)\b", "older", -yr)
        add(r"\b(obscure|hidden gem|hidden gems|underrated|less popular|lesser known|lesser-known)\b", "more obscure", np.clip(-self.log_pop_z / 2, -1, 1))
        add(r"\b(popular|mainstream|well known|well-known|famous)\b", "more mainstream", np.clip(self.log_pop_z / 2, -1, 1))
        add(r"\b(family|kid friendly|kid-friendly|for kids)\b", "family friendly (children's, animation up; horror down)",
            self._genre_bonus(["Children's", "Animation"]) - self._genre_bonus(["Horror"]))
        for word in sorted(_GENRE_WORDS, key=len, reverse=True):
            add(rf"\b{re.escape(word)}\b", f"more {'/'.join(_GENRE_WORDS[word])}", self._genre_bonus(_GENRE_WORDS[word]))
        ignored = [w for w in t.split() if w not in _STOP]
        return bonus, labels, ignored

    def similar(self, movie_id, text="", n=10, strength=1.5, pool=150):
        """Movies liked by the same audience as `movie_id`, re-ranked by the text modifiers."""
        c = self.col[int(movie_id)]
        order = np.argsort(-self.sim[c], kind="stable")
        order = order[order != c][:pool]
        base = self.sim[c, order]
        base_z = (base - base.mean()) / (base.std() + 1e-9)
        bonus, labels, ignored = self.parse_modifiers(text)
        score = base_z + strength * bonus[order]
        top = order[np.argsort(-score, kind="stable")[:n]]
        rank_of = {int(j): r + 1 for r, j in enumerate(order)}
        rows = [{"movie_id": int(self.item_ids[j]), "title": self.titles[j], "year": self.years[j],
                 "genres": self.movies.loc[self.item_ids[j], "genres"], "audience similarity rank": rank_of[int(j)]} for j in top]
        return pd.DataFrame(rows), labels, ignored
