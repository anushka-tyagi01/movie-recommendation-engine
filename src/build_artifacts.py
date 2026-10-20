"""Build the model files the app needs.   python -m src.build_artifacts --data data/raw/ml-1m --out artifacts"""
import argparse
from .recsys import build_artifacts

if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--data", default="data/raw/ml-1m", help="folder with ratings.dat, movies.dat, users.dat")
    p.add_argument("--out", default="artifacts", help="where to write model.npz and movies.csv")
    p.add_argument("--train-only", action="store_true", help="fit on the training split only (for evaluation)")
    a = p.parse_args()
    build_artifacts(a.data, a.out, train_only=a.train_only)
