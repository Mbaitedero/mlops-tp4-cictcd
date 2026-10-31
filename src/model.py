from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline


def build_model(params):
    return Pipeline([
        (
            "tfidf",
            TfidfVectorizer(
                max_features=params["model"]["max_features"],
                ngram_range=tuple(params["model"]["ngram_range"]),
            ),
        ),
        (
            "clf",
            LogisticRegression(
                C=params["model"]["C"],
                max_iter=params["model"]["max_iter"],
            ),
        ),
    ])
