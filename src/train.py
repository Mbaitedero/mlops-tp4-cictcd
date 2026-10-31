from sklearn.metrics import accuracy_score, f1_score

from src.data import load_data, load_params, split_data
from src.model import build_model


def train():
    params = load_params()
    df = load_data()

    X_train, X_test, y_train, y_test = split_data(df, params)

    model = build_model(params)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    metrics = {
        "accuracy": accuracy_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
    }

    return model, metrics
