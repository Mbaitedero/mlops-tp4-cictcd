from src.data import load_data, load_params
from src.model import build_model


def test_load_params():
    params = load_params()
    assert "data" in params
    assert "model" in params
    assert params["data"]["test_size"] == 0.25


def test_load_data():
    df = load_data()
    assert len(df) >= 20
    assert {"text", "label"}.issubset(df.columns)


def test_build_model():
    params = load_params()
    model = build_model(params)
    assert hasattr(model, "fit")
    assert hasattr(model, "predict")
