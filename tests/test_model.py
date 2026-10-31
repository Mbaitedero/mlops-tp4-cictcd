from src.data import load_params
from src.train import train


def test_model_quality_gate():
    params = load_params()
    _, metrics = train()

    assert metrics["accuracy"] >= params["quality_gate"]["min_accuracy"]
    assert metrics["f1"] >= params["quality_gate"]["min_f1"]


def test_model_can_predict():
    model, _ = train()
    prediction = model.predict(["Ce produit est excellent"])
    assert prediction[0] in [0, 1]
