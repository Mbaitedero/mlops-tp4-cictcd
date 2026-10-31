#!/bin/bash
mkdir -p src tests .github/workflows data
touch src/__init__.py

cat > requirements.txt <<'EOF'
scikit-learn>=1.4,<2.0
pandas>=2.0,<3.0
numpy>=1.26,<3.0
mlflow>=3.1,<4.0
pytest>=8.0,<9.0
pytest-cov>=5.0,<7.0
pyyaml>=6.0,<7.0
ruff>=0.6,<1.0
bandit>=1.7,<2.0
pip-audit>=2.7,<3.0
EOF

cat > .gitignore <<'EOF'
TP4CICD/
.venv/
__pycache__/
.pytest_cache/
.coverage
mlruns/
mlartifacts/
mlflow.db
EOF

cat > params.yaml <<'EOF'
data:
  test_size: 0.25
  random_state: 42

model:
  max_features: 5000
  ngram_range: [1, 2]
  C: 1.0
  max_iter: 1000

quality_gate:
  min_accuracy: 0.60
  min_f1: 0.50
  max_regression: 0.05
  min_coverage: 70

training:
  experiment_name: "sentiment-classification"
  model_name: "sentiment-model"
EOF

cat > data/reviews.csv <<'EOF'
text,label
"Ce produit est excellent et je le recommande",1
"Tres bon produit et service rapide",1
"Super qualite, je suis tres satisfait",1
"Parfait, je racheterai sans hesiter",1
"Excellent rapport qualite-prix",1
"Tres satisfait de mon achat",1
"Produit remarquable et facile a utiliser",1
"La livraison etait rapide et efficace",1
"Tres bonne experience globale",1
"Je recommande vivement ce produit",1
"Qualite mediocre, je suis decu",0
"Produit casse a l'arrivee",0
"Service client horrible",0
"Mauvaise experience globale",0
"Ne fonctionne pas correctement",0
"Perte de temps et d'argent",0
"Tres mauvaise qualite",0
"Je deconseille fortement ce produit",0
"Livraison en retard et emballage abime",0
"Produit inutilisable et decevant",0
"Produit excellent, qualite parfaite",1
"Tres bon service, livraison rapide",1
"Je recommande, produit de bonne qualite",1
"Super achat, tres satisfait du produit",1
"Excellent service client, rapide et efficace",1
"Bonne qualite et livraison rapide",1
"Tres bon rapport qualite prix, je recommande",1
"Produit parfait, je suis satisfait",1
"Service excellent et produit remarquable",1
"Achat tres satisfaisant, bonne experience",1
"Mauvaise qualite, produit decevant",0
"Service client mauvais, livraison en retard",0
"Je ne recommande pas, produit de mauvaise qualite",0
"Tres decu du produit, mauvaise experience",0
"Livraison lente et service horrible",0
"Produit defectueux, qualite mediocre",0
"Achat decevant, je deconseille ce produit",0
"Service client inutile et livraison en retard",0
"Mauvais rapport qualite prix, tres decu",0
"Produit abime et service mauvais",0
EOF

cat > src/data.py <<'EOF'
import pandas as pd
import yaml
from sklearn.model_selection import train_test_split


def load_params(path="params.yaml"):
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_data(path="data/reviews.csv"):
    return pd.read_csv(path)


def split_data(df, params):
    X = df["text"]
    y = df["label"]
    return train_test_split(
        X,
        y,
        test_size=params["data"]["test_size"],
        random_state=params["data"]["random_state"],
        stratify=y,
    )
EOF

cat > src/model.py <<'EOF'
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
EOF

cat > src/train.py <<'EOF'
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
EOF

cat > tests/test_unit.py <<'EOF'
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
EOF

cat > tests/test_data.py <<'EOF'
from src.data import load_data


def test_no_missing_values():
    df = load_data()
    assert df.isnull().sum().sum() == 0


def test_labels_are_binary():
    df = load_data()
    assert set(df["label"].unique()).issubset({0, 1})


def test_text_not_empty():
    df = load_data()
    assert (df["text"].str.strip() != "").all()


def test_class_balance_is_reasonable():
    df = load_data()
    counts = df["label"].value_counts()
    assert len(counts) == 2
    assert counts.min() >= 5
EOF

cat > tests/test_model.py <<'EOF'
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
EOF

cat > train_and_register.py <<'EOF'
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient

from src.data import load_params
from src.train import train

params = load_params()

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment(params["training"]["experiment_name"])

model_name = params["training"]["model_name"]

with mlflow.start_run(run_name="sentiment-candidate") as run:
    model, metrics = train()

    mlflow.log_params({
        "C": params["model"]["C"],
        "max_features": params["model"]["max_features"],
        "ngram_range": str(params["model"]["ngram_range"]),
    })

    mlflow.log_metrics(metrics)

    model_info = mlflow.sklearn.log_model(
        model,
        name="model",
        registered_model_name=model_name,
    )

client = MlflowClient()

versions = client.search_model_versions(
    f"name = '{model_name}'"
)
latest = max(versions, key=lambda v: int(v.version))

client.set_model_version_tag(
    model_name,
    latest.version,
    "validation_status",
    "passed",
)

client.set_model_version_tag(
    model_name,
    latest.version,
    "use_case",
    "sentiment",
)

client.set_registered_model_tag(
    model_name,
    "task",
    "text-classification",
)

client.set_registered_model_alias(
    model_name,
    "candidate",
    latest.version,
)

print(f"Version enregistree : {latest.version}")
print(f"Alias candidate -> {latest.version}")
print(f"Accuracy : {metrics['accuracy']:.3f}")
print(f"F1 : {metrics['f1']:.3f}")
EOF

cat > promote.py <<'EOF'
import argparse

from mlflow import MlflowClient

from src.data import load_params

parser = argparse.ArgumentParser()
parser.add_argument(
    "--approve",
    action="store_true",
    help="Simule l'approbation humaine avant champion",
)
args = parser.parse_args()

params = load_params()
model_name = params["training"]["model_name"]

client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
candidate = client.get_model_version_by_alias(
    model_name,
    "candidate",
)

status = candidate.tags.get("validation_status")

if status != "passed":
    raise RuntimeError(
        "La version candidate n'a pas passe les Quality Gates."
    )

print(f"Candidate : version {candidate.version}")
print(f"validation_status = {status}")

if not args.approve:
    print("Aucune approbation : le modele reste candidate.")
    raise SystemExit(0)

client.set_registered_model_alias(
    model_name,
    "champion",
    candidate.version,
)

client.set_model_version_tag(
    model_name,
    candidate.version,
    "approval_status",
    "approved",
)

print(f"champion -> version {candidate.version}")
EOF

cat > .github/workflows/mlops-pipeline.yml <<'EOF'
name: MLOps CI/CD

on:
  push:
    branches: [main, develop]
  pull_request:
    branches: [main]

permissions:
  contents: read

concurrency:
  group: mlops-${{ github.ref }}
  cancel-in-progress: true

jobs:
  quality:
    runs-on: ubuntu-latest

    steps:
      - name: Checkout
        uses: actions/checkout@v4

      - name: Setup Python
        uses: actions/setup-python@v5
        with:
          python-version: "3.11"
          cache: "pip"

      - name: Install dependencies
        run: |
          python -m pip install --upgrade pip
          pip install -r requirements.txt

      - name: Lint
        run: ruff check .

      - name: Unit and data tests
        env:
          PYTHONPATH: .
        run: |
          pytest tests/test_unit.py tests/test_data.py -v

      - name: Model quality gates
        env:
          PYTHONPATH: .
        run: |
          pytest tests/test_model.py -v

      - name: Security - Bandit
        run: bandit -r src

      - name: Security - pip-audit
        run: pip-audit

  staging:
    needs: quality
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    runs-on: ubuntu-latest
    environment: staging

    steps:
      - name: Simulated deployment to staging
        run: echo "Deploy candidate to STAGING"

  production:
    needs: staging
    if: github.ref == 'refs/heads/main' && github.event_name == 'push'
    runs-on: ubuntu-latest
    environment: production

    steps:
      - name: Simulated deployment to production
        run: echo "Deploy champion to PRODUCTION"
EOF

cat > model_card.md <<'EOF'
# Sentiment Model

## Use case
Classification binaire de sentiment.

## Version
A completer.

## Dataset
reviews.csv -- jeu pedagogique.

## Method
TF-IDF + Logistic Regression.

## Parameters
C = 1.0
max_features = 5000
ngram_range = [1, 2]

## Metrics
Accuracy = ...
F1 = ...

## Quality Gates
Unit tests = ...
Data tests = ...
Model tests = ...
Security checks = ...

## Known limitations
- Petit jeu de donnees pedagogique.
- Langue principalement francaise.
- Evaluation non representative d'un contexte industriel.

## Governance
validation_status = passed
approval_status = ...
alias = candidate / champion

## Deployment decision
A completer.
EOF

echo "# mlops-tp4-cictcd" > README.md
echo "Projet cree."
