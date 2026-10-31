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
