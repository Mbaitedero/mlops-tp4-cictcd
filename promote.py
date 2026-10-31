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
