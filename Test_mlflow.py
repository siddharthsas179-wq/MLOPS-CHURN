import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("sqlite:///mlflow.db")
client = MlflowClient()

experiments = client.search_experiments()

print("\n=== MLflow Experiments ===")

for experiment in experiments:
    print(f"\nExperiment: {experiment.name}")
    print(f"Experiment ID: {experiment.experiment_id}")

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id]
    )

    print(f"Number of runs: {len(runs)}")

    for run in runs:
        print("Run ID:", run.info.run_id)
        print("Status:", run.info.status)
        print("Parameters:", run.data.params)
        print("Metrics:", run.data.metrics)