import mlflow
from sklearn.model_selection import train_test_split
from src.data import load_data, preprocess
from src.model import train_model, evalute_model
from src.config import load_config
from src.logger import get_logger

logger = get_logger("main")


def main():
    cfg = load_config()

    logger.info("Loading data from %s", cfg["data"]["path"])

    df = load_data(cfg["data"]["path"])
    X, y = preprocess(df)

    logger.info(
        "Data ready: %d rows, %d features",
        X.shape[0],
        X.shape[1]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=cfg["data"]["test_size"],
        random_state=42,
        stratify=y
    )

    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("Churn Prediction")

    with mlflow.start_run():
        print("STEP 5: MLflow run started", flush=True)

        logger.info("Training model")

        model = train_model(
            X_train,
            y_train,
            max_iter=cfg["model"]["max_iter"]
        )

        metrics = evalute_model(model, X_test, y_test)

        logger.info("Metrics: %s", metrics)

        # Log parameters
        mlflow.log_param("model", "LogisticRegression")
        mlflow.log_param("test_size", cfg["data"]["test_size"])
        mlflow.log_param("max_iter", cfg["model"]["max_iter"])

        # Log metrics
        mlflow.log_metric("accuracy", metrics["accuracy"])
        mlflow.log_metric("f1", metrics["f1"])

        print("STEP 6: Parameters and metrics logged", flush=True)

    print("STEP 7: MLflow run completed successfully", flush=True)


if __name__ == "__main__":
    main()