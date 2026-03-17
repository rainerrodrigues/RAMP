import argparse
import pandas as pd
import joblib
import mlflow
import mlflow.sklearn

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score


def train(data_path, model_path):

    df = pd.read_csv(data_path)

    # X = df.drop("label", axis=1)
    X = df.drop(columns=['label', 'file_name'])
    y = df["label"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    n_estimators = 100
    max_depth = 5

    with mlflow.start_run():

        model = RandomForestClassifier(
            n_estimators=n_estimators,
            max_depth=max_depth,
            random_state=42
        )

        model.fit(X_train, y_train)

        preds = model.predict(X_test)
        accuracy = accuracy_score(y_test, preds)

        # Log parameters
        mlflow.log_param("n_estimators", n_estimators)
        mlflow.log_param("max_depth", max_depth)

        # Log metric
        mlflow.log_metric("accuracy", accuracy)

        # Log model
        mlflow.sklearn.log_model(model, "model")

        joblib.dump(model, model_path)

        print(f"Model saved to {model_path}")
        print(f"Accuracy: {accuracy}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--data", required=True)
    parser.add_argument("--model", required=True)

    args = parser.parse_args()

    train(args.data, args.model)
