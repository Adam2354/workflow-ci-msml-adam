import pandas as pd
import mlflow
import mlflow.sklearn

from pathlib import Path

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score
)


def main():

    print("=" * 60)
    print("TRAINING MODEL TELCO CUSTOMER CHURN")
    print("=" * 60)

    # ==================================================
    # Load Dataset
    # ==================================================
    current_dir = Path(__file__).resolve().parent

    dataset_path = (
        current_dir /
        "dataset" /
        "telco_churn_preprocessing.csv"
    )

    print("\nLoading dataset...")
    print(dataset_path)

    df = pd.read_csv(dataset_path)

    print(f"Shape dataset : {df.shape}")

    # ==================================================
    # Pisahkan Fitur dan Target
    # ==================================================
    X = df.drop("Churn", axis=1)
    y = df["Churn"]

    # ==================================================
    # Train Test Split
    # ==================================================
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    print(f"\nData Train : {X_train.shape}")
    print(f"Data Test  : {X_test.shape}")

    # ==================================================
    # MLflow Tracking
    # ==================================================
    mlflow.set_experiment("Telco_Churn_Classification")

    # Basic -> gunakan autolog
    mlflow.sklearn.autolog()

    with mlflow.start_run():

        # ==============================================
        # Training Model
        # ==============================================
        print("\nTraining Random Forest...")

        model = RandomForestClassifier(
            n_estimators=100,
            random_state=42
        )

        model.fit(X_train, y_train)

        # ==============================================
        # Prediksi
        # ==============================================
        y_pred = model.predict(X_test)

        # ==============================================
        # Evaluasi
        # ==============================================
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred)
        recall = recall_score(y_test, y_pred)
        f1 = f1_score(y_test, y_pred)

        print("\nHASIL EVALUASI")
        print("-" * 40)

        print(f"Accuracy  : {accuracy:.4f}")
        print(f"Precision : {precision:.4f}")
        print(f"Recall    : {recall:.4f}")
        print(f"F1 Score  : {f1:.4f}")

    print("\nTraining selesai.")
    print("Buka MLflow UI untuk melihat hasil tracking.")


if __name__ == "__main__":
    main()