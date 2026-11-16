import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from linear_models import (
    train_linear_regression,
    train_logistic_classifier,
    compare_classical_models,
)
from grader_contracts.linear_models import (
    RegressionInput,
    ClassificationInput,
)


def run_linear_regression():
    df = pd.read_csv("Student_Performance.csv")
    
    if "Extracurricular Activities" in df.columns:
        df["Extracurricular Activities"] = df["Extracurricular Activities"].map({"Да": 1, "Нет": 0, "Yes": 1, "No": 0})
    
    if "Gender" in df.columns:
        df["Gender"] = df["Gender"].map({"M": 1, "F": 0, "Male": 1, "Female": 0})
    
    target_col = "Performance Index"
    feature_cols = [col for col in df.columns if col != target_col]
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    data = RegressionInput(
        train_features=X_train,
        train_target=y_train,
        test_features=X_test,
        test_target=y_test,
        learning_rate=0.05,
        epochs=800,
    )
    
    result = train_linear_regression(data)
    test_mse = float(np.mean((result.predictions - y_test) ** 2))
    
    print(f"Linear Regression Test MSE: {test_mse:.4f}")
    print(f"Final loss (epoch {len(result.loss_history)}): {result.loss_history[-1]:.4f}")
    
    return test_mse, result.loss_history


def run_logistic_classification():
    df = pd.read_csv("german.csv", sep=";")
    
    target_col = df.columns[0]
    feature_cols = [col for col in df.columns if col != target_col]
    
    X = df[feature_cols].values
    y = df[target_col].values
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    data = ClassificationInput(
        train_features=X_train,
        train_target=y_train,
        test_features=X_test,
        test_target=y_test,
        learning_rate=0.1,
        epochs=1000,
    )
    
    result = train_logistic_classifier(data)
    
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
    
    roc_auc = roc_auc_score(y_test, result.probabilities)
    accuracy = accuracy_score(y_test, result.predictions)
    precision = precision_score(y_test, result.predictions, zero_division=0)
    recall = recall_score(y_test, result.predictions, zero_division=0)
    f1 = f1_score(y_test, result.predictions, zero_division=0)
    
    print(f"Own Logistic Regression:")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  Precision: {precision:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1: {f1:.4f}")
    
    own_logistic = {
        "roc_auc": float(roc_auc),
        "accuracy": float(accuracy),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
    }
    
    return own_logistic, data


def run_comparison(data: ClassificationInput):
    """Сравнение моделей."""
    result = compare_classical_models(data)
    
    print(f"Sklearn Logistic Regression:")
    print(f"  ROC-AUC: {result.logistic_regression.roc_auc:.4f}")
    print(f"  Accuracy: {result.logistic_regression.accuracy:.4f}")
    print(f"  Precision: {result.logistic_regression.precision:.4f}")
    print(f"  Recall: {result.logistic_regression.recall:.4f}")
    print(f"  F1: {result.logistic_regression.f1:.4f}")
    
    print(f"Decision Tree:")
    print(f"  ROC-AUC: {result.decision_tree.roc_auc:.4f}")
    print(f"  Accuracy: {result.decision_tree.accuracy:.4f}")
    print(f"  Precision: {result.decision_tree.precision:.4f}")
    print(f"  Recall: {result.decision_tree.recall:.4f}")
    print(f"  F1: {result.decision_tree.f1:.4f}")
    
    print(f"KNN:")
    print(f"  ROC-AUC: {result.knn.roc_auc:.4f}")
    print(f"  Accuracy: {result.knn.accuracy:.4f}")
    print(f"  Precision: {result.knn.precision:.4f}")
    print(f"  Recall: {result.knn.recall:.4f}")
    print(f"  F1: {result.knn.f1:.4f}")
    
    return {
        "logistic_regression": {
            "roc_auc": float(result.logistic_regression.roc_auc),
            "accuracy": float(result.logistic_regression.accuracy),
            "precision": float(result.logistic_regression.precision),
            "recall": float(result.logistic_regression.recall),
            "f1": float(result.logistic_regression.f1),
        },
        "decision_tree": {
            "roc_auc": float(result.decision_tree.roc_auc),
            "accuracy": float(result.decision_tree.accuracy),
            "precision": float(result.decision_tree.precision),
            "recall": float(result.decision_tree.recall),
            "f1": float(result.decision_tree.f1),
        },
        "knn": {
            "roc_auc": float(result.knn.roc_auc),
            "accuracy": float(result.knn.accuracy),
            "precision": float(result.knn.precision),
            "recall": float(result.knn.recall),
            "f1": float(result.knn.f1),
        },
    }


def main():
    # 1. Линейная регрессия
    test_mse, loss_history = run_linear_regression()
    
    # 2. Логистическая регрессия
    own_logistic, data = run_logistic_classification()
    
    # 3. Сравнение моделей
    comparison = run_comparison(data)
    
    results = {
        "linear_regression": {"test_mse": test_mse},
        "classification": {
            "own_logistic": own_logistic,
            **comparison,
        },
    }
    
    with open("experiment_results.json", "w") as f:
        json.dump(results, f, indent=2)
    
    print("\nРезультаты сохранены в experiment_results.json")


if __name__ == "__main__":
    main()