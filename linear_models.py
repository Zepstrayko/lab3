import numpy as np
from typing import Tuple
from grader_contracts.linear_models import (
    BaselineComparisonResult,
    ClassificationInput,
    ClassificationMetrics,
    ClassificationResult,
    RegressionInput,
    RegressionResult,
)


def sigmoid(z: np.ndarray) -> np.ndarray:
    """Sigmoid функция с защитой от переполнения."""
    z = np.clip(z, -500, 500)
    return 1.0 / (1.0 + np.exp(-z))


def standardize_features(X: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Стандартизация признаков: X_scaled = (X - mean) / std."""
    mean = np.mean(X, axis=0)
    std = np.std(X, axis=0)
    std[std == 0] = 1.0
    X_scaled = (X - mean) / std
    return X_scaled, mean, std


def train_linear_regression(data: RegressionInput) -> RegressionResult:
    """Обучите собственную линейную регрессию на train и верните прогноз для test."""
    X_train = np.array(data.train_features, dtype=np.float64)
    y_train = np.array(data.train_target, dtype=np.float64)
    X_test = np.array(data.test_features, dtype=np.float64)
    
    X_train_scaled, mean, std = standardize_features(X_train)
    X_test_scaled = (X_test - mean) / std
    
    n_samples, n_features = X_train_scaled.shape
    weights = np.zeros(n_features, dtype=np.float64)
    bias = 0.0
    
    lr = data.learning_rate
    epochs = data.epochs
    
    loss_history = []
    
    for epoch in range(epochs):
        y_pred = X_train_scaled @ weights + bias
        errors = y_pred - y_train
        mse = np.mean(errors ** 2)
        loss_history.append(float(mse))
        
        dw = (2.0 / n_samples) * (X_train_scaled.T @ errors)
        db = (2.0 / n_samples) * np.sum(errors)
        
        weights -= lr * dw
        bias -= lr * db
    
    predictions = X_test_scaled @ weights + bias
    
    return RegressionResult(
        weights=weights,
        bias=float(bias),
        loss_history=loss_history,
        predictions=predictions,
    )


def train_logistic_classifier(data: ClassificationInput) -> ClassificationResult:
    """Обучите собственный логистический классификатор на train и верните прогноз для test."""
    X_train = np.array(data.train_features, dtype=np.float64)
    y_train = np.array(data.train_target, dtype=np.float64)
    X_test = np.array(data.test_features, dtype=np.float64)
    
    X_train_scaled, mean, std = standardize_features(X_train)
    X_test_scaled = (X_test - mean) / std
    
    n_samples, n_features = X_train_scaled.shape
    weights = np.zeros(n_features, dtype=np.float64)
    bias = 0.0
    
    lr = data.learning_rate
    epochs = data.epochs
    
    loss_history = []
    
    for epoch in range(epochs):
        z = X_train_scaled @ weights + bias
        y_pred = sigmoid(z)
        
        eps = 1e-15
        y_pred_clipped = np.clip(y_pred, eps, 1 - eps)
        bce = -np.mean(y_train * np.log(y_pred_clipped) + (1 - y_train) * np.log(1 - y_pred_clipped))
        loss_history.append(float(bce))
        
        error = y_pred - y_train
        dw = (1.0 / n_samples) * (X_train_scaled.T @ error)
        db = (1.0 / n_samples) * np.sum(error)
        
        weights -= lr * dw
        bias -= lr * db
    
    z_test = X_test_scaled @ weights + bias
    probabilities = sigmoid(z_test)
    predictions = (probabilities >= 0.5).astype(np.float64)
    
    return ClassificationResult(
        weights=weights,
        bias=float(bias),
        loss_history=loss_history,
        probabilities=probabilities,
        predictions=predictions,
    )


def compute_metrics(y_true: np.ndarray, y_prob: np.ndarray, y_pred: np.ndarray) -> ClassificationMetrics:
    """Вычисление метрик классификации."""
    from sklearn.metrics import roc_auc_score, accuracy_score, precision_score, recall_score, f1_score
    
    roc_auc = roc_auc_score(y_true, y_prob)
    accuracy = accuracy_score(y_true, y_pred)
    precision = precision_score(y_true, y_pred, zero_division=0)
    recall = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)
    
    return ClassificationMetrics(
        roc_auc=float(roc_auc),
        accuracy=float(accuracy),
        precision=float(precision),
        recall=float(recall),
        f1=float(f1),
    )


def compare_classical_models(data: ClassificationInput) -> BaselineComparisonResult:
    """Сравните готовые Logistic Regression, Decision Tree и KNN на одном split."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.tree import DecisionTreeClassifier
    from sklearn.neighbors import KNeighborsClassifier
    from sklearn.preprocessing import StandardScaler
    
    X_train = np.array(data.train_features, dtype=np.float64)
    y_train = np.array(data.train_target, dtype=np.float64)
    X_test = np.array(data.test_features, dtype=np.float64)
    y_test = np.array(data.test_target, dtype=np.float64)
    
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Logistic Regression (sklearn)
    lr_model = LogisticRegression(max_iter=1000, random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_prob = lr_model.predict_proba(X_test_scaled)[:, 1]
    lr_pred = lr_model.predict(X_test_scaled)
    lr_metrics = compute_metrics(y_test, lr_prob, lr_pred)
    
    # Decision Tree
    dt_model = DecisionTreeClassifier(random_state=42)
    dt_model.fit(X_train, y_train)
    dt_prob = dt_model.predict_proba(X_test)[:, 1]
    dt_pred = dt_model.predict(X_test)
    dt_metrics = compute_metrics(y_test, dt_prob, dt_pred)
    
    # KNN
    knn_model = KNeighborsClassifier(n_neighbors=5)
    knn_model.fit(X_train_scaled, y_train)
    knn_prob = knn_model.predict_proba(X_test_scaled)[:, 1]
    knn_pred = knn_model.predict(X_test_scaled)
    knn_metrics = compute_metrics(y_test, knn_prob, knn_pred)
    
    return BaselineComparisonResult(
        logistic_regression=lr_metrics,
        decision_tree=dt_metrics,
        knn=knn_metrics,
    )