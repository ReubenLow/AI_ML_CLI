import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import argparse
from scipy.stats import shapiro, kstest, norm, probplot, chi2_contingency
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, PolynomialFeatures, StandardScaler
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn import neural_network
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC, LinearSVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression, Ridge, Perceptron, SGDClassifier
from sklearn.model_selection import cross_val_score, StratifiedShuffleSplit
from sklearn.metrics import (accuracy_score, log_loss, mean_squared_error, confusion_matrix,
                             precision_score, recall_score, auc, roc_curve, roc_auc_score,
                             f1_score, PrecisionRecallDisplay, RocCurveDisplay,
                             ConfusionMatrixDisplay)
from sklearn.metrics import classification_report
from sklearn.calibration import calibration_curve
import joblib

def select_testing_file(input_folder):
    files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    if not files:
        print("No CSV files found in the testing_data folder.")
        return None
    print("Available files for testing:")
    for idx, file in enumerate(files, 1):
        print(f"{idx}: {file}")
    choice = int(input("Select the file number to test the model on: ")) - 1
    return os.path.join(input_folder, files[choice])

def load_data(file_path):
    data = pd.read_csv(file_path)
    target_col = "Survived"
    X_test = data.drop(columns=[target_col])  
    y_test = data[target_col]  
    return X_test, y_test

def load_models(model_folder):
    models = {}
    for model_file in os.listdir(model_folder):
        if model_file.endswith('.pkl'):
            model_path = os.path.join(model_folder, model_file)
            model_name = model_file.split('_model.pkl')[0]
            models[model_name] = joblib.load(model_path)
    return models

def preprocess_test_data(X_test, training_features):
    for col in training_features:
        if col not in X_test.columns:
            X_test[col] = 0
    X_test = X_test[training_features]
    return X_test

def evaluate_model(model, X_test, y_test, output_dir):
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, "predict_proba") else None

    # Create directory for the model if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # Confusion Matrix
    cm = confusion_matrix(y_test, y_pred)
    ConfusionMatrixDisplay(cm).plot()
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(output_dir, "confusion_matrix.png"))
    plt.close()

    # ROC Curve and AUC
    if y_prob is not None:
        RocCurveDisplay.from_predictions(y_test, y_prob)
        plt.title("ROC Curve")
        plt.savefig(os.path.join(output_dir, "roc_curve.png"))
        plt.close()

    # Precision-Recall Curve
    if y_prob is not None:
        PrecisionRecallDisplay.from_predictions(y_test, y_prob)
        plt.title("Precision-Recall Curve")
        plt.savefig(os.path.join(output_dir, "precision_recall_curve.png"))
        plt.close()

    # Calibration Curve
    if y_prob is not None:
        prob_true, prob_pred = calibration_curve(y_test, y_prob, n_bins=10)
        plt.plot(prob_pred, prob_true, marker='o')
        plt.plot([0, 1], [0, 1], linestyle="--", color='gray')
        plt.title("Calibration Curve")
        plt.xlabel("Mean Predicted Probability")
        plt.ylabel("Fraction of Positives")
        plt.savefig(os.path.join(output_dir, "calibration_curve.png"))
        plt.close()

    # Feature Importance or Coefficients (for applicable models)
    if hasattr(model, 'feature_importances_'):
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]
        plt.figure()
        plt.title("Feature Importances")
        plt.bar(range(X_test.shape[1]), importances[indices], align="center")
        plt.xticks(range(X_test.shape[1]), [X_test.columns[i] for i in indices], rotation=90)
        plt.tight_layout()  # Adjust layout to prevent label cutoff
        feature_importance_path = os.path.join(output_dir, "feature_importance.png")
        plt.savefig(feature_importance_path)
        plt.close()
        print(f"Feature importance plot saved for {model} at {feature_importance_path}")
    elif hasattr(model, 'coef_'):
        importances = np.abs(model.coef_).flatten()
        indices = np.argsort(importances)[::-1]
        plt.figure()
        plt.title("Feature Coefficients (Importance)")
        plt.bar(range(X_test.shape[1]), importances[indices], align="center")
        plt.xticks(range(X_test.shape[1]), [X_test.columns[i] for i in indices], rotation=90)
        plt.tight_layout()
        feature_coefficients_path = os.path.join(output_dir, "feature_coefficients.png")
        plt.savefig(feature_coefficients_path)
        plt.close()
        print(f"Feature coefficients plot saved for {model} at {feature_coefficients_path}")
    else:
        print(f"No feature importance or coefficients available for model {model}.")

    # Heatmap of Predictions vs True Values
    sns.heatmap(pd.DataFrame({"True": y_test, "Predicted": y_pred}).pivot_table(index='True', columns='Predicted', aggfunc=len, fill_value=0), annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Heatmap of Predictions vs True Values")
    plt.savefig(os.path.join(output_dir, "heatmap_prediction_vs_true.png"))
    plt.close()

    # Scatterplot with Predicted Probabilities
    if y_prob is not None:
        plt.scatter(range(len(y_test)), y_prob, c=y_test, cmap="bwr", alpha=0.6)
        plt.title("Scatterplot of Predicted Probabilities")
        plt.xlabel("Sample Index")
        plt.ylabel("Predicted Probability")
        plt.colorbar(label="True Value")
        plt.savefig(os.path.join(output_dir, "scatterplot_predicted_probabilities.png"))
        plt.close()

    # Metrics
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred, zero_division=0),
        'recall': recall_score(y_test, y_pred, zero_division=0),
        'f1_score': f1_score(y_test, y_pred, zero_division=0),
        'roc_auc': roc_auc_score(y_test, y_prob) if y_prob is not None else 'N/A',
        'confusion_matrix': cm.tolist(),
        'classification_report': classification_report(y_test, y_pred, zero_division=0)
    }
    return metrics

def main():
    input_folder = "cleaned_data"
    model_folder = "model_outputs"
    output_folder = "visualize_output"

    test_file = select_testing_file(input_folder)
    if not test_file:
        return

    X_test, y_test = load_data(test_file)
    models = load_models(model_folder)
    if not models:
        print("No models found in model_outputs folder.")
        return

    training_features = None
    for model in models.values():
        try:
            training_features = model.named_steps['model'].feature_names_in_
            break
        except AttributeError:
            continue

    if training_features is None:
        print("Using test data columns as a fallback for features.")
        training_features = X_test.columns

    X_test = preprocess_test_data(X_test, training_features)

    report = []
    for model_name, model in models.items():
        model_output_dir = os.path.join(output_folder, model_name)
        metrics = evaluate_model(model, X_test, y_test, model_output_dir)
        report.append((model_name, metrics))
        # print(f"Model: {model_name} - Metrics: {metrics}")

        # Format and print the metrics
        print(f"\n{'='*40}")
        print(f"Model: {model_name}")
        print(f"{'-'*40}")
        print("Classification Report:")
        print(metrics['classification_report'])
        print(f"Accuracy       : {metrics['accuracy']:.4f}")
        if isinstance(metrics['roc_auc'], str):
            print(f"ROC AUC        : {metrics['roc_auc']}")
        else:
            print(f"ROC AUC        : {metrics['roc_auc']:.4f}")
        print("Confusion Matrix:")
        print(metrics['confusion_matrix'])
        print(f"{'='*40}\n")

    report_path = os.path.join(output_folder, 'prediction_report.csv')
    report_df = pd.DataFrame([{**{'Model': m}, **metrics} for m, metrics in report])
    report_df.to_csv(report_path, index=False)
    print(f"Prediction report with confusion matrices saved to {report_path}")

if __name__ == "__main__":
    main()
