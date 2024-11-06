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
from sklearn.metrics import accuracy_score, log_loss, mean_squared_error,confusion_matrix, precision_score, recall_score, auc,roc_curve, roc_auc_score
from xgboost import XGBClassifier
import joblib
import configparser

config_file = "model_params.txt"

def load_model_params(config_file, model_name):
    config = configparser.ConfigParser()
    config.read(config_file)

    def safe_eval(value):
        # Attempt to evaluate the value, return as a string if it fails
        try:
            return eval(value)
        except NameError:
            return value

    # Return the parsed and safely evaluated parameters
    return {key: safe_eval(value) for key, value in config[model_name].items()}


"""
RandomForestClassifier
n_estimators: Increasing this (e.g., 200 or 300) could improve performance but will increase computation time.
max_depth: Try deeper trees (e.g., 10 or 15) or None (fully grown trees), which could capture more complex patterns but might overfit.
min_samples_split: Increasing to 5 or 10 could help reduce overfitting.
min_samples_leaf: Setting to a higher value (e.g., 2 or 5) can prevent the model from learning too finely on noise.

AdaBoostClassifier
n_estimators: Increasing the number of estimators (e.g., 100 or 200) might improve accuracy.
learning_rate: Try lowering the learning rate (e.g., 0.5) to reduce overfitting when using more estimators.

GradientBoostingClassifier
n_estimators: Increase the number of estimators (e.g., 150 or 200).
learning_rate: Adjusting this to a lower value (e.g., 0.1 or 0.05) could stabilize learning.
max_depth: Increase to 5-7 to capture more complex patterns.
min_samples_split/min_samples_leaf: Similar to RandomForest, increasing these values can help reduce overfitting.

KNeighborsClassifier
n_neighbors: Try adjusting the number of neighbors (e.g., 3, 5, or 7) to balance bias and variance.
weights: Using 'distance' weights rather than 'uniform' might help with predictions by giving closer neighbors more influence.

SVC (Support Vector Classifier)
C: Increase or decrease the regularization parameter (e.g., 0.1, 1, 10).
kernel: Experiment with 'rbf' or 'poly' for non-linear relationships.
gamma: Use 'scale' or manually set it (e.g., 0.1, 0.01) for the 'rbf' kernel.

DecisionTreeClassifier
max_depth: Set this to a value between 5 and 15 to prevent overfitting.
min_samples_split/min_samples_leaf: Increasing these values can help reduce overfitting, similar to RandomForest.

LogisticRegression
penalty: Try using different penalties ('l2', 'l1') to see which regularization works best.
C: Adjust the regularization strength (e.g., 0.1, 1, 10) to prevent overfitting.

NaiveBayes (GaussianNB)
var_smoothing: Adjust this parameter (e.g., 1e-8, 1e-9) for handling noisy features.

NeuralNetwork (MLPClassifier)
hidden_layer_sizes: Increase the number of neurons or layers (e.g., (150, 100) or (100, 50, 50)).
learning_rate_init: Lower learning rates (e.g., 0.001) could help stabilize training.
alpha: Regularization parameter, adjust to reduce overfitting (e.g., 0.0001, 0.001).

XGBoost
n_estimators: Increase the number of estimators (e.g., 200).
learning_rate: Try reducing to 0.1 or 0.05 for a more gradual learning process.
max_depth: Increase to capture more complex patterns (e.g., 6 or 8).
subsample: Set to a value below 1 (e.g., 0.8) to introduce more randomness and reduce overfitting.
"""

# MODELS = {
#     "RandomForest": RandomForestClassifier(n_estimators=400, max_depth=12, min_samples_split=2, min_samples_leaf=1, random_state=42),
#     "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.5),
#     "GradientBoosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=12, min_samples_split=10, min_samples_leaf=5),
#     "KNeighbors": KNeighborsClassifier(n_neighbors=5, weights='distance'),
#     "SVC": SVC(C=1, kernel='rbf', gamma='scale'),
#     "DecisionTree": DecisionTreeClassifier(max_depth=15, min_samples_split=5, min_samples_leaf=2),
#     "LogisticRegression": LogisticRegression(max_iter=200, penalty='l2', C=1.0),
#     "NaiveBayes": GaussianNB(var_smoothing=1e-9),
#     "NeuralNetwork": MLPClassifier(hidden_layer_sizes=(100, 50, 50, 50), max_iter=800, learning_rate_init=0.001, alpha=0.0001),
#     "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=500, learning_rate=0.05, max_depth=8, subsample=0.8)
# }

# MODELS = {
#     "RandomForest": RandomForestClassifier(n_estimators=200, max_depth=20, min_samples_split=2, min_samples_leaf=1, random_state=42),
#     "AdaBoost": AdaBoostClassifier(n_estimators=100, learning_rate=0.5),
#     "GradientBoosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.05, max_depth=20, min_samples_split=10, min_samples_leaf=5),
#     "KNeighbors": KNeighborsClassifier(n_neighbors=5, weights='distance'),
#     "SVC": SVC(C=1, kernel='rbf', gamma='scale'),
#     "DecisionTree": DecisionTreeClassifier(max_depth=15, min_samples_split=5, min_samples_leaf=2),
#     "LogisticRegression": LogisticRegression(max_iter=200, penalty='l2', C=1.0),
#     "NaiveBayes": GaussianNB(var_smoothing=1e-9),
#     "NeuralNetwork": MLPClassifier(hidden_layer_sizes=(100, 50, 50, 50), max_iter=800, learning_rate_init=0.001, alpha=0.0001),
#     "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=500, learning_rate=0.05, max_depth=8, subsample=0.8)
# }

MODELS = {
    "RandomForest": RandomForestClassifier(**load_model_params(config_file, "RandomForest")),
    "AdaBoost": AdaBoostClassifier(**load_model_params(config_file, "AdaBoost")),
    "GradientBoosting": GradientBoostingClassifier(**load_model_params(config_file, "GradientBoosting")),
    "KNeighbors": KNeighborsClassifier(**load_model_params(config_file, "KNeighbors")),
    "SVC": SVC(**load_model_params(config_file, "SVC")),
    "DecisionTree": DecisionTreeClassifier(**load_model_params(config_file, "DecisionTree")),
    "LogisticRegression": LogisticRegression(**load_model_params(config_file, "LogisticRegression")),
    "NaiveBayes": GaussianNB(**load_model_params(config_file, "NaiveBayes")),
    "NeuralNetwork": MLPClassifier(**load_model_params(config_file, "NeuralNetwork")),
    "XGBoost": XGBClassifier(**load_model_params(config_file, "XGBoost"))
}


# MODELS = {
#     "RandomForest": RandomForestClassifier(n_estimators=300, max_depth=20, min_samples_split=2, min_samples_leaf=1, random_state=42),
#     "GradientBoosting": GradientBoostingClassifier(n_estimators=300, learning_rate=0.1, max_depth=12, min_samples_split=10, min_samples_leaf=5),
#     "KNeighbors": KNeighborsClassifier(n_neighbors=5, weights='distance'),
#     "LogisticRegression": LogisticRegression(max_iter=200, penalty='l2', C=1.0),
#     "NaiveBayes": GaussianNB(var_smoothing=1e-9),
#     "XGBoost": XGBClassifier(use_label_encoder=False, eval_metric='logloss', n_estimators=500, learning_rate=0.05, max_depth=8, subsample=0.8)
# }

# MODELS = {
#     "RandomForest": RandomForestClassifier(n_estimators=150, max_depth=None, min_samples_split=2, min_samples_leaf=1, random_state=42)
# }



def train_and_save_models(X, y, model_output):
    if not os.path.exists(model_output):
        os.makedirs(model_output)

    # Identify categorical and numerical columns
    categorical_cols = X.select_dtypes(include=['object', 'category']).columns
    numeric_cols = X.select_dtypes(include=[np.number]).columns

    # Create a column transformer to handle both categorical and numeric columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numeric_cols),
            ('cat', OneHotEncoder(), categorical_cols)
        ])

    for model_name, model in MODELS.items():
        pipeline = Pipeline([
            ('preprocessor', preprocessor),  # Preprocess categorical and numeric data
            ('model', model)
        ])

        # Fit the pipeline with preprocessed data
        pipeline.fit(X, y)

        # Save the trained model
        model_path = os.path.join(model_output, f"{model_name}_model.pkl")
        joblib.dump(pipeline, model_path)
        print(f"Trained and saved model: {model_name} to {model_path}")

# def select_features(data):
#     print("Available features:")
#     for idx, column in enumerate(data.columns):
#         print(f"{idx + 1}: {column}")
#     selected = input("Enter the feature numbers to use (comma-separated) or type 'all' to select all: ")
    
#     if selected.lower() == 'all':
#         return data
#     else:
#         selected_indices = [int(i) - 1 for i in selected.split(',')]
#         return data.iloc[:, selected_indices]
def select_features(data):
    print("Available features:")
    for idx, column in enumerate(data.columns):
        print(f"{idx + 1}: {column}")
    selected = input("Enter the feature numbers to use (comma-separated), type 'all' to select all, or press Enter for default (2,3,4,5,6): ")
    
    if selected.lower() == 'all':
        return data
    elif selected.strip() == '':  # Default option if no input
        default_indices = [1, 2, 3, 4, 5]  # 2,3,4,5,6 are indices 1,2,3,4,5 (0-based)
        return data.iloc[:, default_indices]
    else:
        selected_indices = [int(i) - 1 for i in selected.split(',')]
        return data.iloc[:, selected_indices]

def normalize_features(data):
    print("\nAvailable features for normalization:")
    for idx, column in enumerate(data.columns):
        print(f"{idx + 1}: {column}")
    normalize = input("Enter the feature numbers to normalize (comma-separated) or press Enter to skip: ")
    
    if normalize:
        selected_indices = [int(i) - 1 for i in normalize.split(',')]
        selected_features = data.columns[selected_indices]
        for feature in selected_features:
            data[feature] = np.log1p(data[feature]) # Log transformation
            print(f"Applied log transformation on {feature}")
    return data

def select_training_file(input_folder):
    files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    if not files:
        print("No CSV files found in the cleaned_data folder.")
        return None
    print("Available files for model training:")
    for idx, file in enumerate(files, 1):
        print(f"{idx}: {file}")
    choice = int(input("Select the file number to train the model on: ")) - 1
    return os.path.join(input_folder, files[choice])

def main(input_folder, model_output, config_file="model_params.txt"):

    # List available files in cleaned_data folder
    files = [f for f in os.listdir(input_folder) if f.endswith('.csv')]
    print("Available files in cleaned_data:")
    for idx, file in enumerate(files, 1):
        print(f"{idx}: {file}")
    
    # Select file
    file_choice = int(input("Select the file number to train on: ")) - 1
    selected_file = files[file_choice]
    data_path = os.path.join(input_folder, selected_file)
    data = pd.read_csv(data_path)

    # Select features
    data = select_features(data)
    
    # Separate features and target variable
    # target_col = input("Enter the target column (label) by name for training (e.g., 'Survived'): ")
    target_col = "Survived"
    if target_col not in data.columns:
        raise ValueError(f"The specified target column '{target_col}' does not exist in the data.")

    X = data.drop(columns=[target_col])
    y = data[target_col]

    # Normalize features if needed
    X = normalize_features(X)
    
    # Train and save models
    train_and_save_models(X, y, model_output)
    print("Training completed and models saved.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train various models on a selected dataset.")
    parser.add_argument('--input_folder', type=str, default="cleaned_data", help="Path to the cleaned data folder.")
    parser.add_argument('--model_output', type=str, default="model_outputs", help="Path to save the trained models.")
    parser.add_argument('--config_file', type=str, default="model_params.txt", help="Path to the model parameter file.")
    
    args = parser.parse_args()
    main(args.input_folder, args.model_output)
