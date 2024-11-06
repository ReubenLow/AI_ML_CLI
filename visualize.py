import pandas as pd
import joblib
import argparse
from sklearn.metrics import classification_report
from sklearn.preprocessing import LabelEncoder

def preprocess_data(data):
    # Remove $ sign and commas from Passenger Fare and convert to float
    data['Passenger Fare'] = data['Passenger Fare'].replace({'\\$': '', ',': ''}, regex=True).astype(float)
    
    # Encode categorical columns
    le = LabelEncoder()
    for col in ['Ticket Class', 'Embarkation Country', 'Gender']:
        data[col] = le.fit_transform(data[col].astype(str))
    
    # Drop irrelevant columns
    data = data.drop(['Passenger ID', 'Ticket Number', 'Cabin', 'Name'], axis=1)
    return data

def visualize_model_performance(model_path, test_data_path):
    # Load model and data
    model = joblib.load(model_path)
    data = pd.read_csv(test_data_path)
    
    # Encode the target label 'Survived' to match the prediction format
    le = LabelEncoder()
    data['Survived'] = le.fit_transform(data['Survived'].astype(str))  # Convert 'Yes', 'No' to 1, 0
    
    # Drop the target column 'Survived' before prediction
    y = data['Survived']
    X = preprocess_data(data.drop('Survived', axis=1))

    # Predict and evaluate
    y_pred = model.predict(X)
    print("Test set report:\n", classification_report(y, y_pred))

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Visualize model performance on test data.")
    parser.add_argument('--model', type=str, help="Path to the trained model.")
    parser.add_argument('--data', type=str, help="Path to the test dataset.")
    
    args = parser.parse_args()
    visualize_model_performance(args.model, args.data)
