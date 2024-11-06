import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import argparse
from sklearn.preprocessing import LabelEncoder

def eda(data_path, output_folder):
    # Load data
    data = pd.read_csv(data_path)

    # Remove $ sign and commas from Passenger Fare and convert to float
    data['Passenger Fare'] = data['Passenger Fare'].replace({'\\$': '', ',': ''}, regex=True).astype(float)

    # Encode categorical columns
    le = LabelEncoder()
    for col in ['Embarkation Country', 'Gender', 'Survived']:
        data[col] = le.fit_transform(data[col].astype(str))

    # Drop irrelevant columns for correlation analysis
    data = data.drop(['Passenger ID', 'Ticket Number', 'Cabin', 'Name'], axis=1)

    # Descriptive statistics
    print(data.describe())
    
    # Missing values
    print("Missing Values:\n", data.isnull().sum())

    # Correlation heatmap
    plt.figure(figsize=(10, 8))
    sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
    plt.title("Correlation Heatmap")
    plt.savefig(f"{output_folder}/correlation_heatmap.png")
    plt.close()

    # Distribution plots
    data.hist(figsize=(12, 10), bins=30)
    plt.tight_layout()
    plt.savefig(f"{output_folder}/distribution.png")
    plt.close()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Perform EDA on a dataset.")
    parser.add_argument('--data', type=str, help="Path to the dataset.")
    parser.add_argument('--output', type=str, help="Folder to save the EDA outputs.")
    
    args = parser.parse_args()
    eda(args.data, args.output)
