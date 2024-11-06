import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score
from scipy.stats import shapiro, kstest, norm, probplot, chi2_contingency
import argparse
pd.set_option('display.max_rows', None)  # Display all rows

# Clean data function
def clean_data(data):
    # Extract title from name
    if 'Name' in data.columns:
        data['Title'] = data['Name'].str.extract(r',\s*([^\.]*)\s*\.', expand=False)

    # Drop irrelevant columns
    data = data.drop(['Passenger ID', 'Ticket Number', 'Cabin', 'Name'], axis=1, errors='ignore')

    if 'Survived' in data.columns:
        data['Survived'] = data['Survived'].replace({'Yes': 1, 'No': 0})  # Replace yes/no with 1/0
        data['Survived'] = data['Survived'].astype(int)  # Ensure Survived is integer

    # Remove $ sign and commas from 'Passenger Fare' and convert to float
    data['Passenger Fare'] = data['Passenger Fare'].replace({'\\$': '', ',': ''}, regex=True).astype(float)
    data['Passenger Fare'] = data['Passenger Fare'].round(2)

    # Fill zero ages with nan to be processed later
    data['Age'] = data['Age'].replace(0, np.nan)

    title_age_medians = data.groupby('Title')['Age'].median().round().to_dict()

    # Define name titles
    # title_age_medians = {
    #     'Mr': data[data['Title'] == 'Mr']['Age'].median(),
    #     'Mrs': data[data['Title'] == 'Mrs']['Age'].median(),
    #     'Miss': data[data['Title'] == 'Miss']['Age'].median(),
    #     'Master': data[data['Title'] == 'Master']['Age'].median(),
    #     'Dr': data[data['Title'] == 'Mr']['Age'].median(),  # Assume Dr is adult male
    #     'Sir': data[data['Title'] == 'Mr']['Age'].median(),
    #     'Lady': data[data['Title'] == 'Mrs']['Age'].median(),
    #     'Ms': data[data['Title'] == 'Miss']['Age'].median(),
    #     'Mme': data[data['Title'] == 'Mrs']['Age'].median(),
    #     'Mlle': data[data['Title'] == 'Miss']['Age'].median(),
    #     'Rev': data[data['Title'] == 'Mr']['Age'].median(),
    #     'Countess': data[data['Title'] == 'Mrs']['Age'].median(),
    #     'Col': data[data['Title'] == 'Mr']['Age'].median(),
    #     'Major': data[data['Title'] == 'Mr']['Age'].median()
    # }

    # Fill missing embarkation codes by checking for family members
    data['Embarkation Country'] = data['Embarkation Country'].astype(str).str.strip()
    data['Embarkation Country'].replace(
        to_replace=r'^\s*0+\s*$',
        value=np.nan,
        regex=True,
        inplace=True
    )
    # Fill missing embarkation codes with mode
    # data['Embarkation Country'] = data['Embarkation Country'].astype(str).str.strip()
    # data['Embarkation Country'].replace(to_replace=r'^\s*0+\s*$', value=np.nan, regex=True, inplace=True)
    # data['Embarkation Country'].replace('', np.nan, inplace=True)
    # embark_mode = data['Embarkation Country'].mode(dropna=True).iloc[0]
    # data['Embarkation Country'].fillna(embark_mode, inplace=True)
    embark_mode = data['Embarkation Country'].mode(dropna=True).iloc[0]
    data['Embarkation Country'].replace(to_replace=r'^\s*0+\s*$', value=embark_mode, regex=True, inplace=True)
    data['Embarkation Country'].fillna(embark_mode, inplace=True)

    # Fill missing ages based on title medians, ensure that age is at least 1 since original train data has weird ages like 0.73 lol
    data['Age'] = data.apply(lambda row: max(1, int(np.ceil(title_age_medians[row['Title']]))) if pd.isnull(row['Age']) else max(1, int(np.ceil(row['Age']))), axis=1)

    # Create age groups after filling missing ages
    data['Age Group'] = pd.cut(data['Age'], bins=[0, 18, 35, 50, 65, 100], labels=['0-18', '19-35', '36-50', '51-65', '66+'])

    # Handle missing data
    imputer = SimpleImputer(strategy='mean')
    data['Passenger Fare'] = imputer.fit_transform(data[['Passenger Fare']])

    # Retain original Ticket Class for grouping
    original_ticket_class = data['Ticket Class']
    original_embarkation = data['Embarkation Country']

    # Convert Gender to numerical: male=1, female=0 for model training purposes
    data['Gender_num'] = data['Gender'].map({'male': 1, 'female': 0})

    # Convert Embarkation Country to numerical for model training purposes
    embarkation_map = {'C': 1, 'Q': 2, 'S': 3}
    data['Embarkation_Country_num'] = data['Embarkation Country'].map(embarkation_map)

    # Encode categorical columns with OneHotEncoder
    categorical_cols = ['Ticket Class', 'Embarkation Country']
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    encoded_features = pd.DataFrame(encoder.fit_transform(data[categorical_cols]))
    encoded_features.columns = encoder.get_feature_names_out(categorical_cols)

    # Drop original categorical columns and add encoded features
    data = data.drop(categorical_cols, axis=1)
    data = pd.concat([data, encoded_features], axis=1)

    # Add the original 'Ticket Class' back for grouping
    data['Ticket Class'] = original_ticket_class
    data['Embarkation Country'] = original_embarkation

    # data['Embarkation Country'].replace(0, data['Embarkation Country'].mode()[0], inplace=True)
    # Ensure 0 values in 'Embarkation Country' are replaced with the mode using for loop to explicitly replace 0 with common letter
    # for idx, row in data[data['Embarkation Country'] == 0].iterrows():
    #     data.at[idx, 'Embarkation Country'] = data['Embarkation Country'].mode().iloc[0]
    return data


def main(input_folder="training_data", output_folder="cleaned_data", selected_file=None):
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)
    
    # If no file specified, display available files and prompt for selection
    if selected_file is None:
        files = [f for f in os.listdir(input_folder) if f.endswith(".csv")]
        if not files:
            print("No files found in the training_data folder.")
            return
        
        print("\nAvailable files in training_data folder (please wait for a while for files to load):")
        for i, file in enumerate(files, 1):
            print(f"{i}. {file}")
        
        try:
            choice = int(input("Enter the number of the file you want to clean: "))
            if 1 <= choice <= len(files):
                selected_file = files[choice - 1]
            else:
                print("Invalid selection.")
                return
        except ValueError:
            print("Please enter a valid number.")
            return

    input_path = os.path.join(input_folder, selected_file)
    data = pd.read_csv(input_path)

    # Clean the data
    cleaned_data = clean_data(data)
    cleaned_data = cleaned_data.drop(columns=['Title'], errors='ignore')

    # Save to the output folder cleaned_data
    output_path = os.path.join(output_folder, f"cleaned_{selected_file}")
    cleaned_data.to_csv(output_path, index=False)
    print(f"Saved cleaned data to {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Clean specific data file in the training_data folder.")
    parser.add_argument('--input_folder', type=str, default="training_data", help="Folder containing raw data files.")
    parser.add_argument('--output_folder', type=str, default="cleaned_data", help="Folder to save cleaned data files.")
    parser.add_argument('--file', type=str, default=None, help="Specific file to clean.")
    args = parser.parse_args()
    main(args.input_folder, args.output_folder, args.file)
