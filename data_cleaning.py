import pandas as pd
from sklearn.preprocessing import LabelEncoder

# Load the dataset
file_path = "StudentPerformanceFactors.csv"
data = pd.read_csv(file_path)

# Check for missing values before cleaning
print("\nMissing Values Per Column Before Cleaning:")
print(data.isnull().sum())

# Handle missing values
for column in data.columns:
    if data[column].dtype == 'object':
        # Fill missing values with the mode for categorical columns
        data[column].fillna(data[column].mode()[0], inplace=True)
    else:
        # Fill missing values with the mean for numerical columns
        data[column].fillna(data[column].mean(), inplace=True)

# Check for missing values after cleaning
print("\nMissing Values Per Column After Cleaning:")
print(data.isnull().sum())

# Identify categorical columns to encode
categorical_columns = ['Parental_Involvement', 'Parental_Education_Level', 
                       'Distance_from_Home', 'Gender']

# Encode categorical columns
label_encoders = {}
for col in categorical_columns:
    le = LabelEncoder()
    data[col] = le.fit_transform(data[col])
    label_encoders[col] = le  # Save the encoder for consistent future use

# Save the cleaned and encoded dataset to a new CSV
output_file_path = "encoded_student_performance.csv"
data.to_csv(output_file_path, index=False)
print(f"\nCategorical columns encoded and cleaned dataset saved to: {output_file_path}")
