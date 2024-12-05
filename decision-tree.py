import pandas as pd
import numpy as np

from sklearn.tree import DecisionTreeRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import seaborn as sns

# Step 1: Load the dataset
data = pd.read_csv(r'C:\Users\shaun\PycharmProjects\pythonProject\StudentPerformanceFactors.csv')

# Step 2: Drop all rows with missing values and create a clean copy
data_cleaned = data.dropna().copy()

# Step 3: Define mappings and safely apply them
mappings = {
    'Parental_Involvement': {'Low': 1, 'Medium': 2, 'High': 3},
    'Access_to_Resources': {'Low': 1, 'Medium': 2, 'High': 3},
    'Motivation_Level': {'Low': 1, 'Medium': 2, 'High': 3},
    'Family_Income': {'Low': 1, 'Medium': 2, 'High': 3},
    'Teacher_Quality': {'Low': 1, 'Medium': 2, 'High': 3},
    'Distance_from_Home': {'Far': 1, 'Moderate': 2, 'Near': 3},
    'Peer_Influence': {'Negative': 1, 'Neutral': 2, 'Positive': 3},
    'Parental_Education_Level': {'High School': 1, 'College': 2, 'Postgraduate': 3},
    'Internet_Access': {'No': 0, 'Yes': 1},
    'Tutoring_Sessions': {'No': 0, 'Yes': 1},
    'School_Type': {'Public': 0, 'Private': 1},
    'Learning_Disabilities': {'No': 0, 'Yes': 1},
    'Gender': {'Male': 0, 'Female': 1},
    'Extracurricular_Activities': {'No': 0, 'Yes': 1}
}

# Apply the mappings to the appropriate columns using a safe method
for column, mapping in mappings.items():
    if column in data_cleaned.columns:
        # Map and convert to float to avoid dtype warnings
        data_cleaned[column] = data_cleaned[column].map(mapping).astype(float)

# Step 3.1: Fill any remaining missing values after mapping
data_cleaned = data_cleaned.fillna(0)

# Step 4: Define the target and features
X = data_cleaned.drop(columns=['Exam_Score'])  # Assuming 'Exam_Score' is the target variable
y = data_cleaned['Exam_Score']

# Step 5: Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.20, random_state=42)

# Step 6: Train the Decision Tree Regressor (Feature Scaling removed for Decision Trees)
model = DecisionTreeRegressor(max_depth=5, random_state=42)
model.fit(X_train, y_train)

# Step 7: Make predictions on the test set
y_pred = model.predict(X_test)

# Step 8: Evaluate the model
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R^2 Score: {r2}")

# Step 9: Visualize the Decision Tree
plt.figure(figsize=(24, 12))
plot_tree(model, feature_names=X.columns, filled=True, rounded=True, fontsize=12)
plt.tight_layout()
plt.show()

# Step 10: Feature Importance
feature_importances = model.feature_importances_
features = X.columns

plt.figure(figsize=(12, 8))
sns.barplot(x=feature_importances, y=features, palette='viridis', legend=False)
plt.xlabel('Importance', fontsize=14)
plt.ylabel('Feature', fontsize=14)
plt.title('Feature Importance in Predicting Exam Scores', fontsize=16)
plt.tight_layout()
plt.show()

plt.figure(figsize=(12, 10))
correlation_matrix = data_cleaned.corr()
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap', fontsize=16)
plt.tight_layout()
plt.show()
