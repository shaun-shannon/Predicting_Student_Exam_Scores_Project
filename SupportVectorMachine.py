import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.impute import SimpleImputer

df = pd.read_csv("StudentPerformanceFactors.csv")

df['Teacher_Quality'] = df['Teacher_Quality'].fillna(df['Teacher_Quality'].mode()[0])
df['Parental_Education_Level'] = df['Parental_Education_Level'].fillna(df['Parental_Education_Level'].mode()[0])
df['Distance_from_Home'] = df['Distance_from_Home'].fillna(df['Distance_from_Home'].mode()[0])

X = df.drop(columns=['Exam_Score'])
y = df['Exam_Score']

categorical_features = ['Parental_Involvement', 'Access_to_Resources', 'Extracurricular_Activities', 
                        'Motivation_Level', 'Internet_Access', 'Family_Income', 'Teacher_Quality', 
                        'School_Type', 'Peer_Influence', 'Learning_Disabilities', 'Parental_Education_Level', 
                        'Distance_from_Home', 'Gender']
numerical_features = ['Hours_Studied', 'Attendance', 'Sleep_Hours', 'Previous_Scores', 
                      'Tutoring_Sessions', 'Physical_Activity']

categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='most_frequent')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))
])

numerical_transformer = Pipeline(steps=[
    ('scaler', StandardScaler())
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numerical_transformer, numerical_features),
        ('cat', categorical_transformer, categorical_features)
    ]
)

svm_pipeline = Pipeline(steps=[
    ('preprocessor', preprocessor),
    ('regressor', SVR(kernel='rbf', C=100, gamma=0.1, epsilon=0.1))
])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

svm_pipeline.fit(X_train, y_train)

y_pred = svm_pipeline.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print(f"Mean Squared Error: {mse}")
print(f"R-squared: {r2}")

X_test['Predicted_Exam_Score'] = y_pred
X_test['Actual_Exam_Score'] = y_test.values
X_test.to_csv("test_predictions.csv", index=False)
print("Test predictions saved as 'test_predictions.csv'.")
