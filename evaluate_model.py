from sklearn.metrics import mean_squared_error, r2_score

# Load cleaned and encoded data
import pandas as pd
file_path = 'encoded_student_performance.csv'  # Use the encoded dataset
data = pd.read_csv(file_path)

# Define features (X) and target (y)
X = data[['Hours_Studied', 'Attendance', 'Parental_Involvement', 'Sleep_Hours']]
y = data['Exam_Score']

# Train-test split
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train the model
from sklearn.linear_model import LinearRegression
model = LinearRegression()
model.fit(X_train, y_train)

# Predict on test set
y_pred = model.predict(X_test)

# Calculate MSE and R²
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print results
print(f"Mean Squared Error (MSE): {mse}")
print(f"R-squared (R²): {r2}")
