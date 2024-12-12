import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_val_score


data = pd.read_csv('/Users/sai/Downloads/StudentPerformanceFactors.csv')
# print(data.head())
data.drop(columns=['Teacher_Quality', 'Parental_Education_Level', 'Distance_from_Home'], inplace = True)
# print(data.isnull().sum())

X = data.drop('Exam_Score', axis = 1)
y = data['Exam_Score']
X = pd.get_dummies(X, drop_first = True)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state = 42)

randomForest_model = RandomForestRegressor(n_estimators = 100, random_state = 42)
randomForest_model.fit(X_train, y_train)
y_pred = randomForest_model.predict(X_test)



feature_importance = randomForest_model.feature_importances_
feature_importance_df = pd.DataFrame({
    'Feature': X.columns,
    'Importance': feature_importance
}).sort_values(by='Importance', ascending=False)

mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

# Print MSE and R^2
print(f"Mean Squared Error (MSE): {mse:.2f}")
print(f"R-squared (R^2) Score: {r2:.2f}")

# Scatter plot for Predicted vs Actual
plt.figure(figsize=(8, 6))
plt.scatter(y_test, y_pred, alpha=0.6, color='blue')
plt.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], color='red', linestyle='--', linewidth=2)
plt.title('Actual vs Predicted Values')
plt.xlabel('Actual Exam Scores')
plt.ylabel('Predicted Exam Scores')
plt.grid(True)
plt.show()

plt.figure(figsize=(10, 8))
plt.barh(feature_importance_df['Feature'], feature_importance_df['Importance'], color='skyblue')
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.grid(True, axis='x', linestyle='--', linewidth=0.7)
plt.gca().invert_yaxis()  # Invert y-axis to show the most important feature on top
plt.show()

# Plot the first tree in the forest
plt.figure(figsize=(20, 10))
plot_tree(randomForest_model.estimators_[0], feature_names=X.columns, filled=True, max_depth=3)
plt.show()
