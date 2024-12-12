import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.tree import plot_tree
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV


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

# Define parameter grid
param_grid = {
    'n_estimators': [100, 200, 300],
    'max_depth': [None, 10, 20, 30],
    'min_samples_split': [2, 5, 10],
    'min_samples_leaf': [1, 2, 4]
}

# Grid search
grid_search = GridSearchCV(
    estimator=RandomForestRegressor(random_state=42),
    param_grid=param_grid,
    scoring='neg_mean_squared_error',
    cv=5,
    n_jobs=-1
)

grid_search.fit(X_train, y_train)
best_model = grid_search.best_estimator_

# Evaluate the best model
y_pred = best_model.predict(X_test)
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Best Hyperparameters:", grid_search.best_params_)
print(f"Improved MSE: {mse:.2f}")
print(f"Improved R^2: {r2:.2f}")

top_features = feature_importance_df['Feature'][:10]
X_train_top = X_train[top_features]
X_test_top = X_test[top_features]

best_model.fit(X_train_top, y_train)
y_pred_top = best_model.predict(X_test_top)

mse_top = mean_squared_error(y_test, y_pred_top)
r2_top = r2_score(y_test, y_pred_top)

print(f"Improved MSE with Top Features: {mse_top:.2f}")
print(f"Improved R^2 with Top Features: {r2_top:.2f}")