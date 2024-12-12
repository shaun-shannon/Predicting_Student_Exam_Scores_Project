import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

test_predictions = pd.read_csv("test_predictions.csv")

plt.figure(figsize=(10, 6))
plt.scatter(test_predictions['Actual_Exam_Score'], test_predictions['Predicted_Exam_Score'], alpha=0.7, label="Data Points")
plt.plot([test_predictions['Actual_Exam_Score'].min(), test_predictions['Actual_Exam_Score'].max()],
         [test_predictions['Actual_Exam_Score'].min(), test_predictions['Actual_Exam_Score'].max()],
         color='red', linestyle='--', linewidth=2, label='Ideal Fit')
plt.title('Actual vs. Predicted Exam Scores', fontsize=16)
plt.xlabel('Actual Exam Score', fontsize=14)
plt.ylabel('Predicted Exam Score', fontsize=14)
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.residplot(x='Actual_Exam_Score', y='Predicted_Exam_Score', data=test_predictions, lowess=True, 
              line_kws={'color': 'red'})
plt.title("Residual Plot: Actual vs Predicted Exam Scores", fontsize=16)
plt.xlabel("Actual Exam Score", fontsize=14)
plt.ylabel("Residuals (Actual - Predicted)", fontsize=14)
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
residuals = test_predictions['Actual_Exam_Score'] - test_predictions['Predicted_Exam_Score']
sns.histplot(residuals, kde=True, bins=30, color='blue', edgecolor='black')
plt.title("Distribution of Residuals", fontsize=16)
plt.xlabel("Residuals (Actual - Predicted)", fontsize=14)
plt.ylabel("Frequency", fontsize=14)
plt.axvline(0, color='red', linestyle='--', linewidth=2, label='Zero Residual')
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()

plt.figure(figsize=(8, 6))
correlation_matrix = test_predictions[['Actual_Exam_Score', 'Predicted_Exam_Score']].corr()
sns.heatmap(correlation_matrix, annot=True, cmap="coolwarm", fmt=".2f")
plt.title("Correlation Heatmap: Actual vs Predicted Exam Scores", fontsize=16)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))
sns.residplot(x='Actual_Exam_Score', y='Predicted_Exam_Score', data=test_predictions, lowess=True, 
              line_kws={'color': 'red'})
plt.title("Residual Plot: Actual vs Predicted Exam Scores", fontsize=16)
plt.xlabel("Actual Exam Score", fontsize=14)
plt.ylabel("Residuals (Actual - Predicted)", fontsize=14)
plt.axhline(0, color='gray', linestyle='--', linewidth=1)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.show()
