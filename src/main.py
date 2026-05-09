import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, confusion_matrix

# -------------------------------
# Paths
# -------------------------------
base_dir = os.getcwd()

data_path = os.path.join(base_dir, 'data', 'StudentPerformanceFactors.csv')
graph_dir = os.path.join(base_dir, 'graphs')
output_path = os.path.join(base_dir, 'outputs', 'results.txt')

# -------------------------------
# Load data
# -------------------------------
df = pd.read_csv(data_path)

# -------------------------------
# Create target
# -------------------------------
threshold = df['Exam_Score'].median()
df['Final_Result'] = df['Exam_Score'].apply(lambda x: 1 if x >= threshold else 0)

# -------------------------------
# Features
# -------------------------------
features = ['Hours_Studied', 'Attendance', 'Previous_Scores']

X = df[features]
y = df['Final_Result']

# -------------------------------
# Split
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

# -------------------------------
# MODELS
# -------------------------------
lr = LogisticRegression()
dt = DecisionTreeClassifier()

lr.fit(X_train, y_train)
dt.fit(X_train, y_train)

# Predictions
lr_pred = lr.predict(X_test)
dt_pred = dt.predict(X_test)

# Accuracy
lr_acc = accuracy_score(y_test, lr_pred)
dt_acc = accuracy_score(y_test, dt_pred)

print("Logistic Regression Accuracy:", lr_acc)
print("Decision Tree Accuracy:", dt_acc)

# -------------------------------
# Save results
# -------------------------------
with open(output_path, 'w') as f:
    f.write(f"Logistic Regression Accuracy: {lr_acc}\n")
    f.write(f"Decision Tree Accuracy: {dt_acc}\n")

# -------------------------------
# 1. Confusion Matrix
# -------------------------------
cm = confusion_matrix(y_test, lr_pred)

plt.figure(figsize=(5,4))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title("Confusion Matrix")
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.savefig(os.path.join(graph_dir, 'confusion_matrix.png'))
plt.close()

# -------------------------------
# 2. Correlation Heatmap
# -------------------------------
plt.figure(figsize=(8,6))
corr = df[features + ['Final_Result']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.savefig(os.path.join(graph_dir, 'correlation_heatmap.png'))
plt.close()

# -------------------------------
# 3. Feature Importance (Decision Tree)
# -------------------------------
importance = dt.feature_importances_

plt.figure(figsize=(10,6))
plt.barh(features, importance)
plt.title("Feature Importance")
plt.xlabel("Importance Score")
plt.savefig(os.path.join(graph_dir, 'feature_importance.png'))
plt.close()

# -------------------------------
# 4. Model Comparison
# -------------------------------
models = ['Decision Tree', 'Logistic Regression']
accuracy = [dt_acc, lr_acc]

plt.figure()
plt.bar(models, accuracy)
plt.title("Model Comparison")
plt.ylabel("Accuracy")

# Add values on top
for i, v in enumerate(accuracy):
    plt.text(i, v + 0.01, f"{v:.2f}", ha='center')

plt.savefig(os.path.join(graph_dir, 'model_comparison.png'))
plt.close()

# -------------------------------
# GUI Prediction
# -------------------------------

import tkinter as tk
from tkinter import messagebox

def predict_result():

    study_hours = float(entry_hours.get())
    attendance = float(entry_attendance.get())
    previous_scores = float(entry_scores.get())

    new_student = [[study_hours, attendance, previous_scores]]

    prediction = lr.predict(new_student)

    if prediction[0] == 1:
        messagebox.showinfo("Prediction", "Student will PASS")
    else:
        messagebox.showerror("Prediction", "Student will FAIL")

# -------------------------------
# GUI Window
# -------------------------------

root = tk.Tk()
root.title("Student Performance Prediction")
root.geometry("400x300")

# Study Hours
tk.Label(root, text="Study Hours").pack(pady=5)
entry_hours = tk.Entry(root)
entry_hours.pack()

# Attendance
tk.Label(root, text="Attendance").pack(pady=5)
entry_attendance = tk.Entry(root)
entry_attendance.pack()

# Previous Scores
tk.Label(root, text="Previous Scores").pack(pady=5)
entry_scores = tk.Entry(root)
entry_scores.pack()

# Predict Button
tk.Button(
    root,
    text="Predict Result",
    command=predict_result,
    bg="blue",
    fg="white"
).pack(pady=20)

root.mainloop()
