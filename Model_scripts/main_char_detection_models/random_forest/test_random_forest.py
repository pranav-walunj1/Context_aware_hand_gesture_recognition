import pandas as pd
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix

# Paths
model_path = "D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_det_models/random_forest/gesture_rf.pkl"
test_csv = "D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_new_dataset_creation/Test_dataset/labelled_dataset/main_char_dataset_cleaned_upperbody_wo_conf.csv"

# Load trained model
clf = joblib.load(model_path)

# Load test dataset
df_test = pd.read_csv(test_csv)

# Separate features and labels
X_test = df_test.drop(columns=['Label','image']).values
y_test = df_test['Label'].values

# Predict with trained model
y_pred = clf.predict(X_test)

# Evaluation
print("=== Classification Report ===")
print(classification_report(y_test, y_pred))

acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("\n=== Metrics on Test Dataset ===")
print(f"Accuracy: {acc:.4f}")
print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro): {recall:.4f}")
print(f"F1-score (macro): {f1:.4f}")

print("\n=== Confusion Matrix ===")
print(confusion_matrix(y_test, y_pred))
