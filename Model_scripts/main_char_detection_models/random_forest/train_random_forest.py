import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib  # to save model
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# Accuracy
# Load your dataset

df = pd.read_csv("D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_new_dataset_creation/body_keypoints_output_labelled_combined_upperbody_wo_conf.csv")

# Separate features and labels
X = df.drop(columns=['Label']).values
y = df['Label'].values

# Split into train/test
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

# Train model
clf = RandomForestClassifier(n_estimators=100, random_state=42, class_weight='balanced')
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(clf, 'D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_det_models/random_forest/gesture_rf.pkl')



# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Macro-averaged Precision, Recall, and F1-score
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("Precision (macro):", precision)
print("Recall (macro):", recall)
print("F1-score (macro):", f1)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)