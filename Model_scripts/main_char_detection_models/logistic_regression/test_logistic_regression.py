import pandas as pd
import numpy as np
import joblib
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import PolynomialFeatures, StandardScaler, LabelEncoder

# ------------------------------
# 1. Load test dataset
# ------------------------------
test_csv = "D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_new_dataset_creation/Test_dataset/labelled_dataset/main_char_dataset_cleaned_upperbody_wo_conf.csv"
df_test = pd.read_csv(test_csv)

X_test = df_test.drop(columns=['Label', 'image']).values
y_test = df_test['Label'].values

# ------------------------------
# 2. Load trained model, label encoder, scaler, polynomial transformer
# ------------------------------
model_path = 'D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_det_models/logistic_regression/logistic_regression_model_poly.pkl'
le_path = 'D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_det_models/logistic_regression/label_encoder_poly.pkl'
scaler_path = 'D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_det_models/logistic_regression/scaler_poly.pkl'
poly_path = 'D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_det_models/logistic_regression/poly_transformer.pkl'

clf = joblib.load(model_path)
le = joblib.load(le_path)
scaler = joblib.load(scaler_path)
poly = joblib.load(poly_path)

# ------------------------------
# 3. Preprocess test features
# ------------------------------
X_test_scaled = scaler.transform(X_test)
X_test_poly = poly.transform(X_test_scaled)

# Encode labels
y_test_encoded = le.transform(y_test)

# ------------------------------
# 4. Predict
# ------------------------------
y_pred = clf.predict(X_test_poly)

# ------------------------------
# 5. Handle missing classes in test set
# ------------------------------
from sklearn.utils.multiclass import unique_labels
present_classes = unique_labels(y_test_encoded, y_pred)

print("=== Classification Report ===")
print(classification_report(
    y_test_encoded,
    y_pred,
    labels=present_classes,
    target_names=le.classes_[present_classes],
    zero_division=0
))

# ------------------------------
# 6. Compute metrics
# ------------------------------
acc = accuracy_score(y_test_encoded, y_pred)
precision = precision_score(y_test_encoded, y_pred, average='macro', zero_division=0)
recall = recall_score(y_test_encoded, y_pred, average='macro', zero_division=0)
f1 = f1_score(y_test_encoded, y_pred, average='macro', zero_division=0)
cm = confusion_matrix(y_test_encoded, y_pred)

print("Accuracy:", acc)
print("Precision (macro):", precision)
print("Recall (macro):", recall)
print("F1-score (macro):", f1)
print("Confusion Matrix:\n", cm)
