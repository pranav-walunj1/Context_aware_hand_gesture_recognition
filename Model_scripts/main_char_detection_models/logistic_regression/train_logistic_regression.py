import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.preprocessing import PolynomialFeatures

# Accuracy
# Step 1: Load your CSV data
df = pd.read_csv("D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_new_dataset_creation/body_keypoints_output_labelled_combined_upperbody_wo_conf.csv")

# Step 2: Split into features and labels
X = df.drop(columns=['Label']).values
y = df['Label'].values

# Step 3: Encode labels to integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Step 4: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 4.5: Add polynomial features (degree=2 for quadratic, try degree=3 if needed)
poly = PolynomialFeatures(degree=2, include_bias=False)
X_poly = poly.fit_transform(X_scaled)

# Step 5: Train-test split
#X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X_poly, y_encoded, test_size=0.2, random_state=42)
# Step 6: Optional - handle class imbalance using sample weights
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Step 7: Initialize and train Logistic Regression model
clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', class_weight='balanced', max_iter=2000, random_state=42)
#clf = LogisticRegression(multi_class='multinomial', solver='saga', class_weight='balanced', max_iter=10000, random_state=42)
clf.fit(X_train, y_train, sample_weight=sample_weights)

# Check how many iterations were actually used
print("Number of iterations until convergence:", clf.n_iter_)

# Step 8: Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Save the poly transformer too
joblib.dump(poly, 'D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_det_models/logistic_regression/poly_transformer.pkl')
# Step 9: Save the model and label encoder
joblib.dump(clf, 'D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_det_models/logistic_regression/logistic_regression_model_poly.pkl')
joblib.dump(le, 'D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_det_models/logistic_regression/label_encoder_poly.pkl')
joblib.dump(scaler, 'D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_det_models/logistic_regression/scaler_poly.pkl')


# Accuracy
acc = accuracy_score(y_test, y_pred)
print("Accuracy:", acc)

# Macro-averaged metrics
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')

print("Precision (macro):", precision)
print("Recall (macro):", recall)
print("F1-score (macro):", f1)

# Confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:\n", cm)