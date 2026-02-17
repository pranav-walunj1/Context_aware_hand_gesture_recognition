import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
import joblib
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

# === Custom function to normalize relative to wrist in 3D ===
def normalize_hand_keypoints_3d(row):
    keypoints = row.values.reshape(-1, 3)  # shape (21, 3)
    wrist = keypoints[0]
    keypoints = keypoints - wrist  # translation-invariant
    max_dist = np.max(np.linalg.norm(keypoints, axis=1))  # scale-invariant
    if max_dist > 0:
        keypoints = keypoints / max_dist
    return keypoints.flatten()

# Step 1: Load your CSV data (must include x, y, z columns for each landmark)
df = pd.read_csv(
    "D:/Pycharm Projects/Hand_pose_estimation/new_project/hand_gesture_recognition/hand_gesture_data_labelled_from_mediapipe/normalized_keypoints_more_unknown_poses.csv"
)

# Step 2: Apply custom normalization to features (not the label)
X_raw = df.drop(columns=['Label'])
X_processed = np.array([normalize_hand_keypoints_3d(row) for _, row in X_raw.iterrows()])
y = df['Label'].values

# Step 3: Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Step 4: Standard scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X_processed)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
)

# Step 6: Handle class imbalance with sample weights
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Step 7: Train Logistic Regression model
clf = LogisticRegression(
    multi_class='multinomial',
    solver='lbfgs',
    class_weight='balanced',
    max_iter=500,
    random_state=42
)
clf.fit(X_train, y_train, sample_weight=sample_weights)

# Step 8: Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

acc = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='macro')
recall = recall_score(y_test, y_pred, average='macro')
f1 = f1_score(y_test, y_pred, average='macro')
cm = confusion_matrix(y_test, y_pred)

print("Accuracy:", acc)
print("Precision (macro):", precision)
print("Recall (macro):", recall)
print("F1-score (macro):", f1)
print("Confusion Matrix:\n", cm)
# Step 9: Save model, label encoder, and scaler
joblib.dump(clf, "D:/Pycharm Projects/Hand_pose_estimation/new_project/"
                 "hand_gesture_recognition/classifier_trained_on_dataset_mp_labelled/"
                 "logistic_regression/logistic_regression_model_relative_scale_norm_3d_more_unknown_poses.pkl")
joblib.dump(le, "D:/Pycharm Projects/Hand_pose_estimation/new_project/"
                "hand_gesture_recognition/classifier_trained_on_dataset_mp_labelled/"
                "logistic_regression/label_encoder_relative_scale_norm_3d_more_unknown_poses.pkl")
joblib.dump(scaler, "D:/Pycharm Projects/Hand_pose_estimation/new_project/"
                    "hand_gesture_recognition/classifier_trained_on_dataset_mp_labelled/"
                    "logistic_regression/scaler_relative_scale_norm_3d_more_unknown_poses.pkl")
