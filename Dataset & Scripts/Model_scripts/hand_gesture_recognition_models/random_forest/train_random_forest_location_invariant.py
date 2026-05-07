import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix


# === Function to normalize keypoints relative to wrist in 3D ===
def normalize_hand_keypoints_3d(row):
    # Convert row to (21, 3)
    keypoints = row.values.reshape(-1, 3)  # includes x, y, z
    
    # Step 1: translation — subtract wrist (landmark 0) coords
    wrist = keypoints[0]
    keypoints = keypoints - wrist
    
    # Step 2: scale normalization — divide by max 3D distance
    max_dist = np.max(np.linalg.norm(keypoints, axis=1))
    if max_dist > 0:
        keypoints = keypoints / max_dist
    
    return keypoints.flatten()


# === Load dataset ===
#df = pd.read_csv(
#    "D:/Pycharm Projects/Hand_pose_estimation/new_project/"
#    "hand_gesture_recognition/hand_gesture_data_labelled_from_mediapipe/"
#    "normalized_keypoints.csv"  # Now includes z
#)
df = pd.read_csv(
    "D:/Pycharm Projects/Hand_pose_estimation/new_project/"
    "hand_gesture_recognition/hand_gesture_data_labelled_from_mediapipe/"
    "normalized_keypoints_more_unknown_poses.csv"  # Now includes more unknown poses
)

# Separate features (excluding label)
X_raw = df.drop(columns=['Label'])

# Apply relative+scale normalization to every row
X_processed = np.array([normalize_hand_keypoints_3d(row) for _, row in X_raw.iterrows()])

# Labels
y = df['Label'].values

# Split
X_train, X_test, y_train, y_test = train_test_split(
    X_processed, y, test_size=0.2, stratify=y, random_state=42
)

# Train model
clf = RandomForestClassifier(
    n_estimators=100, random_state=42, class_weight='balanced'
)
clf.fit(X_train, y_train)

# Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(
    clf,
    "D:/Pycharm Projects/Hand_pose_estimation/new_project/"
    "hand_gesture_recognition/classifier_trained_on_dataset_mp_labelled/"
    "random_forest/gesture_rf_relative_scale_norm_3d.pkl"
)



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