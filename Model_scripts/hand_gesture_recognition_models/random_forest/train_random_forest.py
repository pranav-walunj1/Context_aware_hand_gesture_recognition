import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import joblib  # to save model

# Load your dataset

#df = pd.read_csv("D:/Pycharm Projects/Hand_pose_estimation/new_project/hand_gesture_recognition/hand_gesture_data_labelled_from_mediapipe/overall_hand_poses_without_frame_id.csv")
df = pd.read_csv("D:/Pycharm Projects/Hand_pose_estimation/new_project/hand_gesture_recognition/hand_gesture_data_labelled_from_mediapipe/normalized_keypoints_wo_z_coords.csv")


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
joblib.dump(clf, 'D:/Pycharm Projects/Hand_pose_estimation/new_project/hand_gesture_recognition/classifier_trained_on_dataset_mp_labelled/random_forest/gesture_rf_on_normalized_points_wo_z_coords.pkl')
