import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.utils.class_weight import compute_sample_weight
import joblib

# Step 1: Load your CSV data
#df = pd.read_csv("D:/Pycharm Projects/Hand_pose_estimation/new_project/hand_gesture_recognition/hand_gesture_data_labelled_from_mediapipe/overall_hand_poses_without_frame_id.csv")
df = pd.read_csv("D:/Pycharm Projects/Hand_pose_estimation/new_project/hand_gesture_recognition/hand_gesture_data_labelled_from_mediapipe/normalized_keypoints.csv")


# Step 2: Split into features and labels
X = df.drop(columns=['Label']).values
y = df['Label'].values

# Step 3: Encode labels to integers
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Step 4: Scale features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Step 5: Train-test split
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y_encoded, test_size=0.2, random_state=42)

# Step 6: Optional - handle class imbalance using sample weights
sample_weights = compute_sample_weight(class_weight='balanced', y=y_train)

# Step 7: Initialize and train Logistic Regression model
clf = LogisticRegression(multi_class='multinomial', solver='lbfgs', class_weight='balanced', max_iter=500, random_state=42)
clf.fit(X_train, y_train, sample_weight=sample_weights)

# Step 8: Evaluate
y_pred = clf.predict(X_test)
print(classification_report(y_test, y_pred, target_names=le.classes_))

# Step 9: Save the model and label encoder
#joblib.dump(clf, 'D:/Pycharm Projects/Hand_pose_estimation/new_project/hand_gesture_recognition/classifier_trained_on_dataset_mp_labelled/logistic_regression/logistic_regression_model.pkl')
#joblib.dump(le, 'D:/Pycharm Projects/Hand_pose_estimation/new_project/hand_gesture_recognition/classifier_trained_on_dataset_mp_labelled/logistic_regression/label_encoder.pkl')
#joblib.dump(scaler, 'D:/Pycharm Projects/Hand_pose_estimation/new_project/hand_gesture_recognition/classifier_trained_on_dataset_mp_labelled/logistic_regression/scaler.pkl')


# Step 9: Save the model and label encoder
joblib.dump(clf, 'D:/Pycharm Projects/Hand_pose_estimation/new_project/hand_gesture_recognition/classifier_trained_on_dataset_mp_labelled/logistic_regression/logistic_regression_model_normalzied_points.pkl')
joblib.dump(le, 'D:/Pycharm Projects/Hand_pose_estimation/new_project/hand_gesture_recognition/classifier_trained_on_dataset_mp_labelled/logistic_regression/label_encoder_normalized_points.pkl')
joblib.dump(scaler, 'D:/Pycharm Projects/Hand_pose_estimation/new_project/hand_gesture_recognition/classifier_trained_on_dataset_mp_labelled/logistic_regression/scaler_normalized_points.pkl')
