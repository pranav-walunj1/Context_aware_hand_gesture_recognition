import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load trained models
clf = joblib.load('D:/Pycharm Projects/Hand_pose_estimation/new_project/hand_gesture_recognition/classifier_trained_on_dataset_mp_labelled/logistic_regression/logistic_regression_model.pkl')
le = joblib.load('D:/Pycharm Projects/Hand_pose_estimation/new_project/hand_gesture_recognition/classifier_trained_on_dataset_mp_labelled/logistic_regression/label_encoder.pkl')
scaler = joblib.load('D:/Pycharm Projects/Hand_pose_estimation/new_project/hand_gesture_recognition/classifier_trained_on_dataset_mp_labelled/logistic_regression/scaler.pkl')

# Init Mediapipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(image)
    image_height, image_width = frame.shape[:2]

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            keypoints_2d = []
            for lm in hand_landmarks.landmark:
                x = lm.x * image_width
                y = lm.y * image_height
                z = lm.z
                keypoints_2d.extend([x, y, z])

            keypoints_2d = np.array(keypoints_2d).reshape(1, -1)
            keypoints_scaled = scaler.transform(keypoints_2d)

            pred = clf.predict(keypoints_scaled)[0]
            gesture = le.inverse_transform([pred])[0]
            ##################################################################################
            probs = clf.predict_proba(keypoints_scaled)[0]  # e.g., [0.8, 0.1, 0.1]
            predicted_class = clf.classes_[np.argmax(probs)]
            confidence = np.max(probs)

            print(f"Predicted: {gesture}, Confidence: {confidence:.2f}")
            #######################################################################################

            if confidence >= 0.90:
                cv2.putText(frame, f'{gesture}', (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
