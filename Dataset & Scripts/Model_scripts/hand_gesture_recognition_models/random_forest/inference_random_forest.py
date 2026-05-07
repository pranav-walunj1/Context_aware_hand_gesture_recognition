import joblib
import mediapipe as mp
import cv2
import numpy as np

model = joblib.load(
    'D:/Pycharm Projects/Hand_pose_estimation/new_project/hand_gesture_recognition/classifier_trained_on_dataset_mp_labelled/random_forest/gesture_rf.pkl')  # or your filename

"""
# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False,
                       max_num_hands=1,
                       min_detection_confidence=0.5,
                       min_tracking_confidence=0.5)
mp_draw = mp.solutions.drawing_utils

# Start video capture
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Convert image to RGB
    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Process with MediaPipe
    results = hands.process(image_rgb)

    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Extract x, y, and dummy confidence (set to 1.0, or you can estimate)
            keypoints = []
            for lm in hand_landmarks.landmark:
                x = lm.x
                y = lm.y
                c = 0.5  # Confidence (since MediaPipe doesn't provide per-point confidence)
                keypoints.extend([x, y, c])

            # Convert to numpy array and reshape
            input_vector = np.array(keypoints).reshape(1, -1)

            # Predict gesture
            output_vector = model.predict(input_vector)
            print('output vetor is :', output_vector)
            gesture_label = output_vector[0]

            # Draw label
            cv2.putText(frame, f"Gesture: {gesture_label}", (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                        1, (0, 255, 0), 2, cv2.LINE_AA)

            # Optional: draw hand landmarks
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

    # Display
    cv2.imshow("Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release
cap.release()
cv2.destroyAllWindows()

"""

########################################################################################################################
# Initialize Mediapipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

# Open webcam
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
            # Extract keypoints as [x1, y1, x2, y2, ..., x21, y21]
            keypoints_2d = []
            for lm in hand_landmarks.landmark:
                x = lm.x * image_width
                y = lm.y * image_height
                z = lm.z
                keypoints_2d.extend([x, y, z])
            
            # Convert to np.array and reshape if needed
            keypoints_2d = np.array(keypoints_2d).reshape(1, -1)  # shape (1, 63)
            
            # Predict gesture
            prediction = model.predict(keypoints_2d)[0]
            # print("Predicted gesture:", prediction)
            
            probs = model.predict_proba(keypoints_2d)[0]  # e.g., [0.8, 0.1, 0.1]
            predicted_class = model.classes_[np.argmax(probs)]
            confidence = np.max(probs)
            
            print(f"Predicted: {predicted_class}, Confidence: {confidence:.2f}")
            
            if confidence >= 0.90:
                # Draw landmarks and prediction
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                cv2.putText(frame, prediction, (10, 40), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
    
    cv2.imshow("Hand Gesture Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()