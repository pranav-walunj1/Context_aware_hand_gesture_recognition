import cv2
import csv
import os
import numpy as np
import mediapipe as mp

# Setup MediaPipe Hands
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Setup output folder
output_folder = "gesture_saved_frames_mediapipe"
os.makedirs(output_folder, exist_ok=True)

# Setup CSV file
csv_filename = "keypoints_data_mediapipe.csv"
csv_file = open(csv_filename, mode='w', newline='')
csv_writer = csv.writer(csv_file)

# CSV header
header = ["frame_id", "hand_index"]
for i in range(21):
    #header += [f"kp{i}_x", f"kp{i}_y", f"kp{i}_z", f"kp{i}_conf"]
    header += [f"kp{i}_x", f"kp{i}_y", f"kp{i}_z"]
csv_writer.writerow(header)

# Webcam capture
cap = cv2.VideoCapture(0)

#cap = cv2.VideoCapture("path_to_your_video.mp4")  # replace with your .mp4 path

frame_id = 0

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Flip and convert for MediaPipe
    #frame_rgb = cv2.cvtColor(cv2.flip(frame, 1), cv2.COLOR_BGR2RGB)
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(frame_rgb)

    hands_found = False

    if results.multi_hand_landmarks:
        for hand_index, hand_landmarks in enumerate(results.multi_hand_landmarks):
            hands_found = True
            h, w, _ = frame.shape

            row = [frame_id, hand_index]
            for i, landmark in enumerate(hand_landmarks.landmark):
                x = landmark.x * w
                y = landmark.y * h
                z = landmark.z  # Depth is a relative measure
                print(frame.shape)
                #conf = 1.0  # MediaPipe doesn't give per-point confidence, assume 1.0
                #row.extend([x, y, z, conf])
                row.extend([x, y, z])
            # Draw keypoints and connections
            #frame = cv2.flip(frame, 1)  # Flip horizontally
            mp_drawing.draw_landmarks(
                frame,
                hand_landmarks,
                mp_hands.HAND_CONNECTIONS,
                mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2, circle_radius=3),
                mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2)
            )

            csv_writer.writerow(row)

    if hands_found:
        save_path = os.path.join(output_folder, f"{frame_id}.jpg")
        cv2.imwrite(save_path, frame)
        print(f"Saved frame {frame_id} with detected hands.")

    frame_id += 1

    cv2.imshow("MediaPipe Hand Keypoints", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Cleanup
cap.release()
cv2.destroyAllWindows()
csv_file.close()
hands.close()
