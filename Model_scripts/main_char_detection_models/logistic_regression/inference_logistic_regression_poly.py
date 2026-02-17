import cv2
import mediapipe as mp
import numpy as np
import joblib

# Load trained models
#clf = joblib.load('D:/Pycharm Projects/Hand_pose_estimation/new_project/hand_gesture_recognition/classifier_trained_on_dataset_mp_labelled/logistic_regression/logistic_regression_model.pkl')
import pyzed.sl as sl
import cv2
import numpy as np
import joblib
import os
import time

# === Load your trained model ===
model_path = 'D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_det_models/logistic_regression/logistic_regression_model_poly.pkl'  # or model_logreg.pkl
model = joblib.load(model_path)
le_main = joblib.load('D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_det_models/logistic_regression/label_encoder_poly.pkl')
scaler_main = joblib.load('D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_det_models/logistic_regression/scaler_poly.pkl')
poly = joblib.load('D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_det_models/logistic_regression/poly_transformer.pkl')
# === Parameters ===
confidence_threshold = 0.4

# for hand posture recognition
# Load trained models
clf = joblib.load('D:/Pycharm Projects/Hand_pose_estimation/new_project/hand_gesture_recognition/classifier_trained_on_dataset_mp_labelled/logistic_regression/logistic_regression_model_relative_scale_norm_3d.pkl')
le = joblib.load('D:/Pycharm Projects/Hand_pose_estimation/new_project/hand_gesture_recognition/classifier_trained_on_dataset_mp_labelled/logistic_regression/label_encoder_relative_scale_norm_3d.pkl')
scaler = joblib.load('D:/Pycharm Projects/Hand_pose_estimation/new_project/hand_gesture_recognition/classifier_trained_on_dataset_mp_labelled/logistic_regression/scaler_relative_scale_norm_3d.pkl')


# === Before while loop ===
active_counter = {}  # Dictionary to track each object_id
active_start_time = {}  # At top of file or before loop
activation_duration = 3  # seconds
elapsed_time = 0
activation_threshold = 75  # ~3 seconds at 30 FPS
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils
def extract_keypoints(obj):
    keypoints = obj.keypoint
    confidences = obj.keypoint_confidence
    features = []

    # Only include keypoints 0–7 and 14–17
    selected_indices = list(range(8)) + list(range(14, 18))

    for i in selected_indices:
        x, y, z = keypoints[i]
        #c = confidences[i]
        features.extend([x, y, z])

    return features

def project_point(point3D, intrinsics, image_scale):
    x, y, z = point3D
    if z == 0 or np.isnan(z):
        return None
    u = int((intrinsics.fx * x / z) + intrinsics.cx)
    v = int((intrinsics.fy * y / z) + intrinsics.cy)
    u = int(u * image_scale[0])  # scale x-coordinate
    v = int(v * image_scale[1])  # scale y-coordinate
    return (u, v)


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


def resize_with_padding(image, target_size=(480, 640)):
    target_h, target_w = target_size
    h, w = image.shape[:2]
    scale = min(target_w / w, target_h / h)
    
    new_w = int(w * scale)
    new_h = int(h * scale)
    resized = cv2.resize(image, (new_w, new_h))
    
    # Create padded image
    pad_w = target_w - new_w
    pad_h = target_h - new_h
    top, bottom = pad_h // 2, pad_h - pad_h // 2
    left, right = pad_w // 2, pad_w - pad_w // 2
    
    padded = cv2.copyMakeBorder(resized, top, bottom, left, right, cv2.BORDER_CONSTANT, value=(0, 0, 0))
    return padded


def main():
    # === ZED initialization ===
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.coordinate_units = sl.UNIT.MILLIMETER
    init_params.camera_fps = 30

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera.")
        exit(1)

    tracking_params = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(tracking_params)

    obj_param = sl.ObjectDetectionParameters()
    obj_param.enable_tracking = True
    obj_param.enable_body_fitting = True
    obj_param.detection_model = sl.DETECTION_MODEL.HUMAN_BODY_FAST
    obj_param.body_format = sl.BODY_FORMAT.POSE_18
    zed.enable_object_detection(obj_param)

    runtime_params = sl.RuntimeParameters()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 40

    # Get ZED camera information
    camera_info = zed.get_camera_information()
    intrinsics = camera_info.camera_configuration.calibration_parameters.left_cam
    # 2D viewer utilities
    display_resolution = sl.Resolution(min(camera_info.camera_resolution.width, 1280),
                                       min(camera_info.camera_resolution.height, 720))
    image_scale = [display_resolution.width / camera_info.camera_resolution.width
        , display_resolution.height / camera_info.camera_resolution.height]

    image = sl.Mat()
    objects = sl.Objects()
    prev_time = 0
    print("Starting real-time inference...")
    while zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        zed.retrieve_objects(objects, obj_runtime_param)

        frame = image.get_data()
        frame_bgr = frame[:, :, :3].copy()

        for obj in objects.object_list:
            if obj.confidence < confidence_threshold:
                continue

            features = extract_keypoints(obj)
            features_np = np.array(features).reshape(1, -1)

            try:
                #prediction = model.predict(features_np)[0]
                ###########################################################
                #features_np = np.array(keypoints_2d).reshape(1, -1)
                keypoints_scaled = scaler_main.transform(features_np)

                keypoints_poly = poly.fit_transform(keypoints_scaled)  # <-- expand polynomial features

                pred = model.predict(keypoints_poly)[0]
                prediction = le_main.inverse_transform([pred])[0]
                
                probs = model.predict_proba(keypoints_poly)[0]  # e.g., [0.8, 0.1, 0.1]
                predicted_class = model.classes_[np.argmax(probs)]
                confidence = np.max(probs)
                
                ###########################################################
            except Exception as e:
                prediction = "Error"
                print('Exception is :', e)

            # Draw skeleton keypoints
            for i, (x, y, z) in enumerate(obj.keypoint):
                if obj.keypoint_confidence[i] > confidence_threshold:
                    keypoint = (x, y, z)
                    coords_2d = project_point(keypoint, intrinsics, image_scale)
                    cv2.circle(frame_bgr, (int(coords_2d[0]), int(coords_2d[1])), 4, (0, 255, 0), -1)

            # Show predicted label near head (keypoint 0)
            #if obj.keypoint_confidence[0] > confidence_threshold:
            #    head_x, head_y, _ = obj.keypoint[0]
            #    head_point = project_point(obj.keypoint[0], intrinsics, image_scale)
            #    cv2.putText(frame_bgr, f"{prediction}", (int(head_point[0]), int(head_point[1] - 20)),
            #                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                    # === Draw label above the skeleton ===
            visible_points = [
                project_point(obj.keypoint[i], intrinsics, image_scale)
                for i in range(len(obj.keypoint))
                if obj.keypoint_confidence[i] > confidence_threshold
            ]

            if visible_points:
                xs, ys = zip(*[(pt[0], pt[1]) for pt in visible_points])
                min_x, min_y = min(xs), min(ys)
                max_x, max_y = max(xs), max(ys)
    
                label_x = int((min_x + max_x) / 2)
                label_y = int(min_y - 20)
                cv2.putText(frame_bgr, f"{prediction}", (label_x, label_y),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

            # === Track "Active_main_Char" status ===
            obj_id = obj.id
            # if prediction == "Active_main_char":
            #    active_counter[obj_id] = active_counter.get(obj_id, 0) + 1
            # else:
            #    active_counter[obj_id] = 0
            curr_time1 = time.time()

            if prediction == "Active_main_char":
                if obj_id not in active_start_time:
                    active_start_time[obj_id] = curr_time1  # Start tracking time
                elapsed_time = curr_time1 - active_start_time[obj_id]
            else:
                active_start_time.pop(obj_id, None)
                elapsed_time = 0
            frame_width = frame_bgr.shape[1]
            frame_height = frame_bgr.shape[0]
            # === Trigger MediaPipe after 3 seconds ===
            # if active_counter.get(obj_id, 0) >= activation_threshold:
            # print('elapsed time is :', elapsed_time)
            if elapsed_time >= activation_duration:
                # Add padding (e.g., 15%)
                pad_ratio = 0.05
                x1 = int(max(0, min_x - frame_width * pad_ratio))
                y1 = int(max(0, min_y - frame_height * pad_ratio))
                x2 = int(min(frame_bgr.shape[1], max_x + frame_width * pad_ratio))
                y2 = int(min(frame_bgr.shape[0], max_y + frame_height * pad_ratio))
    
                person_crop = frame_bgr[y1:y2, x1:x2]  # Only draw if box has positive size
                # if x2 > x1 and y2 > y1:
                #    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)  # color BGR, thickness=2

                # person_crop = frame_bgr
                # Run MediaPipe Hands on cropped image
                rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                # Resize to (width=640, height=480)
                # rgb_crop = cv2.resize(rgb_crop, (640, 480))
                rgb_crop = resize_with_padding(rgb_crop, (480, 640))
                results = hands.process(rgb_crop)
    
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
            
                        # mp_drawing.draw_landmarks(person_crop, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                        mp_drawing.draw_landmarks(rgb_crop, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
                        # === Extract 21 keypoints (x, y, z) from MediaPipe ===
                        hand_keypoints = []
                        h, w, _ = person_crop.shape
                        for lm in hand_landmarks.landmark:
                            # x_pixel = lm.x * w
                            # y_pixel = lm.y * h
                            x_pixel = lm.x
                            y_pixel = lm.y
                            z_pixel = lm.z  # z is often scaled similarly to x
                            # Transform back to full-frame coordinate space
                            # x_pixel = x_pixel + x1
                            # y_pixel = y_pixel + y1
                
                            hand_keypoints.extend([x_pixel, y_pixel, z_pixel])
            
                        # Convert to pandas Series (so .values works in your function)
                        import pandas as pd
                        hand_keypoints_series = pd.Series(hand_keypoints)
            
                        # Normalize relative to wrist in 3D
                        normalized_keypoints = normalize_hand_keypoints_3d(hand_keypoints_series).reshape(1, -1)
            
                        # Scale
                        keypoints_scaled = scaler.transform(normalized_keypoints)
                        # === Convert to NumPy and reshape ===
                        # hand_keypoints_np = np.array(hand_keypoints).reshape(1, -1)
                        # keypoints_scaled = scaler.transform(hand_keypoints_np)
                        try:
                
                            pred = clf.predict(keypoints_scaled)[0]
                            gesture = le.inverse_transform([pred])[0]
                            ##################################################################################
                            probs = clf.predict_proba(keypoints_scaled)[0]  # e.g., [0.8, 0.1, 0.1]
                            predicted_class = clf.classes_[np.argmax(probs)]
                            confidence = np.max(probs)
                
                            print(f"Predicted: {gesture}, Confidence: {confidence:.2f}")
                            # Show on cropped person image
                            if confidence >= 0.95:
                                cv2.putText(person_crop, f"{gesture}", (10, 30),
                                            cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 0, 0), 2)
            
                        except Exception as e:
                            print("Error in hand gesture prediction:", e)
    
                # Optional: reset counter to avoid repeated detection
                # active_counter[obj_id] = 0
            # === Show FPS on screen ===
        # === FPS Calculation Started ===
        curr_time = time.time()
        fps = 1 / (curr_time - prev_time) if prev_time != 0 else 0
        prev_time = curr_time
        # === FPS Calculation Ended ===
        cv2.putText(frame_bgr, f'FPS: {int(fps)}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                            1, (0, 255, 0), 2)
        cv2.imshow("Live Pose Classification", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # === Cleanup ===
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
