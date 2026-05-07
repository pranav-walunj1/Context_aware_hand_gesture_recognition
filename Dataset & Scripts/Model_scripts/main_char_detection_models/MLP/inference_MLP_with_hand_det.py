import pyzed.sl as sl
import cv2
import numpy as np
import joblib
import os
import mediapipe as mp
import time
import math
import torch

# === Load your trained model ===
#model_path = "D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_det_models/random_forest/gesture_rf.pkl"  # or model_logreg.pkl

#model = joblib.load(model_path)

# for model trained on normalized points wrt to wrist point as origin
# Load trained models
clf = joblib.load(
    'D:/Pycharm Projects/Hand_pose_estimation/new_project/hand_gesture_recognition/classifier_trained_on_dataset_mp_labelled/logistic_regression/logistic_regression_model_relative_scale_norm_3d.pkl')
le = joblib.load(
    'D:/Pycharm Projects/Hand_pose_estimation/new_project/hand_gesture_recognition/classifier_trained_on_dataset_mp_labelled/logistic_regression/label_encoder_relative_scale_norm_3d.pkl')
scaler = joblib.load(
    'D:/Pycharm Projects/Hand_pose_estimation/new_project/hand_gesture_recognition/classifier_trained_on_dataset_mp_labelled/logistic_regression/scaler_relative_scale_norm_3d.pkl')
# === Load your trained PyTorch model + LabelEncoder ===
label_encoder = joblib.load("D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_det_models/MLP/label_encoder_30_E.pkl")

class KeypointMLP(torch.nn.Module):
    def __init__(self, input_dim=36, hidden_dim=128, num_classes=3):
        super(KeypointMLP, self).__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, num_classes)  # logits (no softmax)
        )

    def forward(self, x):
        return self.net(x)

# Recreate model with correct output size
model = KeypointMLP(input_dim=36, hidden_dim=128, num_classes=len(label_encoder.classes_))
model.load_state_dict(torch.load("D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_det_models/MLP/mlp_model_30_E.pth"))
model.eval()
# === Parameters ===
confidence_threshold = 0.4

# === Before while loop ===
active_counter = {}  # Dictionary to track each object_id
active_start_time = {}  # At top of file or before loop
activation_duration = 3  # seconds
elapsed_time = 0
activation_threshold = 75  # ~3 seconds at 30 FPS
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, min_detection_confidence=0.5)
mp_drawing = mp.solutions.drawing_utils

import numpy as np

def select_main_hand(obj, mediapipe_hands,left_wrist_2d,right_wrist_2d, frame_w, frame_h):
    # Step 1: pick best ZED wrist
    left_wrist = obj.keypoint[7]     # (x,y,z) 2D or 3D depending on ZED config
    right_wrist = obj.keypoint[4]
    left_conf = obj.keypoint_confidence[7]
    right_conf = obj.keypoint_confidence[4]
    hand_conf_thres = 40
    # Choose based on confidence
    #if abs(left_conf - right_conf) > 0.2:
    #    chosen_wrist = left_wrist if left_conf > right_conf else right_wrist
    #    chosen_wrist_2d = left_wrist_2d if left_conf > right_conf else right_wrist_2d
    #else:
    #    # fallback: higher wrist in image space (smaller y)
    #    chosen_wrist = left_wrist if left_wrist[1] < right_wrist[1] else right_wrist
    #    chosen_wrist_2d = left_wrist_2d if left_wrist_2d[1] < right_wrist_2d[1] else right_wrist_2d
    
    print('++++++++++ confidence of left hand :', left_conf)
    print('++++++++++ confidence of right hand :', right_conf)
    # Check if confidences are valid numbers and above threshold
    left_valid = not math.isnan(left_conf) and left_conf >= hand_conf_thres
    right_valid = not math.isnan(right_conf) and right_conf >= hand_conf_thres

    if left_valid and not right_valid:
        chosen_wrist_2d = left_wrist_2d
    elif right_valid and not left_valid:
        chosen_wrist_2d = right_wrist_2d
    elif left_valid and right_valid:
        # both valid, choose higher wrist (smaller y) as fallback
        chosen_wrist_2d = left_wrist_2d if left_wrist_2d[1] < right_wrist_2d[1] else right_wrist_2d
    else:
        # both invalid or below threshold
        return None
    #chosen_wrist_2d = left_wrist_2d if left_wrist_2d[1] < right_wrist_2d[1] else right_wrist_2d
    # Step 2: match with mediapipe hands
    chosen_hand = None
    min_dist = float("inf")

    for hand in mediapipe_hands:
        mp_wrist = hand.landmark[0]  # MediaPipe wrist
        # Convert normalized coords → pixel space
        mp_x, mp_y = int(mp_wrist.x * frame_w), int(mp_wrist.y * frame_h)

        dist = np.linalg.norm([
            #mp_x - chosen_wrist[0],
            #mp_y - chosen_wrist[1]
            mp_x - chosen_wrist_2d[0],
            mp_y - chosen_wrist_2d[1]
        ])

        if dist < min_dist:
            min_dist = dist
            chosen_hand = hand

    return chosen_hand


def extract_keypoints(obj):
    keypoints = obj.keypoint
    confidences = obj.keypoint_confidence
    features = []
    
    # Only include keypoints 0–7 and 14–17
    selected_indices = list(range(8)) + list(range(14, 18))
    
    for i in selected_indices:
        x, y, z = keypoints[i]
        # c = confidences[i]
        features.extend([x, y, z])
    
    return features


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
    return padded, (left, top)


def project_point(point3D, intrinsics, image_scale):
    x, y, z = point3D
    if z == 0 or np.isnan(z):
        return None
    u = int((intrinsics.fx * x / z) + intrinsics.cx)
    v = int((intrinsics.fy * y / z) + intrinsics.cy)
    u = int(u * image_scale[0])  # scale x-coordinate
    v = int(v * image_scale[1])  # scale y-coordinate
    return (u, v)


def map_wrist_to_crop(wrist_2d, bbox_coords, target_size=(480, 640), padding=(0, 0)):
    """
        Maps wrist coordinates from original frame to the resized crop passed to MediaPipe.

        wrist_2d: (x, y) in original frame
        bbox_coords: (x1, y1, x2, y2) bbox of person in original frame
        target_size: (height, width) of resized rgb_crop
        """
    x1, y1, x2, y2 = bbox_coords
    crop_h, crop_w = target_size
    pad_x, pad_y = padding

    orig_w = x2 - x1
    orig_h = y2 - y1

    # Shift to crop coords
    x_shifted = wrist_2d[0] - x1
    y_shifted = wrist_2d[1] - y1

    # Scale to resized crop size
    x_scaled = x_shifted * (crop_w / orig_w) + pad_x
    y_scaled = y_shifted * (crop_h / orig_h) + pad_y

    return int(x_scaled), int(y_scaled)

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
            
            # === Convert to torch tensor ===
            sample_tensor = torch.tensor(features_np, dtype=torch.float32)
            try:
                with torch.no_grad():
                    logits = model(sample_tensor)
                    pred_idx = torch.argmax(logits, dim=1).item()
                    prediction = label_encoder.inverse_transform([pred_idx])[0]
            except Exception as e:
                prediction = "Error"
                print('Exception is:', e)
                
            # Draw skeleton keypoints
            for i, (x, y, z) in enumerate(obj.keypoint):
                if obj.keypoint_confidence[i] > confidence_threshold:
                    keypoint = (x, y, z)
                    coords_2d = project_point(keypoint, intrinsics, image_scale)
                    cv2.circle(frame_bgr, (int(coords_2d[0]), int(coords_2d[1])), 4, (0, 255, 0), -1)
            
            
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
                if x2 > x1 and y2 > y1:
                    cv2.rectangle(frame_bgr, (x1, y1), (x2, y2), (0, 255, 0), 2)  # color BGR, thickness=2

                left_wrist_3d = obj.keypoint[7]  # left wrist point from zed camera
                left_wrist_3d_conf = obj.keypoint_confidence[7]
                right_wrist_3d = obj.keypoint[4]  # right wrist point from zed camera
                right_wrist_3d_conf = obj.keypoint_confidence[4]
                # projecting wrist points into 2D
                left_wrist_2d = project_point(left_wrist_3d, intrinsics, image_scale)
                right_wrist_2d = project_point(right_wrist_3d, intrinsics, image_scale)
                
                
                # person_crop = frame_bgr
                # Run MediaPipe Hands on cropped image
                rgb_crop = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
                # Resize to (width=640, height=480)
                # rgb_crop = cv2.resize(rgb_crop, (640, 480))
                rgb_crop, padding = resize_with_padding(rgb_crop, (480, 640))
                results = hands.process(rgb_crop)
                left_wrist_scaled = map_wrist_to_crop(left_wrist_2d, (x1, y1, x2, y2), target_size=(480, 640), padding= padding)
                right_wrist_scaled = map_wrist_to_crop(right_wrist_2d, (x1, y1, x2, y2), target_size=(480, 640), padding= padding)
                chosen_hand = None
                if results.multi_hand_landmarks:
                    
                    
                    # Choose just one hand (instead of looping over all)
                    chosen_hand = select_main_hand(
                        obj,
                        results.multi_hand_landmarks,  # already available
                        #left_wrist_2d,
                        #right_wrist_2d,
                        left_wrist_scaled,
                        right_wrist_scaled,
                        rgb_crop.shape[1],  # frame width
                        rgb_crop.shape[0]  # frame height
                    )
    
                    if chosen_hand is not None:
                        # Draw landmarks for chosen hand only
                        mp_drawing.draw_landmarks(person_crop, chosen_hand, mp_hands.HAND_CONNECTIONS)
        
                        # === Extract 21 keypoints (x, y, z) from MediaPipe ===
                        hand_keypoints = []
                        # h, w, _ = person_crop.shape
                        for lm in chosen_hand.landmark:
                            # normalized coords (0-1), can convert to pixel if needed
                            x_pixel = lm.x
                            y_pixel = lm.y
                            z_pixel = lm.z
                            hand_keypoints.extend([x_pixel, y_pixel, z_pixel])
        
                        import pandas as pd
                        hand_keypoints_series = pd.Series(hand_keypoints)
        
                        # Normalize relative to wrist in 3D
                        normalized_keypoints = normalize_hand_keypoints_3d(hand_keypoints_series).reshape(1, -1)
        
                        # Scale
                        keypoints_scaled = scaler.transform(normalized_keypoints)
        
                        try:
                            pred = clf.predict(keypoints_scaled)[0]
                            gesture = le.inverse_transform([pred])[0]
            
                            probs = clf.predict_proba(keypoints_scaled)[0]
                            confidence = np.max(probs)
            
                            print(f"Predicted: {gesture}, Confidence: {confidence:.2f}")
            
                            # Only display if confident
                            if confidence >= 0.80:
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
