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

# === Load your trained model ===
model_path = "D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_det_models/logistic_regression/logistic_regression_model.pkl"  # or model_logreg.pkl
model = joblib.load(model_path)
le = joblib.load('D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_det_models/logistic_regression/label_encoder.pkl')
scaler = joblib.load('D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_det_models/logistic_regression/scaler.pkl')

# === Parameters ===
confidence_threshold = 0.4

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
                keypoints_scaled = scaler.transform(features_np)

                pred = model.predict(keypoints_scaled)[0]
                prediction = le.inverse_transform([pred])[0]
                
                probs = model.predict_proba(keypoints_scaled)[0]  # e.g., [0.8, 0.1, 0.1]
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
            if obj.keypoint_confidence[0] > confidence_threshold:
                head_x, head_y, _ = obj.keypoint[0]
                head_point = project_point(obj.keypoint[0], intrinsics, image_scale)
                cv2.putText(frame_bgr, f"{prediction}", (int(head_point[0]), int(head_point[1] - 20)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Live Pose Classification", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # === Cleanup ===
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
