import pyzed.sl as sl
import cv2
import numpy as np
import joblib
import torch
import os

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
confidence_threshold = 0.7

def extract_keypoints(obj):
    keypoints = obj.keypoint
    confidences = obj.keypoint_confidence
    features = []

    # Only include keypoints 0–7 and 14–17 → total 12 keypoints × 3 coords = 36 features
    selected_indices = list(range(8)) + list(range(14, 18))

    for i in selected_indices:
        x, y, z = keypoints[i]
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
    display_resolution = sl.Resolution(min(camera_info.camera_resolution.width, 1280),
                                       min(camera_info.camera_resolution.height, 720))
    image_scale = [display_resolution.width / camera_info.camera_resolution.width,
                   display_resolution.height / camera_info.camera_resolution.height]

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
            features_np = np.array(features, dtype=np.float32).reshape(1, -1)

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
                    coords_2d = project_point((x, y, z), intrinsics, image_scale)
                    if coords_2d:
                        cv2.circle(frame_bgr, (int(coords_2d[0]), int(coords_2d[1])), 4, (0, 255, 0), -1)

            # Show predicted label near head (keypoint 0)
            if obj.keypoint_confidence[0] > confidence_threshold:
                head_point = project_point(obj.keypoint[0], intrinsics, image_scale)
                if head_point:
                    cv2.putText(frame_bgr, f"{prediction}", (int(head_point[0]), int(head_point[1] - 20)),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

        cv2.imshow("Live Pose Classification (MLP)", frame_bgr)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    # === Cleanup ===
    zed.close()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
