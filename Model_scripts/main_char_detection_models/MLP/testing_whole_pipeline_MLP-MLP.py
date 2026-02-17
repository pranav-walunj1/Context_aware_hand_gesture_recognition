# evaluate_pipeline_offline.py
import os
import math
import time
import json
from math import sqrt
from typing import Optional, Tuple, Dict

import numpy as np
import pandas as pd
import cv2
import joblib
import torch
import mediapipe as mp
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    classification_report, confusion_matrix
)

# -----------------------
# CONFIG: set these paths
# -----------------------
ZED_CSV = "D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_new_dataset_creation/Test_dataset/yolo_8_m/auto_labelled_dataset/main_char_dataset_new.csv"
HAND_CSV = "D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_new_dataset_creation/Test_dataset/yolo_8_m/auto_labelled_dataset/hand_keypoints_gesture_new.csv"
IMAGES_FOLDER = "D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_new_dataset_creation/Test_dataset/yolo_8_m/auto_labelled_dataset/unlabelled_refined_frames"

# main-char model (sklearn .pkl or torch .pth). If torch .pth, set MAIN_CHAR_LE_PATH.
#MAIN_CHAR_MODEL_PATH = "D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_det_models/random_forest/gesture_rf.pkl"
#MAIN_CHAR_LE_PATH = None  # or path to label encoder used for main-char torch model (if needed)
MAIN_CHAR_MODEL_PATH = "D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_det_models/MLP/mlp_model_30_E.pth"
MAIN_CHAR_LE_PATH ="D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_det_models/MLP/label_encoder_30_E.pkl"
# gesture model: torch .pth (KeypointMLP) + label encoder .pkl
GESTURE_MODEL_PTH = 'D:/Pycharm Projects/Hand_pose_estimation/new_project/hand_gesture_recognition/classifier_trained_on_dataset_mp_labelled/MLP/mlp_model_30_E_scale_norm_3d_more_unknown_poses.pth'
GESTURE_LE_PKL = "D:/Pycharm Projects/Hand_pose_estimation/new_project/hand_gesture_recognition/classifier_trained_on_dataset_mp_labelled/MLP/label_encoder_30_E_scale_norm_3d.pkl"

# Whether ZED kp_x / kp_y are already image pixel coordinates (False = 3D, requires projection)
USE_PROJECTION = True

# If USE_PROJECTION = True, provide intrinsics & image resolution (edit values)
# Path to your intrinsics file
intrinsics_file = "D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_new_dataset_creation/zed_intrinsics.json"

# Load the JSON file
with open(intrinsics_file, "r") as f:
    intrinsics = json.load(f)
INTRINSICS = {"fx": intrinsics["fx"], "fy": intrinsics["fy"], "cx": intrinsics["cx"], "cy": intrinsics["cy"]}  # example; replace with your camera
IMAGE_RESOLUTION = (1280, 720)  # (width, height)
IMAGE_SCALE = (intrinsics["image_scale_x"], intrinsics["image_scale_y"])  # if your image files have different resolution than intrinsics, set scale

# Mediapipe settings for offline hand extraction from crops
MP_STATIC_MODE = True
MP_MAX_HANDS = 1
MP_MIN_DET_CONF = 0.3

# predicted bbox generation params
SCALE_FACTOR = 1.0
MIN_BOX = 100
MAX_BOX = 400
WRIST_EDGE_FRACTION = 0.25  # how close wrist sits to bbox back edge

# output
OUT_FOLDER = "pipeline_eval_results"
os.makedirs(OUT_FOLDER, exist_ok=True)
PER_IMAGE_LOG_CSV = os.path.join(OUT_FOLDER, "per_image_results.csv")

# -----------------------
# Helper functions
# -----------------------
def compute_iou(boxA, boxB):
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH
    boxAArea = max(0, (boxA[2] - boxA[0])) * max(0, (boxA[3] - boxA[1]))
    boxBArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
    union = boxAArea + boxBArea - interArea
    return interArea / union if union > 0 else 0.0

def compute_iogt(boxA, boxB):
    # boxA = predicted, boxB = ground truth
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    interW = max(0, xB - xA)
    interH = max(0, yB - yA)
    interArea = interW * interH

    gtArea = max(0, (boxB[2] - boxB[0])) * max(0, (boxB[3] - boxB[1]))
    return interArea / gtArea if gtArea > 0 else 0.0

def center_distance(boxA, boxB):
    cxA = (boxA[0] + boxA[2]) / 2.0
    cyA = (boxA[1] + boxA[3]) / 2.0
    cxB = (boxB[0] + boxB[2]) / 2.0
    cyB = (boxB[1] + boxB[3]) / 2.0
    return math.hypot(cxA - cxB, cyA - cyB)

def project_point_3d_to_2d(point3D: Tuple[float,float,float], intr: Dict, image_scale=(1.0,1.0)):
    """Project 3D camera coords (x,y,z) to pixel (u,v). Return (u,v) or None on invalid z."""
    x, y, z = point3D
    if z == 0 or np.isnan(z):
        return None
    u = int((intr["fx"] * x / z) + intr["cx"])
    v = int((intr["fy"] * y / z) + intr["cy"])
    u = int(u * image_scale[0])
    v = int(v * image_scale[1])
    return (u, v)

def extract_selected_features_from_zed_row(row: pd.Series):
    """Match your live logic: select keypoints 0..7 and 14..17 (12 keypoints -> 36 features)"""
    selected_idx = list(range(0, 8)) + list(range(14, 18))
    feats = []
    for i in selected_idx:
        # column names like 'kp0_x','kp0_y','kp0_z'
        feats.extend([
            float(row[f"kp{i}_x"]), float(row[f"kp{i}_y"]), float(row[f"kp{i}_z"])
        ])
    return np.array(feats, dtype=np.float32)

# Keypoint normalization for gesture model (wrt wrist and scale)
def normalize_hand_keypoints_3d_from_list(kp_list):
    """kp_list: flat list of 63 values [x0,y0,z0, x1,y1,z1, ...] where x,y are normalized in crop or pixels?
       This function does translation (subtract wrist) and scale normalization.
    """
    arr = np.array(kp_list, dtype=np.float32).reshape(-1, 3)
    wrist = arr[0].copy()
    arr = arr - wrist
    dists = np.linalg.norm(arr, axis=1)
    maxd = np.max(dists)
    if maxd > 0:
        arr = arr / maxd
    return arr.flatten()

# MediaPipe: single Hands instance reused
mp_hands_module = mp.solutions.hands
mp_hands_instance = mp_hands_module.Hands(static_image_mode=MP_STATIC_MODE, max_num_hands=MP_MAX_HANDS, min_detection_confidence=MP_MIN_DET_CONF)
mp_drawing = mp.solutions.drawing_utils


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

def mediapipe_keypoints_from_crop(crop_bgr):
    """Return list of 63 normalized values (lm.x, lm.y, lm.z) or None"""
    if crop_bgr is None or crop_bgr.size == 0:
        return None
    rgb = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2RGB)
    rgb_crop, padding = resize_with_padding(rgb, (480, 640))
    results = mp_hands_instance.process(rgb_crop)
    if not results.multi_hand_landmarks:
        return None
    lm = results.multi_hand_landmarks[0]
    kp = []
    for l in lm.landmark:
        kp.extend([float(l.x), float(l.y), float(l.z)])
    return kp

def build_predicted_hand_bbox_from_row(row: pd.Series, use_projection=False, intrinsics=None, image_scale=(1.0,1.0)):
    """Return predicted bbox [x1,y1,x2,y2] in pixel coords (integers), or None if cannot."""
    # left wrist index = 7, right wrist index = 4; left elbow = 6, right elbow = 3 (same as your live code)
    left_wrist = (row.get("kp7_x"), row.get("kp7_y"), row.get("kp7_z"))
    left_elbow = (row.get("kp6_x"), row.get("kp6_y"), row.get("kp6_z"))
    right_wrist = (row.get("kp4_x"), row.get("kp4_y"), row.get("kp4_z"))
    right_elbow = (row.get("kp3_x"), row.get("kp3_y"), row.get("kp3_z"))
    #use_projection = True
    # optionally project 3D -> 2D
    if use_projection:
        lw2 = project_point_3d_to_2d(left_wrist, intrinsics, image_scale)
        le2 = project_point_3d_to_2d(left_elbow, intrinsics, image_scale)
        rw2 = project_point_3d_to_2d(right_wrist, intrinsics, image_scale)
        re2 = project_point_3d_to_2d(right_elbow, intrinsics, image_scale)
    else:
        # treat kp_x, kp_y as pixel coords
        lw2 = (int(left_wrist[0]), int(left_wrist[1])) if (left_wrist[0] is not None and not np.isnan(left_wrist[0])) else None
        le2 = (int(left_elbow[0]), int(left_elbow[1])) if (left_elbow[0] is not None and not np.isnan(left_elbow[0])) else None
        rw2 = (int(right_wrist[0]), int(right_wrist[1])) if (right_wrist[0] is not None and not np.isnan(right_wrist[0])) else None
        re2 = (int(right_elbow[0]), int(right_elbow[1])) if (right_elbow[0] is not None and not np.isnan(right_elbow[0])) else None

    # choose candidate(s) with valid coords
    candidates = []
    if lw2 and le2:
        candidates.append(("left", lw2, le2))
    if rw2 and re2:
        candidates.append(("right", rw2, re2))
    if not candidates:
        return None

    # choose higher wrist (smaller y)
    if len(candidates) == 1:
        chosen = candidates[0]
    else:
        chosen = candidates[0] if candidates[0][1][1] < candidates[1][1][1] else candidates[1]

    chosen_wrist2d = np.array(chosen[1], dtype=np.float32)
    chosen_elbow2d = np.array(chosen[2], dtype=np.float32)

    # compute wrist->elbow distance in pixels to set box size
    wrist_elbow_dist = float(np.linalg.norm(chosen_wrist2d - chosen_elbow2d))
    box_size = int(wrist_elbow_dist * SCALE_FACTOR)
    box_size = max(MIN_BOX, min(MAX_BOX, box_size))

    dir_vec = chosen_wrist2d - chosen_elbow2d
    if np.linalg.norm(dir_vec) == 0:
        dir_vec = np.array([0.0, -1.0])
    dir_vec = dir_vec / np.linalg.norm(dir_vec)

    center = chosen_wrist2d + dir_vec * (box_size * WRIST_EDGE_FRACTION)
    cx, cy = int(center[0]), int(center[1])

    x1 = max(0, cx - box_size // 2)
    y1 = max(0, cy - box_size // 2)
    x2 = cx + box_size // 2
    y2 = cy + box_size // 2

    return [int(x1), int(y1), int(x2), int(y2)]

# -----------------------
# Load models
# -----------------------
print("Loading models...")
# Load main-char model (sklearn .pkl assumed here; if .pth adapt similarly)
if MAIN_CHAR_MODEL_PATH.lower().endswith(".pkl"):
    main_model = joblib.load(MAIN_CHAR_MODEL_PATH)
    main_model_type = "sklearn"
else:
    # assume PyTorch state dict
    main_model_le = joblib.load(MAIN_CHAR_LE_PATH) if MAIN_CHAR_LE_PATH else None
    # recreate KeypointMLP used earlier with input_dim=36
    class KeypointMLPMain(torch.nn.Module):
        def __init__(self, input_dim=36, hidden_dim=128, num_classes=3):
            super().__init__()
            self.net = torch.nn.Sequential(
                torch.nn.Linear(input_dim, hidden_dim),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(hidden_dim, 64),
                torch.nn.ReLU(),
                torch.nn.Dropout(0.3),
                torch.nn.Linear(64, num_classes)
            )
        def forward(self, x):
            return self.net(x)
    if main_model_le is None:
        raise ValueError("If main char model is a .pth, you must provide MAIN_CHAR_LE_PATH to reconstruct classes.")
    main_model = KeypointMLPMain(input_dim=36, hidden_dim=128, num_classes=len(main_model_le.classes_))
    main_model.load_state_dict(torch.load(MAIN_CHAR_MODEL_PATH))
    main_model.eval()
    main_model_type = "torch"
print("Main char model type:", main_model_type)

# Gesture model load (PyTorch)
class KeypointMLP(torch.nn.Module):
    def __init__(self, input_dim=63, hidden_dim=128, num_classes=3):
        super().__init__()
        self.net = torch.nn.Sequential(
            torch.nn.Linear(input_dim, hidden_dim),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(hidden_dim, 64),
            torch.nn.ReLU(),
            torch.nn.Dropout(0.3),
            torch.nn.Linear(64, num_classes)
        )
    def forward(self, x):
        return self.net(x)

gesture_le = joblib.load(GESTURE_LE_PKL)
gesture_model = KeypointMLP(input_dim=63, hidden_dim=128, num_classes=len(gesture_le.classes_))
gesture_model.load_state_dict(torch.load(GESTURE_MODEL_PTH))
gesture_model.eval()
print("Gesture model loaded. Classes:", gesture_le.classes_)

# -----------------------
# Load CSVs
# -----------------------
print("Loading CSVs...")
df_zed = pd.read_csv(ZED_CSV)
df_hand = pd.read_csv(HAND_CSV)

# build mapping from image->GT hand row (if multiple rows per image keep first; adapt if needed)
gt_hand_by_image = df_hand.set_index("image").to_dict(orient="index")

# -----------------------
# Main evaluation loop
# -----------------------
per_image_rows = []
main_gt_labels = []   # per-human evaluation arrays
main_pred_labels = []
main_pred_probs = []

gesture_gt = []
gesture_pred = []
gesture_pred_conf = []
iou_list = []
center_list = []

# for each image, we will select the predicted active main char (highest active score)
images = sorted(list(set(df_hand["image"].tolist())))  # iterate over images that have a GT hand
print(f"Evaluating {len(images)} images...")

for img_name in images:
    # get all humans in this frame
    humans = df_zed[df_zed["image"] == img_name]
    if humans.empty:
        # no humans detected in zed csv
        continue

    # For per-human main-char metrics (evaluate classifier on each human row)
    for idx, row in humans.iterrows():             # here it iterates over all the humans from the zed csv file
        feats = extract_selected_features_from_zed_row(row)     # here we extract all 12 upperbody keypoints of each human from zed csv file
        if main_model_type == "sklearn":
            # sklearn model: features shape (n_samples, n_feats)
            pred = main_model.predict(feats.reshape(1, -1))[0]
            try:
                proba = float(max(main_model.predict_proba(feats.reshape(1,-1))[0]))
            except Exception:
                proba = 1.0
        else:
            with torch.no_grad():
                logits = main_model(torch.tensor(feats.reshape(1,-1), dtype=torch.float32))
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                pred_idx = int(np.argmax(probs))
                # map to label via provided label encoder
                pred = main_model_le.inverse_transform([pred_idx])[0]
                proba = float(probs[pred_idx])

        main_gt_labels.append(row["Label"])
        main_pred_labels.append(pred)
        main_pred_probs.append(proba)

    # Now pick the best candidate (highest probability of Active_main_char) as chosen main char
    best_candidate = None
    best_active_score = -1.0
    best_row = None
    for idx, row in humans.iterrows():
        feats = extract_selected_features_from_zed_row(row)
        if main_model_type == "sklearn":
            # If predict_proba available and model knows the class label index
            try:
                probs = main_model.predict_proba(feats.reshape(1,-1))[0]
                # find index of Active_main_char in model.classes_ if present
                target_class = "Active_main_char"
                if hasattr(main_model, "classes_"):
                    classes = list(main_model.classes_)
                    if target_class in classes:
                        score = float(probs[classes.index(target_class)])
                    else:
                        # fallback: probability of predicted class
                        score = max(probs)
                else:
                    score = max(probs)
            except Exception:
                # fallback to deterministic predict (1.0 if predicted active)
                p = main_model.predict(feats.reshape(1,-1))[0]
                score = 1.0 if p == "Active_main_char" else 0.0
        else:
            with torch.no_grad():
                logits = main_model(torch.tensor(feats.reshape(1,-1), dtype=torch.float32))
                probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                # main_model_le maps indices->labels
                classes = list(main_model_le.classes_)
                target_class = "Active_main_char"
                if target_class in classes:
                    score = float(probs[classes.index(target_class)])
                else:
                    score = float(np.max(probs))
        if score > best_active_score:
            best_active_score = score
            best_candidate = row
    # best_candidate is a pandas Series row for the chosen human
    if best_candidate is None:
        # no candidate, skip
        continue

    # Build predicted bbox from chosen human row
    chosen_row = best_candidate
    pred_bbox = build_predicted_hand_bbox_from_row(chosen_row, use_projection=USE_PROJECTION, intrinsics=INTRINSICS, image_scale=IMAGE_SCALE)
    # get GT hand data for this image
    if img_name not in gt_hand_by_image:
        # no GT hand for this image
        continue
    gt = gt_hand_by_image[img_name]
    gt_bbox = [int(gt["x_min"]), int(gt["y_min"]), int(gt["x_max"]), int(gt["y_max"])]
    gt_gesture = gt["gesture"]

    # IoU + center distance
    if pred_bbox is None:
        iou = 0.0
        cdist = None
    else:
        #iou = compute_iou(pred_bbox, gt_bbox)
        iou = compute_iogt(pred_bbox, gt_bbox)
        cdist = center_distance(pred_bbox, gt_bbox)
    iou_list.append(iou)
    center_list.append(cdist if cdist is not None else np.nan)

    # Now run MediaPipe on the predicted bbox crop (if bbox exists and image available)
    image_path = os.path.join(IMAGES_FOLDER, img_name)
    pred_g = "NO_DETECTION"
    pred_conf = 0.0
    if pred_bbox is not None and os.path.exists(image_path):
        img = cv2.imread(image_path)
        h,w = img.shape[:2]
        x1,y1,x2,y2 = pred_bbox
        # clamp
        x1,y1 = max(0,x1), max(0,y1)
        x2,y2 = min(w-1,x2), min(h-1,y2)
        if x2 > x1 and y2 > y1:
            crop = img[y1:y2, x1:x2]
            mp_kp = mediapipe_keypoints_from_crop(crop)
            if mp_kp is not None and len(mp_kp) == 63:
                # normalize relative wrist and predict
                normalized_kp = normalize_hand_keypoints_3d_from_list(mp_kp).reshape(1,-1)
                with torch.no_grad():
                    logits = gesture_model(torch.tensor(normalized_kp, dtype=torch.float32))
                    probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                    pred_idx = int(np.argmax(probs))
                    pred_g = gesture_le.inverse_transform([pred_idx])[0]
                    pred_conf = float(probs[pred_idx])
            else:
                pred_g = "MP_NOT_DETECTED"
                pred_conf = 0.0
        else:
            pred_g = "INVALID_BBOX"
            pred_conf = 0.0
    else:
        if not os.path.exists(image_path):
            pred_g = "NO_IMAGE"
            pred_conf = 0.0
        else:
            pred_g = "NO_BBOX"
            pred_conf = 0.0

    gesture_gt.append(gt_gesture)
    gesture_pred.append(pred_g)
    gesture_pred_conf.append(pred_conf)

    per_image_rows.append({
        "image": img_name,
        "pred_bbox_x1": None if pred_bbox is None else pred_bbox[0],
        "pred_bbox_y1": None if pred_bbox is None else pred_bbox[1],
        "pred_bbox_x2": None if pred_bbox is None else pred_bbox[2],
        "pred_bbox_y2": None if pred_bbox is None else pred_bbox[3],
        "gt_bbox_x1": gt_bbox[0], "gt_bbox_y1": gt_bbox[1], "gt_bbox_x2": gt_bbox[2], "gt_bbox_y2": gt_bbox[3],
        "iou": iou,
        "center_dist": cdist if cdist is not None else np.nan,
        "gt_gesture": gt_gesture,
        "pred_gesture": pred_g,
        "pred_gesture_conf": pred_conf,
        "main_active_score": float(best_active_score)
    })

# -----------------------
# Compute & print metrics
# -----------------------
print("\n=== Main-char classification metrics (per human row) ===")
print("Samples (humans):", len(main_gt_labels))
print(classification_report(main_gt_labels, main_pred_labels, zero_division=0))
print("Accuracy:", accuracy_score(main_gt_labels, main_pred_labels))

print("\n=== Gesture classification (pipeline) ===")
# filter out invalid preds (strings like NO_DETECTION etc) optionally
valid_idx = [i for i, p in enumerate(gesture_pred) if p not in ("MP_NOT_DETECTED","NO_BBOX","NO_IMAGE","NO_DETECTION","INVALID_BBOX")]
if len(valid_idx) == 0:
    print("No valid gesture predictions to evaluate.")
else:
    y_true = [gesture_gt[i] for i in valid_idx]
    y_pred = [gesture_pred[i] for i in valid_idx]
    print("Samples (valid predictions):", len(valid_idx))
    print(classification_report(y_true, y_pred, zero_division=0))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1 (macro):", f1_score(y_true, y_pred, average="macro", zero_division=0))

# Detection bbox stats:
iou_arr = np.array([v for v in iou_list if v is not None])
print("\n=== BBox statistics ===")
print("Mean IoU:", float(np.nanmean(iou_arr)) if iou_arr.size>0 else 0.0)
print("IoU >= 0.5 rate:", float(np.sum(iou_arr >= 0.5))/len(iou_arr) if iou_arr.size>0 else 0.0)
print("Mean center distance (pixels):", np.nanmean(center_list))

# Save per-image CSV for inspection
df_out = pd.DataFrame(per_image_rows)
df_out.to_csv(PER_IMAGE_LOG_CSV, index=False)
print(f"\nPer-image log saved to: {PER_IMAGE_LOG_CSV}")




# -----------------------
# Visualization
# -----------------------
VIS_FOLDER = os.path.join(OUT_FOLDER, "vis")
os.makedirs(VIS_FOLDER, exist_ok=True)

font = cv2.FONT_HERSHEY_SIMPLEX

print("Saving visualizations to:", VIS_FOLDER)

for row in per_image_rows:
    img_path = os.path.join(IMAGES_FOLDER, row["image"])
    if not os.path.exists(img_path):
        continue
    img = cv2.imread(img_path)

    # Draw GT bbox (green)
    gt_bbox = [row["gt_bbox_x1"], row["gt_bbox_y1"], row["gt_bbox_x2"], row["gt_bbox_y2"]]
    cv2.rectangle(img, (gt_bbox[0], gt_bbox[1]), (gt_bbox[2], gt_bbox[3]), (0, 255, 0), 2)
    cv2.putText(img, f"GT: {row['gt_gesture']}", (gt_bbox[0], gt_bbox[1] - 10),
                font, 0.6, (0, 255, 0), 2, cv2.LINE_AA)

    # Draw Pred bbox (red) if available
    if row["pred_bbox_x1"] is not None:
        pred_bbox = [row["pred_bbox_x1"], row["pred_bbox_y1"], row["pred_bbox_x2"], row["pred_bbox_y2"]]
        cv2.rectangle(img, (pred_bbox[0], pred_bbox[1]), (pred_bbox[2], pred_bbox[3]), (0, 0, 255), 2)
        cv2.putText(img, f"Pred: {row['pred_gesture']} ({row['pred_gesture_conf']:.2f})",
                    (pred_bbox[0], pred_bbox[1] - 10), font, 0.6, (0, 0, 255), 2, cv2.LINE_AA)

        # Draw IoU and center dist
        text = f"IoU: {row['iou']:.2f}, Cdist: {row['center_dist']:.1f}"
        cv2.putText(img, text, (10, img.shape[0] - 10), font, 0.6, (255, 255, 0), 2, cv2.LINE_AA)


        # === NEW: Draw Active Main Char label + keypoints ===
        # Load the ZED row again for this image to fetch keypoints of chosen main char
        humans = df_zed[df_zed["image"] == row["image"]]
        if not humans.empty:
            # Pick the one with highest active score (same logic as above)
            best_score, best_row = -1, None
            for _, r in humans.iterrows():
                feats = extract_selected_features_from_zed_row(r)
                if main_model_type == "torch":
                    with torch.no_grad():
                        logits = main_model(torch.tensor(feats.reshape(1,-1), dtype=torch.float32))
                        probs = torch.softmax(logits, dim=1).cpu().numpy()[0]
                        score = float(probs[list(main_model_le.classes_).index("Active_main_char")])
                else:
                    try:
                        probs = main_model.predict_proba(feats.reshape(1,-1))[0]
                        score = probs[list(main_model.classes_).index("Active_main_char")]
                    except:
                        score = 0.0
                if score > best_score:
                    best_score, best_row = score, r

            if best_row is not None:
                # Project 3D keypoints -> 2D
                kp2d = []
                for i in range(18):  # assuming 18 upper-body keypoints in ZED
                    x = best_row.get(f"kp{i}_x", np.nan)
                    y = best_row.get(f"kp{i}_y", np.nan)
                    z = best_row.get(f"kp{i}_z", np.nan)
                    if not np.isnan(x) and not np.isnan(y) and not np.isnan(z):
                        p2d = project_point_3d_to_2d((x, y, z), INTRINSICS, IMAGE_SCALE)
                        if p2d:
                            kp2d.append(p2d)
                            cv2.circle(img, p2d, 4, (0, 255, 255), -1)

                # Draw label near bbox
                cv2.putText(img, "Active Main Char", (pred_bbox[0], pred_bbox[1] - 30),
                            font, 0.8, (0, 0, 255), 2, cv2.LINE_AA)

    out_path = os.path.join(VIS_FOLDER, row["image"])
    cv2.imwrite(out_path, img)

print(f"All visualizations saved to: {VIS_FOLDER}")
print("\nDone.")
