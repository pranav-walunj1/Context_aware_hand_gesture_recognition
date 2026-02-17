# evaluate_pipeline_offline_rf_then_mlp.py
import os
import math
import json
import numpy as np
import pandas as pd
import cv2
import joblib
import torch
import mediapipe as mp
from sklearn.metrics import (
    accuracy_score, classification_report, f1_score
)

# -----------------------
# CONFIG
# -----------------------
ZED_CSV = "D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_new_dataset_creation/sampled_dataset/zed_dataset.csv"
HAND_CSV = "D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_new_dataset_creation/sampled_dataset/hand_gesture_dataset.csv"
IMAGES_FOLDER = "D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_new_dataset_creation/Test_dataset/yolo_8_m/auto_labelled_dataset/unlabelled_refined_frames"

# 1️⃣ Main-char model: Random Forest
MAIN_CHAR_MODEL_PATH = "D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_det_models/random_forest/gesture_rf.pkl"

# 2️⃣ Gesture model (same as before)
GESTURE_MODEL_PTH = "D:/Pycharm Projects/Hand_pose_estimation/new_project/hand_gesture_recognition/classifier_trained_on_dataset_mp_labelled/MLP/mlp_model_30_E_scale_norm_3d_more_unknown_poses.pth"
GESTURE_LE_PKL = "D:/Pycharm Projects/Hand_pose_estimation/new_project/hand_gesture_recognition/classifier_trained_on_dataset_mp_labelled/MLP/label_encoder_30_E_scale_norm_3d_more_unknown_poses.pkl"

# ZED intrinsics for projection
intrinsics_file = "D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_new_dataset_creation/zed_intrinsics.json"
with open(intrinsics_file, "r") as f:
    intrinsics = json.load(f)
INTRINSICS = {"fx": intrinsics["fx"], "fy": intrinsics["fy"], "cx": intrinsics["cx"], "cy": intrinsics["cy"]}
IMAGE_SCALE = (intrinsics["image_scale_x"], intrinsics["image_scale_y"])

OUT_FOLDER = "pipeline_eval_results_rf"
os.makedirs(OUT_FOLDER, exist_ok=True)
PER_IMAGE_LOG_CSV = os.path.join(OUT_FOLDER, "per_image_results.csv")

USE_PROJECTION = True
SCALE_FACTOR = 1.0
MIN_BOX = 100
MAX_BOX = 400
WRIST_EDGE_FRACTION = 0.25

# -----------------------
# Helper functions
# -----------------------

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


def project_point_3d_to_2d(point3D, intr, image_scale=(1.0, 1.0)):
    x, y, z = point3D
    if z == 0 or np.isnan(z):
        return None
    u = int((intr["fx"] * x / z) + intr["cx"])
    v = int((intr["fy"] * y / z) + intr["cy"])
    u = int(u * image_scale[0])
    v = int(v * image_scale[1])
    return (u, v)

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
    cxA, cyA = (boxA[0] + boxA[2]) / 2.0, (boxA[1] + boxA[3]) / 2.0
    cxB, cyB = (boxB[0] + boxB[2]) / 2.0, (boxB[1] + boxB[3]) / 2.0
    return math.hypot(cxA - cxB, cyA - cyB)

def extract_selected_features_from_zed_row(row):
    selected_idx = list(range(0, 8)) + list(range(14, 18))
    feats = []
    for i in selected_idx:
        feats.extend([float(row[f"kp{i}_x"]), float(row[f"kp{i}_y"]), float(row[f"kp{i}_z"])])
    return np.array(feats, dtype=np.float32)

def build_predicted_hand_bbox_from_row(row, use_projection=True):
    lw, le = (row["kp7_x"], row["kp7_y"], row["kp7_z"]), (row["kp6_x"], row["kp6_y"], row["kp6_z"])
    rw, re = (row["kp4_x"], row["kp4_y"], row["kp4_z"]), (row["kp3_x"], row["kp3_y"], row["kp3_z"])

    lw2 = project_point_3d_to_2d(lw, INTRINSICS, IMAGE_SCALE)
    le2 = project_point_3d_to_2d(le, INTRINSICS, IMAGE_SCALE)
    rw2 = project_point_3d_to_2d(rw, INTRINSICS, IMAGE_SCALE)
    re2 = project_point_3d_to_2d(re, INTRINSICS, IMAGE_SCALE)

    candidates = []
    if lw2 and le2: candidates.append(("left", lw2, le2))
    if rw2 and re2: candidates.append(("right", rw2, re2))
    if not candidates: return None

    chosen = min(candidates, key=lambda c: c[1][1])  # smaller y = higher wrist
    wrist2d, elbow2d = np.array(chosen[1]), np.array(chosen[2])
    dist = np.linalg.norm(wrist2d - elbow2d)
    box_size = int(max(MIN_BOX, min(MAX_BOX, dist * SCALE_FACTOR)))
    dir_vec = (wrist2d - elbow2d) / (np.linalg.norm(wrist2d - elbow2d) + 1e-6)
    center = wrist2d + dir_vec * (box_size * WRIST_EDGE_FRACTION)
    cx, cy = int(center[0]), int(center[1])
    return [max(0, cx - box_size // 2), max(0, cy - box_size // 2),
            cx + box_size // 2, cy + box_size // 2]

# -----------------------
# Load models
# -----------------------
print("Loading Random Forest main-char model...")
main_model = joblib.load(MAIN_CHAR_MODEL_PATH)
main_model_type = "sklearn"
print("Main char model loaded:", MAIN_CHAR_MODEL_PATH)

# Gesture model (same as before)
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
    def forward(self, x): return self.net(x)

gesture_le = joblib.load(GESTURE_LE_PKL)
gesture_model = KeypointMLP(63, 128, len(gesture_le.classes_))
gesture_model.load_state_dict(torch.load(GESTURE_MODEL_PTH))
gesture_model.eval()
print("Gesture model loaded:", gesture_le.classes_)

# Mediapipe instance
mp_hands = mp.solutions.hands.Hands(static_image_mode=True, max_num_hands=1, min_detection_confidence=0.3)

# -----------------------
# Load CSVs
# -----------------------
df_zed = pd.read_csv(ZED_CSV)
df_hand = pd.read_csv(HAND_CSV)
gt_hand_by_image = df_hand.set_index("image").to_dict(orient="index")

# -----------------------
# Evaluation
# -----------------------
#images = sorted(df_hand["image"].unique())
images = sorted(df_zed["image"].unique())
results = []
iou_list, center_list, gesture_gt, gesture_pred = [], [], [], []

# NEW: for main-char evaluation
mainchar_y_true, mainchar_y_pred = [], []

print(f"Evaluating {len(images)} images...")

for img_name in images:
    print('processing image :', img_name)
    humans = df_zed[df_zed["image"] == img_name]
    if humans.empty: continue

    # Evaluate all humans
    best_score, best_row = -1, None
    for _, row in humans.iterrows():
        feats = extract_selected_features_from_zed_row(row).reshape(1, -1)
        pred = main_model.predict(feats)[0]
        # collect predictions
        mainchar_y_pred.append(pred)
        mainchar_y_true.append(row["Label"])  # <-- assumes CSV has true label column

        try:
            probs = main_model.predict_proba(feats)[0]
            score = probs[list(main_model.classes_).index("Active_main_char")] if "Active_main_char" in main_model.classes_ else max(probs)
        except:
            score = 1.0 if pred == "Active_main_char" else 0.0
        if score > best_score:
            best_score, best_row = score, row

    if best_row is None: continue
    pred_bbox = build_predicted_hand_bbox_from_row(best_row)
    gt = gt_hand_by_image.get(img_name)
    #print('gt is :', gt)
    if not gt: continue
    gt_bbox = [gt["x_min"], gt["y_min"], gt["x_max"], gt["y_max"]]
    #print('gt_bbox is :', gt_bbox)
    #print('pred_bbox is :', pred_bbox)
    iou = compute_iogt(pred_bbox, gt_bbox)
    #print('iou is :', iou)
    center_d = center_distance(pred_bbox, gt_bbox)
    iou_list.append(iou)
    center_list.append(center_d)
    
    # Gesture prediction from crop
    # Update image name (zero-pad to 5 digits and add .jpg)
    img_name = str(img_name).zfill(5) + ".jpg"
    img_path = os.path.join(IMAGES_FOLDER, img_name)
    img = cv2.imread(img_path)
    pred_gesture, pred_conf = "NO_DET", 0.0
    if img is not None:
        x1, y1, x2, y2 = pred_bbox
        crop = img[y1:y2, x1:x2]
        rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        rgb_crop, padding = resize_with_padding(rgb, (480, 640))
        res = mp_hands.process(rgb_crop)
        if res.multi_hand_landmarks:
            lm = res.multi_hand_landmarks[0]
            kp = np.array([[l.x, l.y, l.z] for l in lm.landmark]).flatten()
            #wrist = kp[:3].copy()
            #kp = kp - np.tile(wrist, 21)
            #kp = kp / (np.linalg.norm(kp) + 1e-6)
            normalized_kp = normalize_hand_keypoints_3d_from_list(kp).reshape(1, -1)
            with torch.no_grad():
                out = gesture_model(torch.tensor(normalized_kp, dtype=torch.float32))
                probs = torch.softmax(out, dim=1).cpu().numpy()[0]
                pred_idx = np.argmax(probs)
                pred_gesture = gesture_le.inverse_transform([pred_idx])[0]
                pred_conf = probs[pred_idx]
    gesture_gt.append(gt["gesture"])
    gesture_pred.append(pred_gesture)
    results.append({"image": img_name, "iou": iou, "center_dist": center_d,
                    "gt_gesture": gt["gesture"], "pred_gesture": pred_gesture,
                    "pred_conf": pred_conf,
                    "gt_bbox_x1": int(gt_bbox[0]), "gt_bbox_y1": int(gt_bbox[1]),
                    "gt_bbox_x2": int(gt_bbox[2]), "gt_bbox_y2": int(gt_bbox[3]),
                    "pred_bbox_x1": None if pred_bbox is None else int(pred_bbox[0]),
                    "pred_bbox_y1": None if pred_bbox is None else int(pred_bbox[1]),
                    "pred_bbox_x2": None if pred_bbox is None else int(pred_bbox[2]),
                    "pred_bbox_y2": None if pred_bbox is None else int(pred_bbox[3])
                    })


# -----------------------
# Main-character metrics
# -----------------------
if mainchar_y_true and mainchar_y_pred:
    print("\n=== Main-char classification metrics (per human row) ===")
    print("Samples (humans):", len(mainchar_y_true))
    print(classification_report(mainchar_y_true, mainchar_y_pred))
    print("Accuracy:", accuracy_score(mainchar_y_true, mainchar_y_pred))

# -----------------------
# Metrics
# -----------------------
valid = [i for i, p in enumerate(gesture_pred) if p not in ("NO_DET","MP_NOT_DETECTED")]
if valid:
    y_true = [gesture_gt[i] for i in valid]
    y_pred = [gesture_pred[i] for i in valid]
    print("\nGesture Classification Report:")
    print(classification_report(y_true, y_pred))
    print("Accuracy:", accuracy_score(y_true, y_pred))
    print("F1 (macro):", f1_score(y_true, y_pred, average="macro"))
print("\nMean IoU:", np.nanmean(iou_list))
print("IoU ≥ 0.5 Rate:", np.mean(np.array(iou_list) >= 0.5))
print("Mean Center Distance:", np.nanmean(center_list))

pd.DataFrame(results).to_csv(PER_IMAGE_LOG_CSV, index=False)
print(f"\nResults saved to {PER_IMAGE_LOG_CSV}")
# -----------------------
# Visualization: draw GT / Pred boxes + labels and save images
# -----------------------
VIS_FOLDER = os.path.join(OUT_FOLDER, "vis")
os.makedirs(VIS_FOLDER, exist_ok=True)

font = cv2.FONT_HERSHEY_SIMPLEX
for r in results:
    # r["image"] is already zero-padded + ".jpg"
    img_path = os.path.join(IMAGES_FOLDER, r["image"])
    if not os.path.exists(img_path):
        # optionally warn or continue silently
        # print("Missing image:", img_path)
        continue
    
    img = cv2.imread(img_path)
    if img is None:
        continue

    # Draw GT bbox (green)
    gx1, gy1, gx2, gy2 = int(r["gt_bbox_x1"]), int(r["gt_bbox_y1"]), int(r["gt_bbox_x2"]), int(r["gt_bbox_y2"])
    cv2.rectangle(img, (gx1, gy1), (gx2, gy2), (0, 255, 0), 2)
    cv2.putText(img, f"GT: {r['gt_gesture']}", (gx1, max(gy1-8,0)), font, 0.6, (0,255,0), 2, cv2.LINE_AA)

    # Draw predicted bbox (red) if present
    if r["pred_bbox_x1"] is not None:
        px1, py1, px2, py2 = int(r["pred_bbox_x1"]), int(r["pred_bbox_y1"]), int(r["pred_bbox_x2"]), int(r["pred_bbox_y2"])
        # clamp to image bounds
        h, w = img.shape[:2]
        px1, py1 = max(0, px1), max(0, py1)
        px2, py2 = min(w-1, px2), min(h-1, py2)
        cv2.rectangle(img, (px1, py1), (px2, py2), (0, 0, 255), 2)
        cv2.putText(img, f"Pred: {r['pred_gesture']} ({r['pred_conf']:.2f})", (px1, max(py1-8,0)), font, 0.6, (0,0,255), 2, cv2.LINE_AA)
    else:
        # show reason as NO_DET
        cv2.putText(img, f"Pred: {r['pred_gesture']}", (10, 20), font, 0.6, (0,0,255), 2, cv2.LINE_AA)

    out_path = os.path.join(VIS_FOLDER, r["image"])
    cv2.imwrite(out_path, img)

print(f"All visualizations saved to: {VIS_FOLDER}")



# -----------------------
# Save per-person main-char predictions
# -----------------------
mainchar_records = []

for idx, row in df_zed.iterrows():
    feats = extract_selected_features_from_zed_row(row).reshape(1, -1)
    pred_label = main_model.predict(feats)[0]
    gt_label = row["Label"]
    image_name = row["image"]
    person_id = row.get("person_id", "unknown")  # Use "unknown" if column missing
    mainchar_records.append({
        "person_id": person_id,
        "image": image_name,
        "gt_main_char": gt_label,
        "pred_main_char": pred_label
    })

# Convert to DataFrame and save
df_mainchar = pd.DataFrame(mainchar_records)
excel_path = os.path.join(OUT_FOLDER, "mainchar_predictions.xlsx")
df_mainchar.to_excel(excel_path, index=False)
print(f"\nMain-character prediction details saved to: {excel_path}")
