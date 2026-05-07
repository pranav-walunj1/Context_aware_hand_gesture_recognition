import pandas as pd

# === Step 1: Read Excel file ===
df = pd.read_csv(r"D:\Pycharm Projects\Hand_pose_estimation\new_project\main_char_det_models\MLP\pipeline_eval_results\per_image_results.csv")  # <-- replace with your actual filename

# === Step 2: Define success conditions ===
# (1) Main Character correctness (use threshold on score)
main_threshold = 0.5
df["MC_correct"] = df["main_active_score"] >= main_threshold

# (2) Gesture correctness
df["Gesture_correct"] = df["pred_gesture"] == df["gt_gesture"]

# (3) Wrist crop success (based on IoU)
iou_threshold = 0.5
df["Crop_success"] = df["iou"] >= iou_threshold

# (4) End-to-end success (all correct together)
df["E2E_success"] = df["MC_correct"] & df["Crop_success"] & df["Gesture_correct"]

# === Step 3: Compute individual accuracies ===
MC_acc = df["MC_correct"].mean()
Crop_success = df["Crop_success"].mean()
Gesture_acc = df["Gesture_correct"].mean()
E2E_acc = df["E2E_success"].mean()

# === Step 4: Compute final pipeline performance index ===
Pipeline_Index = (
    0.3 * MC_acc
    + 0.2 * Crop_success
    + 0.25 * Gesture_acc
    + 0.25 * E2E_acc
)

# === Step 5: Print all results ===
print(f"Main Character Accuracy: {MC_acc*100:.2f}%")
print(f"Wrist Crop Success: {Crop_success*100:.2f}%")
print(f"Gesture Accuracy: {Gesture_acc*100:.2f}%")
print(f"End-to-End Accuracy: {E2E_acc*100:.2f}%")
print(f"\nFinal Pipeline Performance Index: {Pipeline_Index*100:.2f}%")
