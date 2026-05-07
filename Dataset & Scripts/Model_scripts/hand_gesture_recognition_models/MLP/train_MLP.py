import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

import time
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, TensorDataset


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

# ===============================
# 1. Load and preprocess dataset
# ===============================
#df = pd.read_csv("D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_new_dataset_creation/body_keypoints_output_labelled_combined_upperbody_wo_conf.csv")
# === Load dataset ===
#df = pd.read_csv(
#    "D:/Pycharm Projects/Hand_pose_estimation/new_project/"
#    "hand_gesture_recognition/hand_gesture_data_labelled_from_mediapipe/"
#    "normalized_keypoints.csv"  # Now includes z
#)
df = pd.read_csv(
    "D:/Pycharm Projects/Hand_pose_estimation/new_project/"
    "hand_gesture_recognition/hand_gesture_data_labelled_from_mediapipe/"
    "normalized_keypoints_more_unknown_poses.csv"  # Now includes more unknown poses
)
# Features: 54 coords
#X_raw = df.drop(columns=['Label']).values.astype(np.float32)
X_raw = df.drop(columns=['Label'])

# Apply relative+scale normalization to every row
X = np.array([normalize_hand_keypoints_3d(row) for _, row in X_raw.iterrows()])

# Labels: Encode string classes -> integers
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(df['Label'].values)  # e.g. 0=Active_main_char, 1=Inactive_main_char, 2=Non_main_char

# Split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, stratify=y, random_state=42
)

# Convert to PyTorch tensors
X_train_tensor = torch.tensor(X_train, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

train_dataset = TensorDataset(X_train_tensor, y_train_tensor)
test_dataset = TensorDataset(X_test_tensor, y_test_tensor)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32)

# ===============================
# 2. Define MLP Model
# ===============================
class KeypointMLP(nn.Module):
    def __init__(self, input_dim=36, hidden_dim=128, num_classes=3):
        super(KeypointMLP, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, num_classes)  # no softmax here, handled by CrossEntropyLoss
        )

    def forward(self, x):
        return self.net(x)

model = KeypointMLP(input_dim=63, hidden_dim=128, num_classes=len(label_encoder.classes_))

# ===============================
# 3. Training Setup
# ===============================
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# ===============================
# 4. Training Loop
# ===============================
#num_epochs = 20
num_epochs = 30

# Start timer
start_time = time.time()

for epoch in range(num_epochs):
    model.train()
    running_loss = 0.0
    for inputs, labels in train_loader:
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {running_loss/len(train_loader):.4f}")


# End timer
end_time = time.time()
training_time = end_time - start_time
# Print training time
print(f"\nTotal training time: {training_time:.2f} seconds")
print(f"Training time: {training_time/60:.2f} minutes")
print(f"Average time per epoch: {training_time/num_epochs:.2f} seconds")
# ===============================
# 5. Evaluation
# ===============================
model.eval()
all_preds, all_labels = [], []
with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_))

acc = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')
cm = confusion_matrix(all_labels, all_preds)

print("Accuracy:", acc)
print("Precision (macro):", precision)
print("Recall (macro):", recall)
print("F1-score (macro):", f1)
print("Confusion Matrix:\n", cm)

# ===============================
# 6. Save Model + Label Encoder
# ===============================
torch.save(model.state_dict(), "D:/Pycharm Projects/Hand_pose_estimation/new_project/hand_gesture_recognition/classifier_trained_on_dataset_mp_labelled/MLP/mlp_model_30_E_scale_norm_3d_more_unknown_poses.pth")
import joblib
joblib.dump(label_encoder, "D:/Pycharm Projects/Hand_pose_estimation/new_project/hand_gesture_recognition/classifier_trained_on_dataset_mp_labelled/MLP/label_encoder_30_E_scale_norm_3d_more_unknown_poses.pkl")
