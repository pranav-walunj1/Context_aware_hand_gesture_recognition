import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix
import joblib

# ===============================
# 1. Load Model + Label Encoder
# ===============================
model_path = "D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_det_models/MLP/mlp_model_30_E.pth"
encoder_path = "D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_det_models/MLP/label_encoder_30_E.pkl"
test_csv = "D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_new_dataset_creation/Test_dataset/labelled_dataset/main_char_dataset_cleaned_upperbody_wo_conf.csv"

label_encoder = joblib.load(encoder_path)

# Define model (must match architecture used during training)
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
            nn.Linear(64, num_classes)
        )

    def forward(self, x):
        return self.net(x)

model = KeypointMLP(input_dim=36, hidden_dim=128, num_classes=len(label_encoder.classes_))
model.load_state_dict(torch.load(model_path))
model.eval()

# ===============================
# 2. Load Test Data
# ===============================
df_test = pd.read_csv(test_csv)

X_test = df_test.drop(columns=['Label', 'image']).values.astype("float32")
y_test = label_encoder.transform(df_test['Label'].values)

X_test_tensor = torch.tensor(X_test)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

test_dataset = TensorDataset(X_test_tensor, y_test_tensor)
test_loader = DataLoader(test_dataset, batch_size=32)

# ===============================
# 3. Evaluation
# ===============================
all_preds, all_labels = [], []

with torch.no_grad():
    for inputs, labels in test_loader:
        outputs = model(inputs)
        _, preds = torch.max(outputs, 1)
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

from sklearn.utils.multiclass import unique_labels

# Find which classes are present in the test set
present_classes = unique_labels(all_labels, all_preds)
# ===============================
# 4. Metrics
# ===============================
print("=== Classification Report ===")
print(classification_report(all_labels, all_preds, target_names=label_encoder.classes_[present_classes]))

acc = accuracy_score(all_labels, all_preds)
precision = precision_score(all_labels, all_preds, average='macro')
recall = recall_score(all_labels, all_preds, average='macro')
f1 = f1_score(all_labels, all_preds, average='macro')
cm = confusion_matrix(all_labels, all_preds)

print("\n=== Metrics on Test Dataset ===")
print(f"Accuracy: {acc:.4f}")
print(f"Precision (macro): {precision:.4f}")
print(f"Recall (macro): {recall:.4f}")
print(f"F1-score (macro): {f1:.4f}")

print("\n=== Confusion Matrix ===")
print(cm)
