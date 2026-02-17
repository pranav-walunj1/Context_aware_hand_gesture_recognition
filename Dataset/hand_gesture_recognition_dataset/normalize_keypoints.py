import pandas as pd

# Read CSV
df = pd.read_csv("D:/Pycharm Projects/Hand_pose_estimation/new_project/hand_gesture_recognition/hand_gesture_data_labelled_from_mediapipe/open_palm/open_palm_keypoints_2 - Copy.csv")

# Normalize all x columns by 640
df.loc[:, df.columns.str.endswith('_x')] /= 640

# Normalize all y columns by 480
df.loc[:, df.columns.str.endswith('_y')] /= 480

# Save new CSV
df.to_csv("normalized_keypoints1111.csv", index=False)
