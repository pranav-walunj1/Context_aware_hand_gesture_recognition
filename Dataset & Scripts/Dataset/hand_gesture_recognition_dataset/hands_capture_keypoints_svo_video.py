

video_path = "D:/Pycharm Projects/Hand_pose_estimation/new_project/hand_gesture_recognition/svo_dataset_gesture_recognition/zed_capture_2025-08-20_14-55-53.svo"


import cv2
import mediapipe as mp
import pyzed.sl as sl
import numpy as np

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
    # Initialize MediaPipe
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.5)
    mp_drawing = mp.solutions.drawing_utils
    
    # Initialize ZED
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.svo_real_time_mode = True
    init_params.set_from_svo_file(video_path)
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA
    
    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED file.")
        exit(1)
        
    positional_tracking_parameters = sl.PositionalTrackingParameters()
    zed.enable_positional_tracking(positional_tracking_parameters)
    
    # Set object detection parameters (for body tracking)
    obj_param = sl.ObjectDetectionParameters()
    obj_param.enable_body_fitting = True
    obj_param.enable_tracking = True
    obj_param.detection_model = sl.DETECTION_MODEL.HUMAN_BODY_FAST
    obj_param.body_format = sl.BODY_FORMAT.POSE_18
    zed.enable_object_detection(obj_param)
    
    # Runtime parameters
    runtime_params = sl.RuntimeParameters()
    obj_runtime_param = sl.ObjectDetectionRuntimeParameters()
    obj_runtime_param.detection_confidence_threshold = 40
    # Prepare ZED image container
    image_zed = sl.Mat()
    body_data = sl.Objects()
    # Get ZED camera information
    camera_info = zed.get_camera_information()
    intrinsics = camera_info.camera_configuration.calibration_parameters.left_cam
    # 2D viewer utilities
    display_resolution = sl.Resolution(min(camera_info.camera_resolution.width, 1280),
                                       min(camera_info.camera_resolution.height, 720))
    image_scale = [display_resolution.width / camera_info.camera_resolution.width
        , display_resolution.height / camera_info.camera_resolution.height]
    
    print("Processing video...")
    while zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
    
        zed.retrieve_image(image_zed, sl.VIEW.LEFT)
        
    
        zed.retrieve_objects(body_data, obj_runtime_param)
        
        frame = image_zed.get_data()
        frame_bgr = frame[:, :, :3].copy()
        frame = frame_bgr
        bodies = body_data.object_list
        print('inside while')
        
        for obj in bodies:
            person_id = obj.id
            keypoints = obj.keypoint  # (18, 3)
            confidences = obj.keypoint_confidence  # (18,)
            # Extract wrist keypoints and confidence scores
            left_wrist_kp = keypoints[7]
            right_wrist_kp = keypoints[4]
            left_conf = confidences[7]
            right_conf = confidences[4]
            
            #if confidences[7] > 0.5 or confidences[4] > 0.5:
            #    if confidences[7] > confidences[4] or np.isnan(confidences[4]):
            #        wrist_kp = keypoints[7]
            #    elif confidences[7] < confidences[4] or np.isnan(confidences[7]:
            #        wrist_kp = keypoints[4]
            #    wrist = project_point(wrist_kp, intrinsics, image_scale)
            
            #    x, y = int(wrist[0]), int(wrist[1])
            # Check for valid confidence on either wrist
            if (not np.isnan(left_conf) and left_conf > 0.5) or (not np.isnan(right_conf) and right_conf > 0.5):
                # Choose the wrist with higher confidence
                if (np.isnan(right_conf) or left_conf > right_conf):
                    wrist_kp = left_wrist_kp
                else:
                    wrist_kp = right_wrist_kp
    
                # Project and draw
                wrist = project_point(wrist_kp, intrinsics, image_scale)
                x, y = int(wrist[0]), int(wrist[1])
                # Crop a region above the wrist
                roi_size = 400
                x1 = max(0, x - roi_size // 2)
                y1 = max(0, y - roi_size - 40)
                x2 = min(frame.shape[1], x + roi_size // 2)
                y2 = min(frame.shape[0], y + 40)
                hand_roi = frame[y1:y2, x1:x2]
    
                # Feed ROI to MediaPipe
                image_rgb = cv2.cvtColor(hand_roi, cv2.COLOR_BGRA2RGB)
                results = hands.process(image_rgb)
    
                # Visualize results
                if results.multi_hand_landmarks:
                    for hand_landmarks in results.multi_hand_landmarks:
                        mp_drawing.draw_landmarks(
                            hand_roi, hand_landmarks, mp_hands.HAND_CONNECTIONS
                        )
    
                # Draw on full frame
                cv2.circle(frame, (x, y), 6, (0, 255, 0), -1)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 0), 2)
    
        # Display
        cv2.imshow("ZED Frame", frame[:, :, :3])  # Convert BGRA to BGR
        key = cv2.waitKey(10)
        if key == ord('q'):
            break
    
    zed.close()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
    