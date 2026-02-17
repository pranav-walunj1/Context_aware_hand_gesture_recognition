import pyzed.sl as sl
import csv
import cv2
import numpy as np
import os
import datetime
#import ogl_viewer.viewer as gl
#import cv_viewer.tracking_viewer as cv_viewer

# Paths

#video_path = "D:/Pycharm Projects/Hand_pose_estimation/new_project/svo_dataset_new_dataset/zed_capture_2025-07-04_12-26-17.svo"    # video 1
#video_path = "D:/Pycharm Projects/Hand_pose_estimation/new_project/svo_dataset_new_dataset/zed_capture_2025-07-04_12-29-24.svo"    # video 2
#video_path = "D:/Pycharm Projects/Hand_pose_estimation/new_project/svo_dataset_new_dataset/zed_capture_2025-07-04_12-39-12.svo"    # video 3
#video_path = "D:/Pycharm Projects/Hand_pose_estimation/new_project/svo_dataset_new_dataset/zed_capture_2025-07-04_12-40-31.svo"    # video 4
#video_path = "D:/Pycharm Projects/Hand_pose_estimation/new_project/svo_dataset_new_dataset/zed_capture_2025-07-04_12-42-34.svo"    # video 5
#video_path = "D:/Pycharm Projects/Hand_pose_estimation/new_project/svo_dataset_new_dataset/zed_capture_2025-07-04_12-45-32.svo"    # video 6
#video_path = "D:/Pycharm Projects/Hand_pose_estimation/new_project/svo_dataset_new_dataset/zed_capture_2025-07-04_12-49-44.svo"    # video 7
#video_path = "D:/Pycharm Projects/Hand_pose_estimation/new_project/svo_dataset_new_dataset/zed_capture_2025-07-04_13-05-46.svo"    # video 8
#video_path = "D:/Pycharm Projects/Hand_pose_estimation/new_project/svo_dataset_new_dataset_test/zed_capture_2025-09-12_14-43-59.svo"     # Test dataset video 1
video_path = "D:/Pycharm Projects/Hand_pose_estimation/new_project/svo_dataset_new_dataset_test/zed_capture_2025-09-12_14-48-45.svo"     # Test dataset video 2


csv_path = "D:/Pycharm Projects/Hand_pose_estimation/new_project/main_char_new_dataset_creation/Test_dataset/test_dataset_body_keypoints2.csv"


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
    # Initialize ZED camera in SVO playback mode
    count = 0
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.svo_real_time_mode = False
    init_params.set_from_svo_file(video_path)
    #init_params.svo_real_time_mode = False
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED file.")
        exit(1)

    # === Set up .svo recording ===
    save_dir = "svo_dataset_new_dataset"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    svo_path = os.path.join(save_dir, f"zed_capture_{timestamp}.svo")
    
    
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
    
    # CSV writer
    os.makedirs(os.path.dirname(csv_path), exist_ok=True)
    csv_file = open(csv_path, mode='w', newline='')
    csv_writer = csv.writer(csv_file)

    headers = ['frame', 'person_id']
    for i in range(18):
        headers += [f'kp{i}_x', f'kp{i}_y', f'kp{i}_z', f'kp{i}_conf']
    csv_writer.writerow(headers)

    image = sl.Mat()
    objects = sl.Objects()
    frame_idx = 0
    #####################################################################################################################
    # Get ZED camera information
    camera_info = zed.get_camera_information()
    intrinsics = camera_info.camera_configuration.calibration_parameters.left_cam
    # 2D viewer utilities
    display_resolution = sl.Resolution(min(camera_info.camera_resolution.width, 1280),
                                       min(camera_info.camera_resolution.height, 720))
    image_scale = [display_resolution.width / camera_info.camera_resolution.width
        , display_resolution.height / camera_info.camera_resolution.height]

    # Save intrinsics and image_scale to JSON
    import json

    intrinsics_dict = {
        "fx": intrinsics.fx,
        "fy": intrinsics.fy,
        "cx": intrinsics.cx,
        "cy": intrinsics.cy,
        "k1": intrinsics.disto[0],
        "k2": intrinsics.disto[1],
        "p1": intrinsics.disto[2],
        "p2": intrinsics.disto[3],
        "k3": intrinsics.disto[4],
        "image_scale_x": image_scale[0],
        "image_scale_y": image_scale[1]
    }

    with open("zed_intrinsics.json", "w") as f:
        json.dump(intrinsics_dict, f, indent=4)

    #####################################################################################################################
    frames_output_dir = "output_frames"
    os.makedirs(frames_output_dir, exist_ok=True)
    
    print("Processing video...")
    #recording_params = sl.RecordingParameters(svo_path, sl.SVO_COMPRESSION_MODE.LOSSLESS)
    
    #if zed.enable_recording(recording_params) != sl.ERROR_CODE.SUCCESS:
    #    print("Failed to start recording")
    #    zed.close()
    #    exit(1)
        
        
    while zed.grab(runtime_params) == sl.ERROR_CODE.SUCCESS:
        zed.retrieve_image(image, sl.VIEW.LEFT)
        count += 1
        # Convert to OpenCV format first
        #image_left_ocv = image.get_data()
        
        zed.retrieve_objects(objects, obj_runtime_param)

        frame = image.get_data()
        frame_bgr = frame[:, :, :3].copy()

        for obj in objects.object_list:
            #if obj.body_format != sl.BODY_FORMAT.POSE_18:
            #    continue

            person_id = obj.id
            keypoints = obj.keypoint  # (18, 3)
            confidences = obj.keypoint_confidence  # (18,)

            # Write to CSV
            row = [frame_idx, person_id]
            for i in range(18):
                x, y, z = keypoints[i]
                c = confidences[i]
                row += [round(x, 2), round(y, 2), round(z, 2), round(c, 2)]
            csv_writer.writerow(row)

            # Draw keypoints
            for i, keypoint in enumerate(keypoints):
                if confidences[i] > 0.4:
                    coords_2d = project_point(keypoint, intrinsics, image_scale)
                    # Flipping the x-coordinate for image orientation
                    #frame_width = frame.shape[1]
                    #flipped_x = frame_width - coords_2d[0]
                    #cv2.circle(frame_bgr, (int(x), int(y)), 5, (0, 255, 0), -1)
                    #cv2.circle(frame_bgr, (int(coords_2d[0]), int(coords_2d[1])), 5, (0, 255, 0), -1)
                    # Draw person ID near head keypoint (kp0)
            if confidences[0] > 0.4:
                head_coords = project_point(keypoints[0], intrinsics, image_scale)
                if head_coords:
                    print('count is :', count)
                    #cv2.putText(frame_bgr, f"ID: {person_id}", (head_coords[0] + 10, head_coords[1] - 10),
                    #            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
        # Save visualized frame
        output_path = os.path.join(frames_output_dir, f"frame_{frame_idx:05d}.jpg")
        cv2.imwrite(output_path, frame_bgr)#, [cv2.IMWRITE_JPEG_QUALITY, 60])
        
        cv2.imshow("ZED Skeleton Tracking", frame_bgr)
        key = cv2.waitKey(1)
        if key == ord('q'):
            break

        frame_idx += 1

    # Cleanup
    # === Clean up ===
    zed.disable_recording()
    csv_file.close()
    zed.disable_object_detection()
    zed.close()
    cv2.destroyAllWindows()
    print(f"Finished. Keypoints saved to: {csv_path}")

if __name__ == "__main__":
    main()
