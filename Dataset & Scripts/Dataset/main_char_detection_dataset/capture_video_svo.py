import pyzed.sl as sl
import datetime
import os

def main():
    # === Set up ZED camera ===
    zed = sl.Camera()
    init_params = sl.InitParameters()
    init_params.camera_resolution = sl.RESOLUTION.HD720
    init_params.depth_mode = sl.DEPTH_MODE.ULTRA  # Enables depth!
    init_params.coordinate_units = sl.UNIT.MILLIMETER

    if zed.open(init_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to open ZED camera.")
        exit(1)

    # === Set up .svo recording ===
    save_dir = "svo_dataset_new_dataset_to_test_for_myself"
    os.makedirs(save_dir, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    svo_path = os.path.join(save_dir, f"zed_capture_{timestamp}.svo")

    # Set compression mode
    #recording_params = sl.RecordingParameters(svo_path, sl.SVO_COMPRESSION_MODE.H264)  # Use NONE if GPU encoding fails
    #recording_params = sl.RecordingParameters(svo_path, sl.SVO_COMPRESSION_MODE.NONE)
    recording_params = sl.RecordingParameters(svo_path, sl.SVO_COMPRESSION_MODE.LOSSLESS)

    if zed.enable_recording(recording_params) != sl.ERROR_CODE.SUCCESS:
        print("Failed to start recording")
        zed.close()
        exit(1)

    print(f"Recording... Press 'q' to stop. Saving to: {svo_path}")

    # === Start capturing ===
    while True:
        if zed.grab() == sl.ERROR_CODE.SUCCESS:
            # You can display the live preview (optional)
            image = sl.Mat()
            zed.retrieve_image(image, sl.VIEW.LEFT)
            frame = image.get_data()
            cv2.imshow("ZED Camera", frame[:, :, :3])

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # === Clean up ===
    zed.disable_recording()
    zed.close()
    print("SVO video saved successfully.")

if __name__ == "__main__":
    import cv2
    main()
