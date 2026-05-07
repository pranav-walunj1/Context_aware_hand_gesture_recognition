# Master Project: Context-Aware Hand Gesture Recognition through Main Character Detection

## Description
This project contains codes for model training scripts and dataset creation scripts.

## Folder structure
- `folder Dataset/` → Contains two sub folders `folder main_char_detection_dataset/   and   hand_gesture_recognition_dataset/` :
    - `folder main_char_detection_dataset/` → Contains scripts related to the dataset which was used to train models for detecting main 
                                              character from a video frame
        - Scripts details :
            - capture_video_svo.py   :  this script is used to capture the video data from zed camera, in .svo (Stereolabs Video Object) 
                                        format, which is the video file which we can use and manipulate using the zed sdk. Such as to get 
                                        the human skeleton keypoints and depth of each keypoints.
            - capture_keypoints_new.py  :  this script is used to load the svo video file in the zed sdk that we got in previous script, 
                                           and get 18 human skeleton keypoints with their depth(distance) from the camera.
            
    - `folder hand_gesture_recognition_dataset/` → Contains scripts related to the dataset which was used to train models for recognizing 
                                                   the hand gestures of humans from the video frames
        - Scripts details :
            - gather_hand_keypoints_webcam_video.py : this script is used to capture video data from my laptop's webcam, then process on 
                                                      each frame, scuh that save each frame as an Image, so that mediapipe model is 
                                                      executed on each frame to get the 21 hand keypoints of a hand from the video frame, 
                                                      and finally those 21 keypoints are stored into a .csv file.
            - hands_capture_keypoints_svo_video.py : this script is used to process the svo video data which was captured by zed camera, 
                                                     then load it into zed sdk, save theach frame as an image file, and then process on 
                                                     each frame, so that mediapipe model is executed on each frame to get the 21 hand 
                                                     keypoints of a hand from the video frame, and finally those 21 keypoints are stored 
                                                     into a .csv file.
- `folder Model_scripts/` → Contains two sub folders `folder main_char_detection_models/   and   hand_gesture_recognition_models/` :
    - `folder main_char_detection_models/` → Contains scripts related to the 3 lightweight ML models (random forest, logistic regression 
                                             and multilayer perceptron ; MLP) which was used to train them for detecting main character 
                                             from a video frame. And it has subfolders, with each associated with each model.
        - `folder random_forest/` → Contains scripts related to the random forest model :
            - train_random_forest.py : script to train the random forest model
            - test_random_forest.py : script to test the performance of random forest model
            - inference_random_forest.py : script to do inferencing on the trained random forest model
            - testing_pipeline_RF-MLP.py : script to test the performance of whole pipeline with random forest model for detecting main 
                                           char and MLP for hand gesture recognition.
        - `folder logistic_regression/` → Contains scripts related to the logistic regression model :
            - train_logistic_regression.py : script to train the logistic regression model
            - test_logistic_regression.py : script to test the performance of logistic regression model
            - inference_logistic_regression.py : script to do inferencing on the trained logistic_regression model
            - inference_logistic_regression_poly.py : script to do inferencing on the trained logistic_regression model with polynomial 
                                                      transform for achieving better result.  
        - `folder MLP/` → Contains scripts related to the multi layer perceptron model.

     - `folder hand_gesture_recognition_models/` → Contains scripts related to the 3 lightweight ML models (random forest, logistic 
                                                   regression and multilayer perceptron ; MLP) which was used to train them for 
                                                   recognizing hand gestures from a video frame. And it has subfolders, with each 
                                                   associated with each model.
            


## Requirements
- Python-3.8.10
- torch-2.4.1, Ultralytics 8.3.202, ZED_SDK_Windows10_cuda10.2_v3.8.2 
- Other dependencies listed in `requirements.txt`

## How to run
1. Prepare dataset in the correct folder structure
2. Create the venv based on the libraries from `requirements.txt`, and activate it
2. Run training scripts: through directly through venv
3. Evaluate results using: the test scripts asscoiated with each model

## Author
Pranav Vijay Walunj
