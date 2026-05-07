## Folder files execution
- So this folder contains all the scripts related to Random forest model, for hand gesture recognition dataset :
    - Important scripts :
        - `train_random_forest.py` : script to train random_forest model
	- `train_random_forest_location_invariant.py` : script to train random_forest model with added feature of hand location invariance
        - `inference_random_forest.py` : script to inference random_forest model on live zed camera data(video)
        
    
- Important other files that you need for execution : 

    	- `gesture_rf.pkl` : main model file
	- `gesture_rf_on_normalized_points.pkl` : main model file for model trained on normalized hand keypoints
	- `gesture_rf_on_normalized_points_wo_z_coords.pkl` : main model file for model trained on normalized hand keypoints without z-coordinate
	- `gesture_rf_relative_scale_norm_3d.pkl` : main model file for model trained on normalized hand keypoints and added feature of hand location to be invariant in the video frame(so that performance doesn't change based on the hand location in the frame)
        

## How to execute 
- So after you have done everything that is mentioned in the `How to run` section and installed everything from `Requirements` from the main README.md file of this repository, then :
    - just activate the venv by command : 
        - on Linux/MacOS OS cmd :
            - source venv/bin/activate

        - on Windows OS :
            - venv\Scripts\activate

    - execution of any script file :
        - python <name_of_file>.py