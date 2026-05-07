## Folder files execution
- So this folder contains all the scripts related to Logistic Regression model, for hand gesture recognition dataset :
    - Important scripts :
        - `train_logistic_regression.py` : script to train logistic_regression model
	- `train_logistic_regression_location_invariant.py` : script to train logistic_regression model with hand location to be anywhere in the video frame
        - `inference_logistic_regression.py` : script to inference logistic_regression model on live zed camera data(video)
	

    - Important other files that you need for execution : 

        - For model trained : 
            - `logistic_regression_model.pkl` : main model file
            - `label_ecoder.pkl` : label encoder for this model file `logistic_regression_model.pkl`
	    - `scaler.pkl` : scaler for this model file `logistic_regression_model.pkl`

	- For model trained after adding the feature of training on normalized hand keypoints (0-1): 
            - `logistic_regression_model_normalzied_points.pkl` : main model file
            - `label_ecoder_normalzied_points.pkl` : label encoder for this model file `logistic_regression_model_normalzied_points.pkl`
	    - `scaler_normalzied_points.pkl` : scaler for this model file `logistic_regression_model_normalzied_points.pkl` 

	- For model trained after adding the feature of being invariant to hand location with normalized points: 
            - `logistic_regression_model_relative_scale_norm_3d.pkl` : main model file
            - `label_ecoder_relative_scale_norm_3d.pkl` : label encoder for this model file `logistic_regression_model_relative_scale_norm_3d.pkl`
	    - `scaler_relative_scale_norm_3d.pkl` : scaler for this model file `logistic_regression_model_relative_scale_norm_3d.pkl` 
		
	- For model trained after adding the feature of being invariant to hand location with normalized points, with more unknown poses: 
            - `logistic_regression_model_relative_scale_norm_3d_more_unknown_poses.pkl` : main model file
            - `label_ecoder_relative_scale_norm_3d_more_unknown_poses.pkl` : label encoder for this model file `logistic_regression_model_relative_scale_norm_3d_more_unknown_poses.pkl`
	    - `scaler_relative_scale_norm_3d_more_unknown_poses.pkl` : scaler for this model file `logistic_regression_model_relative_scale_norm_3d_more_unknown_poses.pkl` 
        

## How to execute 
- So after you have done everything that is mentioned in the `How to run` section and installed everything from `Requirements` from the main README.md file of this repository, then :
    - just activate the venv by command : 
        - on Linux/MacOS OS cmd :
            - source venv/bin/activate

        - on Windows OS :
            - venv\Scripts\activate

    - execution of any script file :
        - python <name_of_file>.py