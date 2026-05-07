## Folder files execution
- So this folder contains all the scripts related to MLP model, for hand gesture recognition dataset :
    - Important scripts :
        - `train_MLP.py` : script to train MLP model
        - `inference_MLP.py` : script to inference MLP model on live zed camera data(video)

    - Important other files that you need for execution : 

        - For model : 
            - `mlp_model_30_E_scale_norm_3d.pth` : main model file
            - `label_ecoder_30_E_scale_norm_3d.pkl` : lable encoder for this model file `mlp_model_30_E_scale_norm_3d.pth`

        - For model trained with more unknown poses : 
            - `mlp_model_30_E_scale_norm_3d_more_unknown_poses.pth` : main model file
            - `label_ecoder_30_E_scale_norm_3d_more_unknown_poses.pkl` : lable encoder for this model file `mlp_model_30_E_scale_norm_3d_more_unknown_poses.pth`

## How to execute 
- So after you have done everything that is mentioned in the `How to run` section and installed everything from `Requirements` from the main README.md file of this repository, then :
    - just activate the venv by command : 
        - on Linux/MacOS OS cmd :
            - source venv/bin/activate

        - on Windows OS :
            - venv\Scripts\activate

    - execution of any script file :
        - python <name_of_file>.py