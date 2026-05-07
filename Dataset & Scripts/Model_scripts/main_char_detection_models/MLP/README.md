## Folder files execution
- So this folder contains all the scripts related to MLP model, for main character detection dataset :
    - Important scripts :
        - `train_MLP.py` : script to train MLP model
        - `inference_MLP.py` : script to inference MLP model on live zed camera data(video)
        - `test_MLP.py` : script to test MLP model on the separate test dataset which was captured for overall pipeline performance measure

    - Important other files that you need for execution : 

        - For model trained for 10 epochs : 
            - `mlp_model.pth` : main model file
            - `label_ecoder.pkl` : lable encoder for this model file `mlp_model.pth`

        - For model trained for 30 epochs : 
            - `mlp_model_30_E.pth` : main model file
            - `label_ecoder_30_E.pkl` : lable encoder for this model file `mlp_model_30_E.pth`

        - For model trained for 50 epochs : 
            - `mlp_model_50_E.pth` : main model file
            - `label_ecoder_50_E.pkl` : lable encoder for this model file `mlp_model_50_E.pth`

## How to execute 
- So after you have done everything that is mentioned in the `How to run` section and installed everything from `Requirements` from the main README.md file of this repository, then :
    - just activate the venv by command : 
        - on Linux/MacOS OS cmd :
            - source venv/bin/activate

        - on Windows OS :
            - venv\Scripts\activate

    - execution of any script file :
        - python <name_of_file>.py