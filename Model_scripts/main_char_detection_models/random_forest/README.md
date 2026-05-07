## Folder files execution
- So this folder contains all the scripts related to Random forest model, for main character detection dataset :
    - Important scripts :
        - `train_random_forest.py` : script to train random_forest model
        - `inference_random_forest.py` : script to inference random_forest model on live zed camera data(video)
        - `test_random_forest.py` : script to test random_forest model on the separate test dataset which was captured for overall pipeline performance measure
	- `testing_pipeline_RF-MLP.py` : script to test whole pipeline on the separate test dataset with RF for stage 1 and MLP for stage 2

    - Important other files that you need for execution : 

        
    	- `gesture_rf.pkl` : main model file
        

## How to execute 
- So after you have done everything that is mentioned in the `How to run` section and installed everything from `Requirements` from the main README.md file of this repository, then :
    - just activate the venv by command : 
        - on Linux/MacOS OS cmd :
            - source venv/bin/activate

        - on Windows OS :
            - venv\Scripts\activate

    - execution of any script file :
        - python <name_of_file>.py