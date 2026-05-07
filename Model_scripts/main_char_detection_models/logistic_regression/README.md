## Folder files execution
- So this folder contains all the scripts related to Logistic Regression model, for main character detection dataset :
    - Important scripts :
        - `train_logistic_regression.py` : script to train logistic_regression model
        - `inference_logistic_regression.py` : script to inference logistic_regression model on live zed camera data(video)
	- `inference_logistic_regression_poly.py` : script to inference logistic_regression with polynomial transform to degree 2 , (this was done to improve the performance) model on live zed camera data(video)
        - `test_logistic_regression.py` : script to test logistic_regression model on the separate test dataset which was captured for overall pipeline performance measure

    - Important other files that you need for execution : 

        - For model trained : 
            - `logistic_regression_model.pkl` : main model file
            - `label_ecoder.pkl` : label encoder for this model file `logistic_regression_model.pkl`

        - For model trained with polynomial transform (improved performance) : 
            - `logistic_regression_model_poly.pkl` : main model file
            - `label_encoder_poly.pkl` : label encoder for this model file `logistic_regression_model_poly.pkl`

## How to execute 
- So after you have done everything that is mentioned in the `How to run` section and installed everything from `Requirements` from the main README.md file of this repository, then :
    - just activate the venv by command : 
        - on Linux/MacOS OS cmd :
            - source venv/bin/activate

        - on Windows OS :
            - venv\Scripts\activate

    - execution of any script file :
        - python <name_of_file>.py