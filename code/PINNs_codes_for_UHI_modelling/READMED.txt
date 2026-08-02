Source code for the paper: “Physics-Informed Neural Network Calibration and Application for Urban Cooling”. Please cite the original paper when using this code:

Nizamani, A. G., Vu, D. T., Nizamani, R. A., Li, C. (2025). Physics-Informed Neural Network Calibration and Application for Urban Cooling

Installation

Please use the file requirements.txt to install the correct dependencies for the project

How the code works

The codes follow a sequence of steps:

1. Download Landsat 8 satellite data via Google Earth Engineer (https://earthengine.google.com/) 
•	Landsat_GEE_data.txt

2. Download Urban Morphology of study area via OSMnx library in Python
	•	urban_features_Paris.py

3. Training of the PINNs model: PINNs_model.py

4. Results of PINNs model for UHI modelling: 
	•	PINNs_predicted_LST.py
	•	PINNs_for_GRs_LST_simulation.py

#----Another models for comparison with PINNs model:
•	RF_pred_LST.py
•	XGBoost_pred_LST.py
•	SVM_pred_LST.py

•	CNN_pred_LST.py
•	GCN_pred_LST.py
•	U_Net_pred_LST.py
•	



	




