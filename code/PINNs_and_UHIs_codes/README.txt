
PINNs-quantifying temperature reduction-UHIs

Source code for the paper: “Smart Urban Design with Physics-Informed Neural Networks: Quantifying Temperature Green Infrastructure Using Satellite Thermal Data”. Please cite the original paper when using this code:

Vu, D. T., Nizamani, R. A., Nizamani, A. G., (2025). Smart Urban Design with Physics-Informed Neural Networks: Quantifying Temperature Green Infrastructure Using Satellite Thermal Data. arx....????

Installation

Use the file requirements.txt to install the correct dependencies for the project

How the code works

The code follows a sequence of steps:

1. Download Landsat 8 satellite data via Google Earth Engineer (https://earthengine.google.com/) 
•	GEE_data.txt

2. Download Urban Morphology of study area via OSMnx library in Python
	•	data_OSMnx.py

3. Training of the model: PINNs_train.py

4. Result: 
	•	PINNs_predicted_LST.py
	•	PINNs_green_roofs_predicted_LST.py
	•	PINNs_combined_trees_green_roofs_predicted_LST.py




