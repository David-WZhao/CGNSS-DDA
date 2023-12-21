# CGNSS-DDA

### 注意事项
This is the code for an efficient negative sample selection framework for drug repurposing in the manuscript we are reviewing.

### Requirement
python 3.6
pytorch
dgl == 0.4.1
numpy== 1.19.5
scipy==1.5.4


### Data
The procedure of data collection is described in our manuscript, and the processed data is accessible upon request.

### Main Script
The code of the framework main program: my_model_K_Means.py;
The main program loads the tool code of the data set: my_utils.py
Code about the negative sample selection process: con_sim_clu_cf2.py.


### How to run the code?
python my_model_K_Means.py
