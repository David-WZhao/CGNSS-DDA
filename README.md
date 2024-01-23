# CGNSS-DDA

This is the source code for an efficient negative sample selection framework for drug repurposing proposed in the manuscript which is in under review.

### Requirement
python 3.6  
pytorch == 1.10.1+cu102  
dgl == 0.4.1  
numpy == 1.19.5  
scipy ==1.5.4  


### Dataset
The input data consist of five files, which are placed under the "my_dataset" directory with the corresponding default file names. 
Note that the procedure of data collection is described in our manuscript, and the original data is accessible upon request.

### Run the code
To run the prediction model on the input dataset, use the command: "python my_model_K_Means.py". The performance according to the evaluation measure (AUPR and AUC) will be output in the console.
Note that the detailed prediction result (e.g., the prediction on specific samples) is contained in the object "predict" (in Line 233 of "my_model_K_Means.py"), which can be exported as a text file conveniently.


### Instruction for three main scripts
my_model_K_Means.py: the code to conduct the prediction model. 
my_utils.py: the code to complete the loading and preprocessing the input data set.  
con_sim_clu_cf2.py: the code to complete the negative sample selection process.

