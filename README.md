# Thermal-UDE: Physics-Informed Machine Learning for Battery Heat Generation Estimation and Temperature Prediction
This repository contains the dataset and Julia code for the article "Physics-Informed Machine Learning for Battery Heat Generation Estimation and Temperature Prediction" by the authors Ashima Kalathingal, Jishnu Ayyangatu Kuzhiyil, Thomas Grandjean, Truong Dinh, and Mona Faraji Niri. 

https://doi.org/10.1016/j.egyai.2026.100808

## Contents 

- Case_1 folder: Consists of 3 files corresponding to Case 1 of the paper
    1. `Dataset.jld2` : The data set needed to run the code
    2. `opt_para_case_1.jld2`   : The trained parameters of the neural network for Case 1
    3. `Temp_pred_case1.jl`          : The main code to run to get the results

- Case_2 folder: Consists of 3 files corresponding to Case 2 of the paper
    1. `Dataset.jld2` : The data set needed to run the code
    2. `opt_para_case_2.jld2`   : The trained parameters of the neural network
    3. `Temp_pred_case2.jl`          : The main code to run to get the results
 
## Usage
Run the main code in each folder using the corresponding dataset and parameters to view the results. Ensure that all the packages mentioned at the top of the main code are installed in the Julia environment. The conditions section of the code refers to the C-rate/Drive cycle and ambient temperature conditions in which the experiment is conducted. For example, in the conditions part, the following code excerpt ``Crate1, Temp1 = "0p5C",10`` means that under this condition, the cell is discharged at a 0.5C rate at 10°C. There are 12 conditions, and at most 6 can be simulated at a time. The Crate conditions are ``["0p5C","1C","2C","WLTP"]`` and the ambient temperature conditions are ``["0","10","25"]``.

## Citation
If you use the code or dataset from this repository, please cite the article

Kalathingal, Ashima and Ayyangatu Kuzhiyil, Jishnu and Grandjean, Thomas and Dinh, Truong and Faraji Niri, Mona, Physics-Informed Machine Learning for Battery Heat Generation Estimation and Temperature Prediction with Minimal Experimentation. Available at Energy and AI : https://doi.org/10.1016/j.egyai.2026.100808
 
