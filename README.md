# Median housing value prediction

The housing data can be downloaded from https://raw.githubusercontent.com/ageron/handson-ml/master/. The script has codes to download the data. We have modelled the median house value on given housing data. 

The following techniques have been used: 

 - Linear regression
 - Decision Tree
 - Random Forest

## Steps performed
 - We prepare and clean the data. We check and impute for missing values.
 - Features are generated and the variables are checked for correlation.
 - Multiple sampling techinuqies are evaluated. The data set is split into train and test.
 - All the above said modelling techniques are tried and evaluated. The final metric used to evaluate is mean squared error.

## To excute the script
python < scriptname.py >

## Project Description
1. Create a private repository.
2. Create an Issue and a new branch with issue ID created.
3. Download python and readme files.
4. Run python files with required packages.
5. Create a python mle-dev environment and install required packages.
6. Create env yml and commit all files to issue branch.

## Commands
1. conda env export > env.yml
2. conda activate mle-dev
3  python3 nonstandardcode.py
4. conda env create -f env.yml
