## Project (as per Assignmnt)
 - Project Desciption : Housing Dataset
 - Command to Create env.yml: conda env export --name mle-dev > env.yaml
 - Command to activate the enviroment: conda activate mle-dev



 - Command to create enviroment from env.yaml: conda env create -f env.yaml

 ## Command to run and Install package
 ### Way 1: Build code
  #### (from mle_training)
- pip install build
- python -m build

### Way 2: Install through pip
 #### (from mle_training)
- pip install .

### For Developer Purpose installation
- pip install -e .

### Run python code:
- python <script>.py
#### Run Main Code:
- python main.py --workflow ingest --log-level DEBUG --log-to-file
- python main.py --workflow train --log-level DEBUG --log-to-file
- python main.py --workflow score --log-level DEBUG --log-to-file

## Import Package
- import HousePricePrediction.<module_name> as hp

## Command to create python distribution
- conda install build
- pip install build
- python -m build
- pip install HousePricePrediction-0.0.1con.tar.gz > installation_log.txt 2>&1

## Command to install Docker
- docker pull sayantanchakrab/mle_training:v1
- docker run -it -p 5000:5000 sayantanchakrab/mle_training:v1

