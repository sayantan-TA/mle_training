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

