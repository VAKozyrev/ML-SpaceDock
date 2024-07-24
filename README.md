# ML-SpaceDock

## Introduction
ML-SpaceDock is an active learning workflow aimed to retrive maximum number of hits from the virtual library with minimum data exploration

## Requirements
1. Python 3.10
2. Conda environment creted from environmet.yml
3. Reagent building blocks and hit lists (provided in 'data' directory)

## Installation 
1. Clone the repository
2. Create a conda environment
  ```
  conda env create -f environment.yml
  ```

## Data 

## Reproduce the results of the paper

## Exploration of large chemical spaces of multiple chemical reactions
Requirements:

    1. Morgan fingerprints of the building blocks saved as .npy files
    2. Reaction rules matrix saved as .npy file
    3. Indexes of hit pairs saved as .npy file

All of these can be obtained from the list of SMILES strings of building blocks for each reaction type (data/CBLB/building_blocks.tsv) by running the CBLB_descriptors_generation_and_indexing.ipynb notebook.
Some cells must be modified in case additional reactions are added or removed.

After all required files are generated, we can run the active learning workflow, which consists of 4 Python scripts:

    1. select_0_batch.py
    2. train.py
    3. predict.py
    4. select_batch.py

Script 1 should be run only once to select the first random batch of pairs to start exploration. Afterward, scripts 2, 3, and 4 are run iteratively until the desired portion of the chemical space is explored.




 
