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


### `Train.py`

This script trains the MLP classifier model and saves its parameters as a `.pt` file. 

#### Required Arguments:
1. **--fingerprints (-fps)**: Paths to files with fingerprints of building blocks.
2. **--reaction_rules (-r)**: Path to the file with the reaction rules matrix.
3. **--hit_indexes (-hi)**: Path to the file with hit indexes.
4. **--train_indexes (-ti)**: Paths to files with train indexes.
5. **--save_path (-sp)**: Path to save the file with the model’s parameters.
6. **--seed (-s)**: Random seed for reproducible results (optional, not implemented).

The output of the script is the parameters of the trained model saved to the disk.

#### Example Command:
```bash
python train.py -fps bb_0.npy bb_1.npy bb_2.npy -r reaction_rules.npy -hi hits_idxs_q_0.6.npy -ti 0_batch_idxs.npy 1_batch_idxs.npy -sp 1_batch_model.pt
```
This command trains the model using the pairs with indexes from `0_batch_idxs.npy` and `1_batch_idxs.npy` and saves the model’s parameters as `1_batch_model.pt`.

#### Comments:
- **BinaryClassifierNN**: This class defines the model to train, which is a simple MLP implemented in PyTorch. It can be easily modified by changing the number of neurons and the number of hidden layers by modifying the `self.linear_relu_stack` parameter. If changes are made, the same changes should be made in the `predict.py` script.
- **PairsDataset**: This dataset object is built on top of the standard map-style PyTorch dataset class. It reads files with fingerprints, reaction rules, hit indexes, and training set indexes. The function of this class is to provide methods to retrieve fingerprints of a building block pair and the corresponding label.

#### Batch Size Parameters:
- **Batch size in PairsDataset class**: 
  This parameter was added to increase speed performance. When we provide training indexes to the dataset object, it will randomly shuffle them, then break them down into batches. It's better to set a small batch size. 

  Example:
  - Train indexes: 1, 64, 35, 132, 72, 25, 986, 12, 356, 10
  - With batch size = 3, they will be broken down as:
    - (1, 64, 35), (132, 72, 25), (986, 12, 356)
  - Elements that don’t fit into batches are omitted. 

  Elements in one batch are always retrieved together by the same index, which is suboptimal for training but allows for significant speed increases given the large training set size. Setting the batch size to 1 will retrieve every element one by one.

- **Batch size in Dataloader**: 
  The Dataloader is a default PyTorch dataloader class. The batch size parameter determines how many elements from the dataset will be retrieved simultaneously (the elements in the dataset are batches themselves). The final batch size retrieved to compute the gradient during training is `dataset batch size * dataloader batch size`. 

  Batches in the dataset object are fixed, but batches in the Dataloader are shuffled at every training epoch. The standard values are 8 for the dataset batch size and 256 for the Dataloader batch size, resulting in a training batch size of 8 * 256 = 2048. Given the large training set, these values optimize speed performance.

#### Other Model Hyperparameters:
- **Learning Rate and L2 Weight Regularization**: Set as optimizer parameters (`optim.Adam()`).
- **Training Epochs**: The model is trained for a maximum of 200 epochs, with early stopping if the loss hasn’t changed by more than 0.0001 for 10 epochs.






 
