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
### 0. Requirements:

1. **File with Building Blocks**:
   - Contains Enamine IDs, SMILES, and types of corresponding reactions.
   - Path: `/data/CBLB/building_blocks.tsv.gz`

2. **File with Hits in the Chemical Space**:
   - Output of SpaceDock.
   - Contains Enamine IDs of building blocks comprising the hit pairs, corresponding reactions, full IFP, and polar IFP similarities.
   - Path: `/data/CBLB/hits.tsv.gz`

### I. Preparation of Inputs:
To explore large chemical spaces involving multiple chemical reactions, we need the following inputs:

1. **Fingerprints of the Building Blocks** saved as `.npy` files.
2. **Reaction Rules Matrix** saved as a `.npy` file.
3. **Indexes of Hit Pairs** saved as `.npy` files, for different IFP similarity thresholds.

All these files can be generated by running the `CBLB_descriptor_generation_and_labeling.ipynb` notebook. Follow the comments in the notebook and modify the cells where necessary. An additional example of generating inputs for a smaller subset of the CBLB chemical space (2 reactions) can be found in `example.ipynb`.

### II. Active Learning Workflow
After generating all required files, you can run the active learning workflow, which consists of four Python scripts:

1. **select_0_batch.py**:
   - This script should be run only once to select the first random batch of pairs to start the exploration.
2. **train.py**:
   - This script trains the MLP classifier model.
3. **predict.py**:
   - This script uses the trained model to make predictions.
4. **select_batch.py**:
   - This script selects the next batch of pairs for exploration based on the predictions.

After running `select_0_batch.py` to select the initial batch, scripts `train.py`, `predict.py`, and `select_batch.py` are run iteratively until the desired portion of the chemical space is explored.

### II.1 `Select_0_batch.py`

The `select_0_batch.py` script selects random indexes within the range from 0 to the size of the chemical space. These indexes will serve as the training set for training the model in the next step.

#### Required Arguments:
1. **--fingerprints (-fps)**: Paths to files with fingerprints of building blocks.
2. **--reaction_rules (-r)**: Path to the file with the reaction rules matrix.
3. **--batch_size (-bs)**: Size of the batch you want to acquire.
4. **--save_path (-sp)**: Path to save the file with selected indexes.
5. **--seed (-s)**: Random seed for reproducible results (optional).

The output of the script is a numpy array containing selected indexes saved to the disk.

#### Example Command:
```bash
python select_0_batch.py -fps bb_0.npy bb_1.npy bb_2.npy -r reaction_rules.npy -bs 10000000 -sp 0_batch_idxs.npy -s 1
```
This command selects 10,000,000 random indexes from the chemical space formed by building blocks with fingerprints `bb_0.npy`, `bb_1.npy`, and `bb_2.npy` and chemical reactions encoded in `reaction_rules.npy`, and saves the result as `0_batch_idxs.npy`.

### II.2 `Train.py`

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
- **PairsDataset**: This dataset object is built on top of the standard map-style PyTorch dataset class. The function of this class is to provide methods to retrieve fingerprints of a building block pair and the corresponding label.
- **Batch size in PairsDataset class**: This parameter was added to increase speed performance. When we provide training indexes to the dataset object, it will randomly shuffle them, then break them down into batches. It's better to set a small batch size.
  
  Example:
  
  Train indexes: 1, 64, 35, 132, 72, 25, 986, 12, 356, 10 with batch size = 3, they will be broken down as:
    - (1, 64, 35), (132, 72, 25), (986, 12, 356). Elements that don’t fit into batches are omitted. 

  Elements in one batch are always retrieved together by the same index, which is suboptimal for training but allows for significant speed increases given the large training set size. Setting the batch size to 1 will retrieve every element one by one.

- **Batch size in Dataloader**: 
  The Dataloader is a default PyTorch dataloader class. The batch size parameter determines how many elements from the dataset will be retrieved simultaneously (the elements in the dataset are batches themselves). The final batch size retrieved to compute the gradient during training is `dataset batch size * dataloader batch size`. 

  Batches in the dataset object are fixed, but batches in the Dataloader are shuffled at every training epoch. The standard values are 8 for the dataset batch size and 256 for the Dataloader batch size, resulting in a training batch size of 8 * 256 = 2048. Given the large training set, these values optimize speed performance.

#### Other Model Hyperparameters:
- **Learning Rate and L2 Weight Regularization**: Set as optimizer parameters (`optim.Adam()`).
- **Training Epochs**: The model is trained for a maximum of 200 epochs, with early stopping if the loss hasn’t changed by more than 0.0001 for 10 epochs.

### II.3 `Predict.py`

The `predict.py` script makes predictions for the entire chemical space and saves the results as numpy arrays with predicted values in the specified directory.

#### Required Arguments:
1. **--fingerprints (-fps)**: Paths to files with fingerprints of building blocks.
2. **--reaction_rules (-r)**: Path to the file with the reaction rules matrix.
3. **--model (-m)**: Path to the file containing the parameters of the model to load.
4. **--save_dir (-sd)**: Path to the directory to save the files with predictions. If the directory doesn’t exist, it will be created.

The output of the script is numpy arrays with predicted values saved in the specified directory.

#### Example Command:
```bash
python predict.py -fps bb_0.npy bb_1.npy bb_2.npy -r reaction_rules.npy -m 0_batch_model.pt -sd ../example/preds/
```
This command makes predictions for the chemical space formed by building blocks with fingerprints `bb_0.npy`, `bb_1.npy`, and `bb_2.npy`, and chemical reactions encoded in `reaction_rules.npy`. It uses the model parameters from `trained_model.pt` and saves the predictions in the directory `../example/preds/`. The predictions will be saved as `.npy` files named `{i}_preds.npy` with i from 0 to number of batches (depends on the size of the chemical space), one batch is ~8M predictions.

### II.4 `Select_batch_heap.py`

The `select_batch_heap.py` script selects the next batch of indexes from the model predictions.

#### Required Arguments:
1. **--size (-s)**: Size of the batch to acquire.
2. **--explored_indexes (-ei)**: Paths to the files with already explored indexes.
3. **--path (-p)**: Path to the directory with `preds.npy` files.
4. **--save_path (-sp)**: Path to save the numpy array with selected indexes.

The output of the script is a numpy array containing the selected indexes, saved to the specified path.

#### Example Command:
```bash
python select_batch_heap.py -s 1000000 -ei 0_batch_idxs.npy 1_batch_idxs.npy -p ../example/preds/ -sp 2_batch_idxs.npy
```
This command selects 1,000,000 indexes according to predictions stored in `preds.npy` files in `../example/preds/`, excluding already selected indexes from `0_batch_idxs.npy` and `1_batch_idxs.npy`, and saves them as `2_batch_idxs.npy`.

### `Predict_cpu_batched.py`

The `predict_cpu_batched.py` script is a batched version of `predict.py` that runs without using a GPU, designed for parallelization.

#### Required Arguments:
1. **--fingerprints (-fps)**: Paths to files with fingerprints of building blocks.
2. **--reaction_rules (-r)**: Path to the file with the reaction rules matrix.
3. **--model (-m)**: Path to the file containing the parameters of the model to load.
4. **--start (-st)**: Index of the building block pair to start prediction from.
5. **--end (-ed)**: Index of the building block pair to end prediction on (last index not included).

The output of the script is predictions for the specified range of building block pairs.

#### Example Command:
```bash
python predict_cpu_batched.py -fps bb_0.npy bb_1.npy bb_2.npy -r reaction_rules.npy -m 0_batch_model.pt -st 10000000 -ed 110000000
```
This command makes predictions for pairs with indexes from 100,000,000 to 109,999,999 using the model saved in `0_batch_model.pt` and saves the predictions as 100000000_110000000_preds.npy

### III. Results exploration

The results of the active learning workflow are the indexes retrieved on each iteration `0_batch_idxs.npy` `1_batch_idxs.npy` ... 
The results of the experiment can be explored in `results_full_space.ipynb` notebook
