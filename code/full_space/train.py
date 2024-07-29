"""
The script trains a pytorch MLP model and saves its parameters as .pt file

Example
-------
python train.py -fps bb_0.npy bb_1.npy bb_2.npy -r reaction_rules.npy -hi hits_idxs_q_0.6.npy -ti 0_batch_idxs.npy 1_batch_idxs.npy -sp ../result/1_batch_model.pt'
	Trains the model on indexes 0_batch_idxs.npy and 1_batch_idxs.npy and saves its parameters as ./result/1_batch_model.pt
"""

import argparse
import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

class BinaryClassifierNN(nn.Module):
	"""
	Class of the MLP model to train

	Atributes
	---------
	input_dim: int 
		dimentionality of the input, if each building block is encoded with 2048 bits and we have 11 reactions, input_dim = 2048*2 + 11 = 4107
	"""
	def __init__(self, input_dim: int):	
		super().__init__()

		self.linear_relu_stack = nn.Sequential(
			nn.Linear(input_dim, 100),
			nn.ReLU(),
			nn.Linear(100, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		return self.linear_relu_stack(x)

class PairsDataset(Dataset):
	"""
	Dataset class for retrieving fingerprints and labels by the index of a pair
	
	Atributes
	---------


	Parameters
	----------
	bb_fps: List[np.array]
		list of numpy array with building blocks' fingerprints
	rules: np.array
		np.array with reaction rules matrix
	hits_idxs: np.array 
		np.array with indexes of hit pairs
	train_idxs: np.array
		np.array with indexes of pairs to use durimg the training
	batch_size: int
		size of the batch to use while retriving the pairs
	"""
	def __init__(self, bb_fps, rules, hits_idxs, train_idxs, batch_size):

		reaction_sizes = np.array([len(bb_fps[rule[0]])*len(bb_fps[rule[1]]) for rule in rules])
		reaction_borders = np.cumsum(reaction_sizes) - 1
		bb_sizes = np.array([len(fps) for fps in bb_fps])
		bb_borders = np.cumsum(bb_sizes) - 1

		self.size = len(train_idxs)
		self.total_size =  reaction_sizes.sum()
		self.batch_size = batch_size
		self.num_full_batches = self.size//self.batch_size

		self.input_dim = 4096+len(rules)

		self.bb_fps = np.vstack([fps for fps in bb_fps])
		reaction_fps = np.identity(len(rules), dtype=bool)
		y = np.zeros(self.total_size, dtype=bool)
		y[hits_idxs] = True
		y = y[train_idxs]

		reaction_numbers = np.searchsorted(reaction_borders, train_idxs).astype(np.int8)
		train_idxs = train_idxs - reaction_borders[reaction_numbers] + reaction_sizes[reaction_numbers] - 1 #Translate to local indexes
		bb1_n, bb2_n = rules[:,0][reaction_numbers], rules[:,1][reaction_numbers]  #find types of building blocks
		bb1_idxs = train_idxs // bb_sizes[bb2_n]  # Local indexes of building blocks
		bb2_idxs = train_idxs % bb_sizes[bb2_n]

		bb1_idxs = bb1_idxs + bb_borders[bb1_n] - bb_sizes[bb1_n] + 1  #Global indexes of building blocks
		bb2_idxs = bb2_idxs + bb_borders[bb2_n] - bb_sizes[bb2_n] + 1

		self.bb1_idxs = bb1_idxs[:self.num_full_batches * self.batch_size].reshape(-1, self.batch_size)
		self.bb2_idxs = bb2_idxs[:self.num_full_batches * self.batch_size].reshape(-1, self.batch_size)
		self.reaction_fps = reaction_fps[reaction_numbers[:self.num_full_batches * self.batch_size]].reshape(-1, self.batch_size, len(rules))
		self.y = y[:self.num_full_batches * self.batch_size].reshape(-1, self.batch_size)

	def __len__(self):
		return self.size//self.batch_size

	def __getitem__(self, idx):
		output = np.empty(shape=(self.batch_size, self.input_dim), dtype=bool)
		fp1, fp2 = self.bb_fps[self.bb1_idxs[idx]], self.bb_fps[self.bb2_idxs[idx]]
		np.bitwise_and(fp1, fp2, out=output[:,:2048])
		np.bitwise_xor(fp1, fp2, out=output[:,2048:4096])
		output[:,4096:] = self.reaction_fps[idx]

		return output, self.y[idx]

def train(args):

	device = ("cuda" if torch.cuda.is_available() else "cpu")
	print(f'Using {device} device')

	dataset = PairsDataset(
		bb_fps = [np.load(fps).astype(bool) for fps in args.fingerprints], 
		rules = np.load(args.reaction_rules), 
		hits_idxs = np.load(args.hit_indexes),
		train_idxs = np.hstack([np.load(batch) for batch in args.train_indexes]),
		batch_size = 8
	) 
	print(f"Training set size: {dataset.size}, hits: {dataset.y.sum()}")

	dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)

	model = BinaryClassifierNN(dataset.input_dim).to(device)
	loss_fn = nn.BCELoss()
	optimizer = optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
	model.train()

	trainingEpoch_loss = np.array([])
	min_loss = 100
	no_change = 0

	for epoch in range(200):
		print(f"Epoch {epoch+1}\n-------------------------------")
		step_loss = []
		for (X, y) in tqdm(dataloader, unit='batch'):
			X, y = X.to(device), y.to(device)
			X, y = X.to(dtype=torch.float), y.to(dtype=torch.float)
			pred = model(X)
			loss = loss_fn(pred, y.view(pred.shape))
			loss.backward()
			optimizer.step()
			optimizer.zero_grad()
			step_loss.append(loss.item())
		print(f'Train loss: {np.array(step_loss).mean()}')
		if epoch > 0:
			if np.array(step_loss).mean() - min_loss <= -0.0001:
				no_change = 0
				min_loss = np.array(step_loss).mean()
			else:
				no_change += 1
			if np.array(step_loss).mean() < trainingEpoch_loss.min(): 
				best_param = model.state_dict()
				torch.save(best_param, args.save_path)
			
		trainingEpoch_loss = np.append(trainingEpoch_loss, np.array(step_loss).mean())

		if no_change >= 10:
			break



if __name__=='__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-fps', '--fingerprints', type=str, nargs='+', required=True,
						help='List of paths to .npy files containing fingerprints of building blocks')

	parser.add_argument('-r', '--reaction_rules', type=str, required=True,
						help='Path to .npy file containing reaction rules matrix')

	parser.add_argument('-hi', '--hit_indexes', type=str, required=True,
						help='Paths to .npy files with indexes of hit pairs')

	parser.add_argument('-ti', '--train_indexes', type=str, nargs='+', required=True,
						help='Paths to .npy files with indexes of training subset')

	parser.add_argument('-sp', '--save_path', type=str, required=True,
						help='Path to save the model as .pt file')

	parser.add_argument('-s', '--seed', type=int, required=False, default=None,
						help='Random seed to select the subset')
	args = parser.parse_args()

	train(args)