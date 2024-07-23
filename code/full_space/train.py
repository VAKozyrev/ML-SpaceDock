import argparse
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm

class BinaryClassifierNN(nn.Module):

	def __init__(self, input_dim):	
		super().__init__()

		self.linear_relu_stack = nn.Sequential(
			nn.Linear(input_dim, 200),
			nn.ReLU(),
			nn.Linear(200, 200),
			nn.ReLU(),
			nn.Linear(200, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		return self.linear_relu_stack(x)

class PairsDataset(Dataset):

	def __init__(self, bb_fps, rules, hits_idxs, train_idxs, batch_size):

		reaction_sizes = np.array([len(bb_fps[rule[0]])*len(bb_fps[rule[1]]) for rule in rules])
		reaction_borders = np.cumsum(reaction_sizes) - 1
		bb_sizes = np.array([len(fps) for fps in bb_fps])
		bb_borders = np.cumsum(bb_sizes) - 1

		self.size = len(train_idxs)
		self.total_size =  reaction_sizes.sum()
		self.batch_size = batch_size
		self.num_full_batches = self.size//self.batch_size

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
		self.reaction_fps = reaction_fps[reaction_numbers[:self.num_full_batches * self.batch_size]].reshape(-1, self.batch_size, 11)
		self.y = y[:self.num_full_batches * self.batch_size].reshape(-1, self.batch_size)

	def __len__(self):
		return self.size//self.batch_size

	def __getitem__(self, idx):
		output = np.empty(shape=(self.batch_size, 4107), dtype=bool)
		fp1, fp2 = self.bb_fps[self.bb1_idxs[idx]], self.bb_fps[self.bb2_idxs[idx]]
		np.bitwise_and(fp1, fp2, out=output[:,:2048])
		np.bitwise_xor(fp1, fp2, out=output[:,2048:4096])
		output[:,4096:] = self.reaction_fps[idx]

		return output, self.y[idx]

def train(args):

	device = ("cuda" if torch.cuda.is_available() else "cpu")
	print(f'Using {device} device')

	dataset = PairsDataset(
		bb_fps = [np.load(f'../data/CBLB/bb_{i}.npy').astype(bool) for i in range(21)], 
		rules = np.load('../data/CBLB/reactions_rules.npy'), 
		hits_idxs = np.load('../data/CBLB/hits_idxs_q_0.6.npy'),
		train_idxs = np.hstack([np.load(f'{i}_batch_idxs.npy') for i in range(args.batch_n+1)]),
		batch_size = 8
	) 
	print(f"Training set size: {dataset.size}, hits: {dataset.y.sum()}")

	dataloader = DataLoader(dataset, batch_size=256, shuffle=True, num_workers=8, pin_memory=True)

	model = BinaryClassifierNN(4107).to(device)
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
				torch.save(best_param, f'model_batch_{args.batch_n}.pt')
			
		trainingEpoch_loss = np.append(trainingEpoch_loss, np.array(step_loss).mean())

		if no_change >= 10:
			break



if __name__=='__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--batch_n', type=int, required=True, default=0,
						help='Number of the batch')
	args = parser.parse_args()

	train(args)