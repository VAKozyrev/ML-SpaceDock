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

	def __init__(self, bb_fps, rules):

		self.fps = np.vstack([fps for fps in bb_fps])
		self.reaction_sizes = np.array([len(bb_fps[rule[0]])*len(bb_fps[rule[1]]) for rule in rules])
		self.reaction_borders = np.cumsum(self.reaction_sizes) - 1
		self.bb_sizes = np.array([len(fps) for fps in bb_fps])
		self.bb_borders = np.cumsum(self.bb_sizes) - 1
		self.bbs_1 = rules[:,0]
		self.bbs_2 = rules[:,1]
		self.idx_global_to_local = self.reaction_borders - self.reaction_sizes + 1
		self.size = self.reaction_sizes.sum()
		self.reaction_fps = np.identity(len(rules), dtype=bool)

	def __len__(self):
		return self.size

	def __getitem__(self, idx):
		output = np.empty(4107, dtype=bool)
		# Reaction number
		reaction_n  = np.searchsorted(self.reaction_borders, idx)
		#Local index
		idx = idx - self.idx_global_to_local[reaction_n]
		# Building blocks' number
		bb1_n, bb2_n = self.bbs_1[reaction_n], self.bbs_2[reaction_n]

		#Indexes of first and second building blocks
		idx_bb_1 = idx // self.bb_sizes[bb2_n] + self.bb_borders[bb1_n] - self.bb_sizes[bb1_n] + 1
		idx_bb_2 = idx % self.bb_sizes[bb2_n] + self.bb_borders[bb2_n] - self.bb_sizes[bb2_n] + 1

		fp1 = self.fps[idx_bb_1]
		fp2 = self.fps[idx_bb_2]

		np.bitwise_and(fp1, fp2, out=output[:2048])
		np.bitwise_xor(fp1, fp2, out=output[2048:4096])
		output[4096:] = self.reaction_fps[reaction_n]

		return output

def train(args):

	device = ("cuda" if torch.cuda.is_available() else "cpu")
	print(f'Using {device} device')

	dataset = PairsDataset(
		bb_fps = [np.load(f'../data/CBLB/bb_{i}.npy').astype(bool) for i in range(21)], 
		rules = np.load('../data/CBLB/reactions_rules.npy')
	) 

	dataloader = DataLoader(dataset, batch_size=2048, shuffle=False, num_workers=8, pin_memory=True)

	model = BinaryClassifierNN(4107).to(device)
	model.load_state_dict(torch.load(f'model_batch_{args.batch_n}.pt'))

	pred = np.array([])
	count = 0
	with torch.no_grad():
		for i, X in enumerate(tqdm(dataloader, unit='batch')):
			X = X.to(device)
			X = X.to(dtype=torch.float) 
			pred = np.hstack([pred, model(X).squeeze().to('cpu').numpy()])
			#print(len(np.unique(model(X).squeeze().to('cpu').numpy())))
			if i!=0 and i % 2000 == 0:
				np.save(f'preds/{count}_preds.npy', pred)
				pred = np.array([])
				count += 1
		np.save(f'preds/{count}_preds.npy', pred)

			
if __name__=='__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--batch_n', type=int, required=True, default=0,
						help='Number of the batch')
	args = parser.parse_args()

	train(args)