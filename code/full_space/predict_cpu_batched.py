import argparse
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
from tqdm import tqdm
from pathlib import Path

class BinaryClassifierNN(nn.Module):

	def __init__(self, input_dim):	
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

	def __init__(self, bb_fps, rules, start, end):

		reaction_sizes = np.array([len(bb_fps[rule[0]])*len(bb_fps[rule[1]]) for rule in rules])
		reaction_borders = np.cumsum(reaction_sizes) - 1
		bb_sizes = np.array([len(fps) for fps in bb_fps])
		bb_borders = np.cumsum(bb_sizes) - 1

		self.input_dim = 4096+len(rules)

		self.size = end - start
		self.total_size =  reaction_sizes.sum()

		self.bb_fps = np.vstack([fps for fps in bb_fps])
		reaction_fps = np.identity(len(rules), dtype=bool)

		train_idxs = np.arange(start, end)

		reaction_numbers = np.searchsorted(reaction_borders, train_idxs).astype(np.int8)
		train_idxs = train_idxs - reaction_borders[reaction_numbers] + reaction_sizes[reaction_numbers] - 1 #Translate to local indexes
		bb1_n, bb2_n = rules[:,0][reaction_numbers], rules[:,1][reaction_numbers]  #find types of building blocks
		bb1_idxs = train_idxs // bb_sizes[bb2_n]  # Local indexes of building blocks
		bb2_idxs = train_idxs % bb_sizes[bb2_n]

		bb1_idxs = bb1_idxs + bb_borders[bb1_n] - bb_sizes[bb1_n] + 1  #Global indexes of building blocks
		bb2_idxs = bb2_idxs + bb_borders[bb2_n] - bb_sizes[bb2_n] + 1

		self.bb1_idxs = bb1_idxs
		self.bb2_idxs = bb2_idxs
		self.reaction_fps = reaction_fps[reaction_numbers]

	def __len__(self):
		return self.size

	def __getitem__(self, idx):
		output = np.empty(self.input_dim, dtype=bool)
		fp1, fp2 = self.bb_fps[self.bb1_idxs[idx]], self.bb_fps[self.bb2_idxs[idx]]
		np.bitwise_and(fp1, fp2, out=output[:2048])
		np.bitwise_xor(fp1, fp2, out=output[2048:4096])
		output[4096:] = self.reaction_fps[idx]

		return output

def train(args):

	dataset = PairsDataset(
		bb_fps = [np.load(fps).astype(bool) for fps in args.fingerprints], 
		rules = np.load(args.reaction_rules),
		start = args.start,
		end = args.end
	) 

	dataloader = DataLoader(dataset, batch_size=4096, shuffle=False, num_workers=8)

	model = BinaryClassifierNN(dataset.input_dim)
	model.load_state_dict(torch.load(args.model, map_location='cpu'))

	pred = np.empty(args.end-args.start)

	with torch.no_grad():
		for i, X in enumerate(tqdm(dataloader, unit='batch')):
			X = X.to(dtype=torch.float) 
			pred[i*4096:i*4096+4096] = model(X).squeeze().numpy()

		np.save(f'{args.start}_{args.end}_preds.npy', pred)

			
if __name__=='__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('-fps', '--fingerprints', type=str, nargs='+', required=True,
						help='List of paths to .npy files containing fingerprints of building blocks')

	parser.add_argument('-r', '--reaction_rules', type=str, required=True,
						help='Path to .npy file containing reaction rules matrix')

	parser.add_argument('-m', '--model', type=str, required=True,
						help='Path to the model state dictionary')

	parser.add_argument('-st', '--start', type=int, required=True,
		help='Index of a building block pair from which script should start the prediction')

	parser.add_argument('-ed', '--end', type=int, required=True,
		help='Index of a building block pair from which script should end the prediction')

	args = parser.parse_args()

	train(args)