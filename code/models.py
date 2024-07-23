from tqdm import tqdm
import numpy as np
from scipy import sparse as ss
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import recall_score, balanced_accuracy_score, f1_score, auc, precision_recall_curve
from sklearn.model_selection import train_test_split

class PairsDataset(Dataset):
	def __init__(self, X, y):
		self.X = torch.tensor(X.todense(), dtype=torch.bool)
		self.y = torch.tensor(y, dtype=torch.bool)

	def __len__(self):
		return len(self.y)

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]

class BinaryClassifierNN(nn.Module):

	def __init__(self, input_dim, random_state=1):	
		super().__init__()
		torch.manual_seed(random_state)
		self.linear_relu_stack = nn.Sequential(
			nn.Linear(input_dim, 100),
			nn.ReLU(),
			nn.Linear(100, 1),
			nn.Sigmoid()
		)

	def forward(self, x):
		return self.linear_relu_stack(x)

	def fit(self, X, y):
		device = ("cuda" if torch.cuda.is_available() else "cpu")
		print(f"Using {device} device")
		self.to(device)

		dataset = PairsDataset(X, y)
		dataloader = DataLoader(dataset, batch_size=200, shuffle=True, num_workers=8, pin_memory=True)

		loss_fn = nn.BCELoss()
		optimizer = optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)

		self.train()
		trainingEpoch_loss = np.array([])
		no_change = 0
		best_param = self.state_dict()

		for epoch in range(200):
			print(f"Epoch {epoch+1}\n-------------------------------")
			step_loss = []

			for (X, y) in tqdm(dataloader, unit='batch'):

				X, y = X.to(device), y.to(device) 
				X, y = X.to(dtype=torch.float), y.to(dtype=torch.float)
				pred = self(X)
				loss = loss_fn(pred, y.unsqueeze(1))
				loss.backward()
				optimizer.step()
				optimizer.zero_grad()
				step_loss.append(loss.item())
			
			print(f'Train loss: {np.array(step_loss).mean()}')

			if epoch > 0 and np.array(step_loss).mean() - trainingEpoch_loss.min() <= -0.0001:
				best_param = self.state_dict()
				no_change = 0
			else:
				no_change += 1
			trainingEpoch_loss = np.append(trainingEpoch_loss, np.array(step_loss).mean())
			if no_change >= 10:
				break
		self.load_state_dict(best_param)

		return self

	def predict_proba(self, X):
		device = ("cuda" if torch.cuda.is_available() else "cpu")
		with torch.no_grad():
			X = torch.tensor(X.todense()).to(device)
			X = X.to(torch.float)
			proba = self(X).to('cpu')
		return proba.numpy().reshape((1000000,))




	

	



	