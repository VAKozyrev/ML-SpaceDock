import numpy as np
import torch
import torch.nn as nn
from torch import optim
from torch.utils.data import Dataset, DataLoader, Subset
from sklearn.metrics import recall_score, balanced_accuracy_score, f1_score, auc, precision_recall_curve
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from scipy import sparse as ss

device = ("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")

class PairsDataset(Dataset):

	def __init__(self, X, y):

		self.X = torch.Tensor(X.todense())
		self.y = torch.Tensor(y)

	def __len__(self):
		return len(self.X)

	def __getitem__(self, idx):
		return self.X[idx], self.y[idx]


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

	def fit(self, X, y):

		dataset = PairsDataset(X, y)
		# train_idxs, val_idxs = train_test_split(range(X.shape[0]), test_size=0.2)
		# train_subset = Subset(dataset, train_idxs)
		# val_subset = Subset(dataset, val_idxs)

		# train_dataloader = DataLoader(train_subset, batch_size=200, shuffle=True, num_workers=8, pin_memory=True)
		# val_dataloader = DataLoader(val_subset, batch_size=200, num_workers=8)
		dataloader = DataLoader(dataset, batch_size=200, shuffle=True, num_workers=8, pin_memory=True)

		loss_fn = nn.BCELoss()
		optimizer = optim.Adam(self.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=1e-4)
		#scheduler = torch.optim.lr_scheduler.ExponentialLR(optimizer, gamma = 0.95)

		self.train()
		trainingEpoch_loss = np.array([])
		validationEpoch_loss = np.array([])

		no_change = 0
		best_param = self.state_dict()

		for epoch in range(200):
			print(f"Epoch {epoch+1}\n-------------------------------")
			step_loss = []

			for (X, y) in tqdm(dataloader, unit='batch'):

				X, y = X.to(device, dtype=torch.float), y.to(device, dtype=torch.float)

				pred = self(X)
				loss = loss_fn(pred, y.unsqueeze(1))

				loss.backward()
				optimizer.step()
				optimizer.zero_grad()

				step_loss.append(loss.item())
			
			print(f'Train loss: {np.array(step_loss).mean()}')

			# with torch.no_grad():
			# 	step_loss = []
			# 	for (X, y) in tqdm(val_dataloader, desc='Predicting probabilities for validation set', unit='batch'):

			# 		X, y = X.to(device, dtype=torch.float), y.to(device, dtype=torch.float)

			# 		pred = self(X)

			# 		loss = loss_fn(pred, y.unsqueeze(1))

			# 		step_loss.append(loss.item())

			# 	if epoch > 0 and np.array(step_loss).mean() < validationEpoch_loss.min():
			# 		best_param = self.state_dict()
			# 		no_change = 0
			# 	else:
			# 		no_change += 1

			# validationEpoch_loss = np.append(validationEpoch_loss, np.array(step_loss).mean())
			# print(f'Validation loss: {np.array(step_loss).mean()}')

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
		proba = self(torch.Tensor(X.todense()).to(device))
		return proba.detach().to('cpu').numpy().reshape((1000000,))




	

	



	