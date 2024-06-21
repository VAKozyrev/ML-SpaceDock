from pathlib import Path

import numpy as np

class Acquirer: 

	def __init__(self, y, args):

		self.size = len(y)
		self.train_horizon = args.train_horizon
		self.initial_batch = args.initial_batch
		self.exploration_batch = args.exploration_batch
		self.batch_sizes = self.get_batch_sizes()

		self.acquisition_function = args.acquisition_function
		self.metric = self.get_metric(self.acquisition_function)

		self.rng = np.random.default_rng(args.seed)

		self.explored_idxs = np.array([])
		self.batch_n = 0

		self.save_dir = args.save_dir

		print(f'Acquirer is initialized: {self}')

	def __str__(self):
		return f"Acquirer object(\n\
		size: {self.size}\n\
		train horizon: {self.train_horizon}%\n\
		initial batch: {self.initial_batch}%\n\
		exploration_batch: {self.exploration_batch}%\n\
		batch sizes: {self.batch_sizes}\n\
		acquisition_function: {self.acquisition_function}\n\
		)"

	def get_batch_sizes(self):
		batch_sizes=list()
		initial_batch = int(self.size * self.initial_batch / 100)
		exploration_batch = int(self.size * self.exploration_batch / 100)
		remaining = int(self.size * self.train_horizon / 100)

		while remaining>0:
			if len(batch_sizes) == 0:
				batch_sizes.append(initial_batch)
				remaining -= initial_batch
			else:
				batch_sizes.append(min(exploration_batch,remaining))
				remaining -= exploration_batch
		return batch_sizes

	def get_metric(self, acquisition_function):
		return {'greedy': self.greedy,
				'thompson': self.thompson,
				'sd_deviation': self.sd_deviation
				}[self.acquisition_function]

	def acquire_initial(self):
		print(f'Acquiring batch #{self.batch_n}')
		size = self.batch_sizes[self.batch_n]

		selected_idxs = self.random()[:size]
		self.explored_idxs = selected_idxs

		np.save(Path(self.save_dir, f'{self.batch_n}_batch_idxs.npy'), selected_idxs)

		self.batch_n += 1

	def acquire_batch(self, y_mean, y_var):
		print(f'Acquiring batch #{self.batch_n}')
		size = self.batch_sizes[self.batch_n]

		sorted_idxs = self.metric(y_mean, y_var)
		mask = np.isin(sorted_idxs, self.explored_idxs, invert=True)
		selected_idxs = sorted_idxs[mask][:size]
		self.explored_idxs = np.hstack([self.explored_idxs, selected_idxs])

		np.save(Path(self.save_dir, f'{self.batch_n}_batch_idxs.npy'), selected_idxs)

		self.batch_n += 1

	def random(self):
		return self.rng.choice(self.size, self.size, replace=False)
	def greedy(self, y_mean, y_var):
		return np.argsort(y_mean)[::-1]
	def thompson(self, y_mean, y_var):
		y_sd = np.sqrt(y_var)
		return np.argsort(self.rng.normal(y_mean, y_sd))[::-1]
	def sd_deviation(self, y_mean, y_var):
		y_sd = np.sqrt(y_var)
		return np.argsort(y_sd)[::-1]