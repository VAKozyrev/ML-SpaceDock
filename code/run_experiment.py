import time
from pathlib import Path
import argparse

import numpy as np
from scipy import sparse as ss

from acquirer import Acquirer
from estimator import CommonEstimator

def run_experiment(args):

	save_dir = Path(args.save_dir)
	save_dir.mkdir(exist_ok=True)

	training_time = list()
	predicting_time = list()
	acquisition_time = list()

	X = ss.load_npz(args.fingerprints)
	y = np.load(args.labels)

	acquirer = Acquirer(y, args)
	estimator = CommonEstimator(args)	

	start = time.time()
	acquirer.acquire_initial() 
	acquisition_time.append(time.time() - start)

	for _ in range(len(acquirer.batch_sizes)-1):

		start = time.time()
		estimator.fit(X[acquirer.explored_idxs], y[acquirer.explored_idxs])
		training_time.append(time.time() - start)

		start = time.time()
		y_mean, y_var = estimator.predict(X)
		predicting_time.append(time.time() - start)

		start = time.time()
		acquirer.acquire_batch(y_mean, y_var)
		acquisition_time.append(time.time() - start)

	np.save(Path(save_dir, 'training_time.npy'), np.array(training_time))
	np.save(Path(save_dir, 'predicting_time.npy'), np.array(predicting_time))
	np.save(Path(save_dir, 'acquisition_time.npy'), np.array(acquisition_time))

if __name__ == '__main__':

	parser = argparse.ArgumentParser()

	parser.add_argument('-f', '--fingerprints', type=str, required=True,
		help='Path to the .npz sparse matrix with fingerprints')

	parser.add_argument('-l', '--labels', type=str, required=True,
		help='Path to the .npy numpy array with the training labels of reagent pairs')

	parser.add_argument('-sd', '--save_dir', type=str, required=True,
		help="Path to the folder to save the results, if doesn't exist, create")

	parser.add_argument('-m', '--model', type=str, required=False, default='logreg',
			help='Machine learning model to apply (logreg, rf, mlp, mlp_pytorch)')

	parser.add_argument('-a', '--acquisition_function', type=str, required=False, default='greedy',
			help='Accusition function to use for acquiring of new batches (greedy, thompson, sd_deviation)')

	parser.add_argument('-th', '--train_horizon', type=float, required=False, default=10,
			help='Up to what part of the dataset explore')

	parser.add_argument('-ib', '--initial_batch', type=float, required=False, default=0.1,
			help='Size of the first batch as a portion of the whole dataset')

	parser.add_argument('-eb', '--exploration_batch', type=float, required=False, default=0.5,
			help='Size of every exploration batch as portion of the whole dataset')

	parser.add_argument('-nf', '--n_folds', type=int, required=False, default=3,
			help="Number of folds to quantify uncertainty")

	parser.add_argument('-s', '--seed', type=int, required=False, default=1,
			help='Random seed for the experiment')

	args = parser.parse_args()

	run_experiment(args)