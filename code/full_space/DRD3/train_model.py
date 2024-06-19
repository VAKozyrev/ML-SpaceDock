import argparse
import pickle
import time

import numpy as np 
from scipy import sparse as ss
from sklearn.neural_network import MLPClassifier


def get_fps(amines_fps, acids_fps, idxs):

	fps = ss.csr_matrix((0,4096))
	for batch in np.array_split(idxs, 50):

		idxs_amines = batch // len(acids_fps)
		idxs_acids = batch % len(acids_fps)

		fp_amine = amines_fps[idxs_amines]
		fp_acid = acids_fps[idxs_acids]

		fp = np.hstack([np.logical_and(fp_amine, fp_acid), np.logical_xor(fp_amine, fp_acid)])

		fps = ss.vstack([fps, ss.csr_matrix(fp)])

	return fps

def get_labels(training_idxs, hits_idxs):
	mask = np.isin(training_idxs, hits_idxs, invert=True)
	training_idxs[mask] = 0
	training_idxs[~mask] = 1
	return training_idxs


def train_model(args):

	mlp = MLPClassifier()

	amines_fp_path = '/scratch/vkozyrev/ML-Spacedock/data/amines.npy'
	acids_fp_path = '/scratch/vkozyrev/ML-Spacedock/data/acids.npy'
	training_idxs_paths = [f'../{i}_batch_idxs.npy' for i in range(0, args.batch_number+1, 1)]

	start = time.time()
	fps = get_fps(
		amines_fps = np.load(amines_fp_path), 
		acids_fps = np.load(acids_fp_path), 
		idxs = np.hstack([np.load(file) for file in training_idxs_paths])
		)
	print(f'Fingerprints retrieved in {time.time()-start} seconds')

	start = time.time()
	labels = get_labels(
		training_idxs = np.hstack([np.load(file) for file in training_idxs_paths]), 
		hits_idxs = np.load('/scratch/vkozyrev/ML-Spacedock/data/hits_idxs.npy')
		)
	print(f'Labels retrieved in {time.time()-start} seconds')

	print(f"Starting training the model\nTraining batch: {len(labels)} pairs, '0': {np.count_nonzero(labels == 0)} '1': {np.count_nonzero(labels == 1)}")
	start=time.time()
	mlp.fit(fps, labels)
	print(f'Training took {time.time()-start} seconds')

	filename = f'../mlp_batch_{args.batch_number}.model'
	pickle.dump(mlp, open(filename, 'wb'))

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--batch_number', type=int, required=True, default=0,
					help='Batch number')
	args = parser.parse_args()

	train_model(args)

	






