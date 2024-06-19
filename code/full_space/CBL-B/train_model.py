#!/sps/lit/vkozyrev/miniconda3/envs/ML-Spacedock/bin/python3.10
import argparse
import pickle
import time

from sklearn.neural_network import MLPClassifier
import numpy as np 
from scipy import sparse as ss
from tqdm import tqdm

def get_fps(bbs_fps, idxs, reactions_rules):

	fps = ss.csr_matrix((0,4107))
	reactions_fps = np.identity(len(reactions_rules), dtype=np.int8)

	# number of pairs for each reaction:
	pairs_per_reaction = np.array([len(bbs_fps[reaction[0]])*len(bbs_fps[reaction[1]]) for reaction in reactions_rules])
	# indexes of the last pair for each reaction:
	reaction_borders = np.cumsum(pairs_per_reaction) - 1

	for batch in tqdm(np.array_split(idxs, 100), desc='Getting fingerprints', unit='batch'):
		# Reaction numbers
		reaction_n  = np.searchsorted(reaction_borders, batch)
		#Local indexes
		idxs_local = batch - reaction_borders[reaction_n] + pairs_per_reaction[reaction_n] - 1
		# Building blocks' numbers
		bb1_n, bb2_n = reactions_rules[reaction_n][:,0], reactions_rules[reaction_n][:,1]
		# Fps corresponding to this building blocks' numbers
		fps_1, fps_2 = bbs_fps[bb1_n], bbs_fps[bb2_n]

		#Indexes of first and second building blocks
		bb_1_idxs = idxs_local // np.vectorize(lambda x: len(x))(fps_2)
		bb_2_idxs = idxs_local % np.vectorize(lambda x: len(x))(fps_2)

		bb_1_fps = np.array([arr[idx] for arr, idx in zip(fps_1, bb_1_idxs)])
		bb_2_fps = np.array([arr[idx] for arr, idx in zip(fps_2, bb_2_idxs)])

		fp = np.hstack([np.logical_and(bb_1_fps, bb_2_fps), np.logical_xor(bb_1_fps, bb_2_fps), reactions_fps[reaction_n]])

		fps = ss.vstack([fps, ss.csr_matrix(fp)])

	return fps

def get_labels(training_idxs, hits_idxs):
	mask = np.isin(training_idxs, hits_idxs, invert=True)
	training_idxs[mask] = 0
	training_idxs[~mask] = 1
	return training_idxs

def train_model(args):

	mlp = MLPClassifier()

	bbs_fps_paths = [f'bb_{i}.npy' for i in range(21)]
	training_idxs_paths = [f'{i}_batch_idxs.npy' for i in range(0, args.batch_number+1, 1)]

	start = time.time()
	fps = get_fps(
	bbs_fps = np.array([np.load(path) for path in bbs_fps_paths], dtype=object),
	idxs = np.hstack([np.load(path) for path in training_idxs_paths]),
	reactions_rules = np.load('reactions_rules.npy')
	)
	print(f'Fingerprints retrieved in {time.time()-start} seconds')

	start = time.time()
	labels = get_labels(
	training_idxs = np.hstack([np.load(path) for path in training_idxs_paths]), 
	hits_idxs = np.load('hits_idxs.npy')
	)
	print(f'Labels retrieved in {time.time()-start} seconds')

	start=time.time()
	print(f"Starting training the model\nTraining batch: {len(labels)} pairs, '0': {np.count_nonzero(labels == 0)} '1': {np.count_nonzero(labels == 1)}")
	mlp.fit(fps, labels)
	print(f'Training took {time.time()-start} seconds')

	filename = f'mlp_batch_{args.batch_number}.model'
	pickle.dump(mlp, open(filename, 'wb'))

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--batch_number', type=int, required=True, default=0,
			help='Batch number')
	args = parser.parse_args()

	train_model(args)

	






