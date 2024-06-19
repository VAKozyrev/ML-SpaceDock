#!/sps/lit/vkozyrev/miniconda3/envs/ML-Spacedock/bin/python3.10
import argparse
import numpy as np

def load_explored_idxs(batch_number):
	files = [f'{i}_batch_idxs.npy' for i in range(0, batch_number, 1)]
	return np.hstack([np.load(file) for file in files])

def load_predictions():
	files = [f'/sps/lit/vkozyrev/ML-Spacedock/CACHE4_full/mlp_greedy_0.25_1/preds_{i*10000000}_{min(i*10000000+10000000, 6639561469)}.npy' for i in range(664)]
	return np.hstack([np.load(file) for file in files])

def select_batch(predictions, explored_idxs, size):
	print(len(predictions))
	sorted_idxs = np.argsort(predictions)[::-1]
	mask = np.isin(sorted_idxs, explored_idxs, invert=True)
	return sorted_idxs[mask][:size]


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-n', '--batch_number', type=int, required=True, default=0,
						help='Batch number')
	parser.add_argument('-s', '--size', type=int, required=True, default=670708,
						help='Size of the batch to acquire')
	args = parser.parse_args()

	batch_idxs = select_batch(
		predictions = load_predictions(),
		explored_idxs = load_explored_idxs(args.batch_number),
		size = args.size
	)

	np.save(f'{args.batch_number}_batch_idxs.npy', batch_idxs)