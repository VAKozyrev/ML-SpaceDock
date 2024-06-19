import argparse
import numpy as np

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--size', type=int, required=True, default=1676772,
						help='Size of the batch to acquire')
	args = parser.parse_args()

	amines_path = '/scratch/vkozyrev/ML-Spacedock/data/amines.npy'
	acids_path =  '/scratch/vkozyrev/ML-Spacedock/data/acids.npy'

	amines_fp = np.load(amines_path)
	acids_fp = np.load(acids_path)

	batch_idxs = np.random.choice((len(amines_fp)*len(acids_fp)), args.size, replace=False)

	np.save('../0_batch_idxs.npy', batch_idxs)

