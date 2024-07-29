"""
The script selects the initial random batch to train the model, saves the result as .npy array
Example:
	python select_0_batch.py -fps bb_0.npy bb_1.npy bb_2.npy -r reaction_rules.npy -bs 10000000 -sp ../result/0_batch_idxs.npy -s 1
Will output an array of 10000000 random integers in range from 0 to size of the chem space and save them as ../result/0_batch_idxs.npy file
"""

import argparse
import numpy as np


def select_initial_batch(args):

	bbs_fps = [np.load(fps).astype(bool) for fps in args.fingerprints]
	reactions_rules = np.load(args.reaction_rules)

	pool_size = np.array([len(bbs_fps[reaction[0]])*len(bbs_fps[reaction[1]]) for reaction in reactions_rules]).sum() #Calculate size of the chem space
	print(f'Pool size: {pool_size}')

	rng = np.random.default_rng(seed=args.seed)          
	batch_idxs = rng.choice(pool_size, args.batch_size)
	print(f'Selected batch size: {len(batch_idxs)}')

	np.save(args.save_path, batch_idxs)


if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-fps', '--fingerprints', type=str, nargs='+', required=True,
						help='List of paths to .npy files containing fingerprints of building blocks')

	parser.add_argument('-r', '--reaction_rules', type=str, required=True,
						help='Path to .npy file containing reaction rules matrix')

	parser.add_argument('-bs', '--batch_size', type=int, required=True,
						help='Size of the batch to acquire')

	parser.add_argument('-sp', '--save_path', type=str, required=True,
						help='Path to save .npy file with the indexes of acquired batch')

	parser.add_argument('-s', '--seed', type=int, required=False, default=None,
						help='Random seed to select the subset')
	args = parser.parse_args()

	select_initial_batch(args)
