import argparse
import random
import numpy as np

def select_initial_batch(args):

	bbs_fps = np.array([np.load(f'/projects/ML-SpaceDock/CACHE4_full_rf_0.0025_0.01_greedy/bb_{i}.npy') for i in range(21)], dtype=object)
	reactions_rules = np.load('/projects/ML-SpaceDock/CACHE4_full_rf_0.0025_0.01_greedy/reactions_rules.npy')

	pool_size = np.array([len(bbs_fps[reaction[0]])*len(bbs_fps[reaction[1]]) for reaction in reactions_rules]).sum()
	print(f'Pool size: {pool_size}')

	batch_idxs = random.sample(range(pool_size), args.size)
	print(f'Selected batch size: {len(batch_idxs)}')

	np.save('../0_batch_idxs.npy', batch_idxs)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--size', type=int, required=True, default=1676772,
						help='Size of the batch to acquire')
	args = parser.parse_args()

	select_initial_batch(args)
