import argparse
import random
import numpy as np

def select_initial_batch(args):

	bbs_fps = [np.load(f'../data/CBLB/bb_{i}.npy').astype(bool) for i in range(21)]
	reactions_rules = np.load('../data/CBLB/reactions_rules.npy')

	pool_size = np.array([len(bbs_fps[reaction[0]])*len(bbs_fps[reaction[1]]) for reaction in reactions_rules]).sum()
	print(f'Pool size: {pool_size}')

	batch_idxs = random.sample(range(pool_size), args.size)
	print(f'Selected batch size: {len(batch_idxs)}')

	np.save('0_batch_idxs.npy', batch_idxs)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--size', type=int, required=True, default=1676772,
						help='Size of the batch to acquire')
	args = parser.parse_args()

	select_initial_batch(args)
