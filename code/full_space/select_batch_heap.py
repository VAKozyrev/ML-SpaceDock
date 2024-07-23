import argparse
import numpy as np 
import heapq
from tqdm import tqdm
import os

def load_explored_idxs(batch_number):
        files = [f'{i}_batch_idxs.npy' for i in range(batch_number+1)]
        return np.hstack([np.load(file) for file in files])

def element_generator(file_paths):
    for file in file_paths:
        array = np.load(file)
        for element in array:
            yield element

def select_batch(args):
	min_heap = []
	heap_size = args.size
	explored_idxs = load_explored_idxs(args.batch_n)
	explored_idxs = set(explored_idxs.tolist())
	file_paths = [f'preds/{i}_preds.npy' for i in range(len(os.listdir('preds/')))]

	for i, element in enumerate(tqdm(element_generator(file_paths))):
		if i not in explored_idxs:
			if len(min_heap) < heap_size:
				heapq.heappush(min_heap, (element, i))
			else:
				heapq.heappushpop(min_heap, (element, i))

	top_indexes = np.array([index for value, index in heapq.nlargest(args.size, min_heap)])
	np.save(f'{args.batch_n + 1}_batch_idxs.npy', top_indexes)

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--size', type=int, required=True, default=1676772,
						help='Size of the batch to acquire')
	parser.add_argument('-n', '--batch_n', type=int, required=True,
						help='batch number')
	args = parser.parse_args()

	select_batch(args)