from pathlib import Path
import argparse
import numpy as np 
import heapq
from tqdm import tqdm
import os

def element_generator(file_paths):
    for file in file_paths:
        array = np.load(file)
        for element in array:
            yield element

def select_batch(args):

	work_dir = Path(args.path)

	min_heap = []
	heap_size = args.size
	explored_idxs = np.hstack([np.load(idxs) for idxs in args.explored_indexes])
	explored_idxs = set(explored_idxs.tolist())
	file_paths = [Path(work_dir, f'{i}_preds.npy') for i in range(len(os.listdir(work_dir)))]

	for i, element in enumerate(tqdm(element_generator(file_paths))):     
		if i not in explored_idxs:
			if len(min_heap) < heap_size:
				heapq.heappush(min_heap, (element, i))
			else:
				heapq.heappushpop(min_heap, (element, i))

	top_indexes = np.array([index for value, index in heapq.nlargest(args.size, min_heap)])
	np.save(args.save_path, top_indexes)

if __name__ == '__main__':
	
	parser = argparse.ArgumentParser()
	parser.add_argument('-s', '--size', type=int, required=True,
						help='Size of the batch to acquire')

	parser.add_argument('-ei', '--explored_indexes', type=str, nargs='+', required=True,
						help='Paths to .npy files with already explored indexes')

	parser.add_argument('-p', '--path', type=str, required=True,
						help='Path to directory where preds.npy files are stored')

	parser.add_argument('-sp', '--save_path', type=str, required=True,
						help='Path where to save .npy file with acquired indexes')

	args = parser.parse_args()

	select_batch(args)