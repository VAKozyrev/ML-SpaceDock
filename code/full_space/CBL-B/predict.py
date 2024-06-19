#!/sps/lit/vkozyrev/miniconda3/envs/ML-Spacedock/bin/python3.10
import pickle
import argparse
import numpy as np
from scipy import sparse as ss
from sklearn.neural_network import MLPClassifier

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

def predict(args):
	
	batch = 10000000
	idxs = np.array(range(args.start, min(args.start+batch, 6639561469)))
	bbs_fps_paths = [f'bb_{i}.npy' for i in range(21)]

	model = pickle.load(open('mlp.model', 'rb'))

	fps = get_fps(
	bbs_fps = np.array([np.load(path) for path in bbs_fps_paths], dtype=object),
	idxs = idxs,
	reactions_rules = np.load('reactions_rules.npy')
	)

	y_pred = model.predict_proba(fps)[:,1]
	np.save(f'preds_{args.start}_{min(args.start+batch, 6639561469)}.npy', y_pred)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	
	parser.add_argument('-s', '--start', type=int, required=True,
			help='Index of a building block pair from which script should start the prediction')

	args = parser.parse_args()

	predict(args)

