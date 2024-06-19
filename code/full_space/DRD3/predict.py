#!/sps/lit/vkozyrev/miniconda3/envs/ML-Spacedock/bin/python3.10

import pickle
import argparse
import numpy as np
from scipy import sparse as ss
from sklearn.neural_network import MLPClassifier

def predict(args):
	
	batch = 1000000
	idxs = np.array(range(args.start, min(args.start+batch, 670708962)))

	model = pickle.load(open('mlp.model', 'rb'))

	fp_amines = np.load('amines.npy')
	fp_acids = np.load('acids.npy')

	idxs_amines = idxs // 19887
	idxs_acids = idxs % 19887

	fp = np.hstack([np.logical_and(fp_amines[idxs_amines], fp_acids[idxs_acids]), np.logical_xor(fp_amines[idxs_amines], fp_acids[idxs_acids])])

	fp = ss.csr_matrix(fp)

	y_pred = model.predict_proba(fp)[:,1]
	np.save(f'preds_{args.start}_{min(args.start+batch, 670708962)}.npy', y_pred)

if __name__ == '__main__':

	parser = argparse.ArgumentParser()
	
	parser.add_argument('-s', '--start', type=int, required=True,
			help='Index of a building block pair from which script should start the prediction')

	args = parser.parse_args()

	predict(args)

