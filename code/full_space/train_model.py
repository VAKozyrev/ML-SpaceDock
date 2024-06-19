import numpy as np
from scipy import sparse as ss
from tqdm import tqdm

def get_fps(bb_fps, reaction_rules, idxs):

	fps = np.vstack([fps for fps in bb_fps])
	fps_per_bb = np.array([len(fps) for fps in bb_fps])
	bb_borders = np.cumsum(fps_per_bb) - 1
	pairs_per_reaction = np.array([len(bb_fps[rule[0]])*len(bb_fps[rule[1]]) for rule in reaction_rules])
	reaction_borders = np.cumsum(pairs_per_reaction) - 1
	reaction_features = np.identity(len(reaction_rules))

	if len(reaction_rules) > 1:
		result = ss.csr_matrix((0,4096 + len(reaction_rules)))
	else:
		result = ss.csr_matrix((0,4096))

	for batch in tqdm(np.array_split(idxs, 100), desc='Getting fingerprints', unit='batch'):
		reaction_n  = np.searchsorted(reaction_borders, idxs)
		idxs_local = idxs - reaction_borders[reaction_n] + pairs_per_reaction[reaction_n] - 1
		bb1_n, bb2_n = reaction_rules[reaction_n][:,0], reaction_rules[reaction_n][:,1]

		#Indexes of first and second building blocks
		bb_1_idxs = idxs_local // fps_per_bb[bb2_n]
		bb_2_idxs = idxs_local % fps_per_bb[bb2_n]

		bb_1_fps = fps[bb_1_idxs + bb_borders[bb1_n] - fps_per_bb[bb1_n] + 1]
		bb_2_fps = fps[bb_2_idxs + bb_borders[bb2_n] - fps_per_bb[bb2_n] + 1]

		fp = np.hstack([np.logical_and(bb_1_fps, bb_2_fps), np.logical_xor(bb_1_fps, bb_2_fps)])
		if len(reaction_rules) > 1:
			fp = np.hstack([result, reaction_features[reaction_n]])

		result = ss.vstack([result, ss.csr_matrix(fp)])

	return result





# 	fps = ss.csr_matrix((0,4107))
# 	reactions_fps = np.identity(len(reactions_rules), dtype=np.int8)

# 	# number of pairs for each reaction:
# 	pairs_per_reaction = np.array([len(bbs_fps[reaction[0]])*len(bbs_fps[reaction[1]]) for reaction in reactions_rules])
# 	# indexes of the last pair for each reaction:
# 	reaction_borders = np.cumsum(pairs_per_reaction) - 1

# 	for batch in tqdm(np.array_split(idxs, 100), desc='Getting fingerprints', unit='batch'):
# 		# Reaction numbers
# 		reaction_n  = np.searchsorted(reaction_borders, batch)
# 		#Local indexes
# 		idxs_local = batch - reaction_borders[reaction_n] + pairs_per_reaction[reaction_n] - 1
# 		# Building blocks' numbers
# 		bb1_n, bb2_n = reactions_rules[reaction_n][:,0], reactions_rules[reaction_n][:,1]
# 		# Fps corresponding to this building blocks' numbers
# 		fps_1, fps_2 = bbs_fps[bb1_n], bbs_fps[bb2_n]

# 		#Indexes of first and second building blocks
# 		bb_1_idxs = idxs_local // np.vectorize(lambda x: len(x))(fps_2)
# 		bb_2_idxs = idxs_local % np.vectorize(lambda x: len(x))(fps_2)

# 		bb_1_fps = np.array([arr[idx] for arr, idx in zip(fps_1, bb_1_idxs)])
# 		bb_2_fps = np.array([arr[idx] for arr, idx in zip(fps_2, bb_2_idxs)])

# 		fp = np.hstack([np.logical_and(bb_1_fps, bb_2_fps), np.logical_xor(bb_1_fps, bb_2_fps), reactions_fps[reaction_n]])

# 		fps = ss.vstack([fps, ss.csr_matrix(fp)])

# 	return fps

# def get_labels(training_idxs, hits_idxs):
# 	mask = np.isin(training_idxs, hits_idxs, invert=True)
# 	training_idxs[mask] = 0
# 	training_idxs[~mask] = 1
# 	return training_idxs

if __name__ == '__main__':

	bb_fps = np.array([np.load(f'/projects/ML-SpaceDock/data/DRD3_full/bb_{i}.npy') for i in range(2)], dtype=object)
	reaction_rules = np.array([[0,1]])

	print(reaction_rules)

	print(get_fps(bb_fps = bb_fps,
		reaction_rules = reaction_rules,
		idxs = np.random.randint(low=0, high=670000000, size=1000000)
		))
	



# class PairsDataset(Dataset):

# 	def __init__(self, bbs_fps, reaction_rules, hits_idxs):

# 		self.fps = np.vstack([bb_fps for bb_fps in bbs_fps]).astype(bool)
# 		self.reaction_rules = reaction_rules

# 		self.fps_per_bb = np.array([len(bb_fps) for bb_fps in bbs_fps])
# 		self.bb_borders = np.cumsum(self.fps_per_bb) - 1
# 		self.pairs_per_reaction = np.array([len(bbs_fps[reaction[0]])*len(bbs_fps[reaction[1]]) for reaction in reaction_rules])
# 		self.reaction_borders = np.cumsum(self.pairs_per_reaction) - 1
# 		self.size = self.pairs_per_reaction.sum()

# 		self.reaction_fps = np.identity(len(reaction_rules), dtype=bool)

# 		self.y = np.zeros(self.size, dtype=bool)
# 		self.y[hits_idxs] = True

# 	def __len__(self):
# 		return self.size

# 	def __getitem__(self, idx):
# 		# Reaction number
# 		reaction_n  = np.searchsorted(self.reaction_borders, idx)
# 		#Local index
# 		idxs_local = idx - self.reaction_borders[reaction_n] + self.pairs_per_reaction[reaction_n] - 1
# 		# Building blocks' number
# 		bb1_n, bb2_n = self.reaction_rules[reaction_n][0], self.reaction_rules[reaction_n][1]

# 		#Indexes of first and second building blocks
# 		bb_1_idxs = idxs_local // self.fps_per_bb[bb2_n]
# 		bb_2_idxs = idxs_local % self.fps_per_bb[bb2_n]

# 		bb_1_fps = self.fps[bb_1_idxs + self.bb_borders[bb1_n] - self.fps_per_bb[bb1_n] + 1]
# 		bb_2_fps = self.fps[bb_2_idxs + self.bb_borders[bb2_n] - self.fps_per_bb[bb2_n] + 1]

# 		fp = np.hstack([np.logical_and(bb_1_fps, bb_2_fps), np.logical_xor(bb_1_fps, bb_2_fps), self.reaction_fps[reaction_n]])

# 		label = self.y[idx]

# 		return fp, label

