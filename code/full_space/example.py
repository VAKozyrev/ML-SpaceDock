import os

if __name__ == '__main__':

	fps_files = [f'../../data/CBLB/bb_{i}.npy' for i in range(21)]
	reaction_rules = '../../data/CBLB/reactions_rules.npy'
	#save_path = '../../test/0_batch_idxs.npy'

	#os.system(f'python select_0_batch.py -fps {" ".join(fps_files)} -r {reaction_rules} -bs 10000000 -sp ../../test/0_batch_idxs.npy -s 4216')
	os.system(f'python train.py -fps {" ".join(fps_files)} -r {reaction_rules} -hi ../../data/CBLB/hits_idxs_q_0.6.npy -ti ../../test/0_batch_idxs.npy -sp ../../test/0_batch_model.pt')
	# os.system(f'python predict.py -fps {" ".join(fps_files)} -r {reaction_rules} -sd {save_dir} -m {model}')
	# os.system(f'python select_batch_heap.py -s 1000000 -ei {" ".join(ti)} -p ../../test/preds/ -sp ../../test/1_batch_idxs.npy')
	#os.system(f'python predict_cpu_batched.py -fps {" ".join(fps_files)} -r {reaction_rules} -m ../../test/0_batch_model.pt -st 180000000 -ed 200000000')