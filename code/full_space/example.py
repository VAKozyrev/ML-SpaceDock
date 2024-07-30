import os

if __name__ == '__main__':

	seed = 1
	fps_files = [f'../../example/bb_{i}.npy' for i in range(4)]
	reaction_rules = '../../example/reactions_rules.npy'






	
	batch_sizes = [1000000, 5000000, 5000000, 5000000]
	batch_number = 0
	os.system(f'python select_0_batch.py -fps {" ".join(fps_files)} -r {reaction_rules} -bs {batch_sizes[batch_number]} -sp ../../example/0_batch_idxs.npy -s {seed}')
	while batch_number < len(batch_sizes)-1:
		train_idxs = [f'../../example/{i}_batch_idxs.npy' for i in range(batch_number+1)]
		os.system(f'python train.py -fps {" ".join(fps_files)} -r {reaction_rules} -hi ../../example/hits_idxs_q_0.6.npy -ti {" ".join(train_idxs)} -sp ../../example/{batch_number}_batch_model.pt')
		os.system(f'python predict.py -fps {" ".join(fps_files)} -r {reaction_rules} -sd ../../example/preds/ -m ../../example/{batch_number}_batch_model.pt')
		os.system(f'python select_batch_heap.py -s {batch_sizes[batch_number + 1]} -ei {" ".join(train_idxs)} -p ../../example/preds/ -sp ../../example/{batch_number+1}_batch_idxs.npy')
		batch_number+= 1		

	#os.system(f'python predict_cpu_batched.py -fps {" ".join(fps_files)} -r {reaction_rules} -m ../../test/0_batch_model.pt -st 180000000 -ed 200000000')