import numpy as np
import pylab as plt
import pickle
import load_data as ld

from matplotlib.gridspec import GridSpec
from keras.utils import to_categorical

from collections import Counter

def check_data( file_name, task):
	"""
	Plots the dataset into the different inputs and outputs provided:
		Rule input, fixation input, modality 1, modelity 2, fixation output,
		reaction to stimulus

		INPUT:
			- file_name    : string, name of the file where data is being checked from
			- task 		   : integer or string, tasks being covered by the dataset

	"""

	Tx = 600
	classes_in = 71
	classes_out = 33
	task = str(task)
	data = ld.load_data(file = 'file_name', task = task)

	X_training = data['X_training']
	Y_training = data['Y_training']

	trn_trials = X_training.shape[0]

	states = data['states_train']


	Y_training_oh = to_categorical(Y_training, num_classes = classes_out) # changing from indices to one-hot vector
	X_training = np.reshape(X_training,(trn_trials,Tx,classes_in))


	for i in range(trn_trials):



		gs = GridSpec(4, 4)

		plt.suptitle('Trial ' + str(i) + ', Task ' + states[i]['task'] , fontsize=16)
		plt.subplot(gs[0,1:3])
		plt.imshow(np.transpose(X_training[i,:,1:7]),cmap="viridis", aspect=10)
		plt.title('Rule Signal')
		plt.subplot(gs[1,:2])
		plt.plot(X_training[i,:,0])
		plt.title('Input Fixation')
		plt.ylim((-0.3, 1.3))
		plt.subplot(gs[1,2:])
		plt.plot(Y_training_oh[i,:,0])
		plt.title('Output Fixation')
		plt.ylim((-0.3, 1.3))
		plt.subplot(gs[2,:2])
		plt.imshow(np.transpose(X_training[i,:,7:39]),cmap="viridis", vmax=1, aspect=2)
		plt.title('Input Modality 1')
		plt.subplot(gs[2,2:])
		plt.imshow(np.transpose(X_training[i,:,39:]),cmap="viridis", vmax=1, aspect=2)
		plt.title('Input Modality 2')
		plt.subplot(gs[3,:])
		plt.imshow(np.transpose(Y_training_oh[i,:,1:]),cmap="viridis")
		plt.title('Output')
		plt.colorbar()
		plt.show()

		return




def data_stats(file_name, task):
	"""
	Analysis variable distribution on the dataset

		INPUT:
			- file_name    : string, name of the file where data is being checked from
			- task 		   : integer or string, tasks being covered by the dataset

	"""

	logdir = './dataset/run-201805141312/'

	Tx = 600
	classes_in = 71
	classes_out = 33

	task = str(task)
	data = ld.load_data(file_name, task)

	X_training = data['X_training']
	states = data['states_train']

	trn_trials = X_training.shape[0]


	task = []
	stim1_dir = []
	stim_mod = []
	Tstim1 = []
	Tdelay = []
	stim2_dir = []
	stim2_mod = []
	Tgo =[]

	for i in range(trn_trials):

		task.append(states[i]['task'])
		stim1_dir.append(states[i]['stim1_dir'])
		stim_mod.append(states[i]['stim1_mod'])
		Tstim1.append(states[i]['Tstim1'])
		Tdelay.append(states[i]['Tdelay'])
		stim2_dir.append(states[i]['stim2_dir'])
		stim2_mod.append(states[i]['stim2_mod'])
		Tgo.append(states[i]['Tgo'])



	count = Counter(task)
	print(count)

	plt.hist(Tgo, rwidth = 0.9)
	plt.title('Tgo')
	plt.show()
	plt.hist(stim1_dir, rwidth = 0.9)
	plt.title('stim1_dir')
	plt.show()
	plt.hist(stim_mod, rwidth = 0.9)
	plt.title('Stim1_mod')
	plt.show()
	plt.hist(Tstim1, rwidth = 0.9)
	plt.title('Tstim1')
	plt.show()
	plt.hist(Tdelay, rwidth = 0.9)
	plt.title('Tdelay')
	plt.show()
	plt.hist(stim2_dir, rwidth = 0.9)
	plt.title('stim2_dir')
	plt.show()
	plt.hist(stim2_mod, rwidth = 0.9)
	plt.title('stim2_mod')
	plt.show()
	plt.bar(count.keys(), count.values(), width = .9, color='g')
	plt.title('Tasks')
	plt.show()

	return
