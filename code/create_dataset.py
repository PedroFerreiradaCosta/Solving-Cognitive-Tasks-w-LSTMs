import numpy as np
import matplotlib.pyplot as plt
import pickle
import utils.generate_trial as gt
from datetime import datetime
import random
import os

from matplotlib.gridspec import GridSpec
from keras.utils import to_categorical

def create_data(task, plot, trials, now=None):

	"""
	Generates a dataset of input and output for a given number of trials

	INPUT:
		- task    : an integer or string, defines which task to follow:
					0 - Go Task
					1 - Reaction Time Go Task
					2 - Delay Go Task
					3 - Anti Task
					4 - Reaction Time Anti Task
					5 - Delay Go Task
					'None' - Every task drawn randomly
		- plot	  : a boolean, to save or not a plot figure per 10 trials
		- trials  : an integer, number of trials to be generated. Value must be >= 10

	"""

	if now is None:
		now = datetime.utcnow().strftime("%Y%m%d%H%M")
	root_logdir = "../data"
	logdir = "{}/run-{}/".format(root_logdir, now)

	trn_trials = int(trials * 0.8)
	tst_trials = int(trials * 0.1)
	val_trials = int(trials * 0.1)

	#Generating a class of trials
	sets = gt.sets()

	if not os.path.exists(logdir):
	    os.makedirs(logdir)
	    os.makedirs(logdir+"images")

	# Saves a text file with the dataset information
	file = open(logdir + "Data.txt", "w")
	file.write('_______________________________________\n')
	file.write('_______________________________________\n')
	file.write("Number of trials: " + str(trials)+ '\n')
	file.write("Number of training trials: " + str(trn_trials)+ '\n')
	file.write("Number of validating trials: " + str(val_trials) + '\n')
	file.write("Number of testing trials: " + str(tst_trials) + '\n')
	file.write("Number of timesteps: " + str(600)+ '\n')
	if task == 'None':
		file.write("Task: All" + '\n')
	else:
		file.write("Task: " + str(task)+ '\n')
	file.write('_______________________________________' + '\n')
	file.write('_______________________________________'+ '\n')







	X_training = []
	Y_training = []
	X_validation = []
	Y_validation = []
	X_testing = []
	Y_testing = []

	states_train = []
	states_val = []
	states_test = []

	data = []


	for i in range(trials):
		tasks = [sets.go_task(), sets.RT_go_task(),
				 sets.Dly_go_task(), sets.anti_task(),
				 sets.RT_anti_task(), sets.Dly_anti_task()]

		if task == 'None':
			task_set = random.randint(0,5) # generates a random distribution between the 6 tasks
		else:
			task_set = task

		temp = tasks[task_set]

		# plots every 10th trial
		if (plot == True and i % 10 ==0):
			print(i)

			gs = GridSpec(4, 4)

			plt.suptitle('Trial ' + str(i) + ', Task ' + temp['task'] , fontsize=16)
			plt.subplot(gs[0,1:3])
			plt.imshow((temp['X'][1:7,:]),cmap="viridis", aspect=10)
			plt.title('Rule Signal')
			plt.subplot(gs[1,:2])
			plt.plot(temp['X'][0,:])
			plt.title('Input Fixation')
			plt.ylim((-0.3, 1.3))
			plt.subplot(gs[1,2:])
			plt.plot(temp['Y'][0,:])
			plt.title('Output Fixation')
			plt.ylim((-0.3, 1.3))
			plt.subplot(gs[2,:2])
			plt.imshow((temp['X'][7:39,:]),cmap="viridis", vmax=1, aspect=2)
			plt.title('Input Modality 1')
			plt.subplot(gs[2,2:])
			plt.imshow((temp['X'][39:,:]),cmap="viridis", vmax=1, aspect=2)
			plt.title('Input Modality 2')
			plt.subplot(gs[3,:])
			plt.imshow((temp['Y'][1:,:]),cmap="viridis")
			plt.title('Output')
			plt.colorbar()
			plt.savefig(logdir + 'images/trial_'+str(i)+temp['task']+'.png')


		if i < trn_trials:
			states_train.append(temp)
			X_training.append(np.ndarray.flatten(np.transpose(temp['X'])))
			Y_training.append(np.asarray(np.where(temp['Y']==1))[0])

		elif i < trn_trials + val_trials:
			states_val.append(temp)
			X_validation.append(np.ndarray.flatten(np.transpose(temp['X'])))
			Y_validation.append(np.asarray(np.where(temp['Y']==1))[0])

		elif i < trn_trials + val_trials + tst_trials:
			states_test.append(temp)
			X_testing.append(np.ndarray.flatten(np.transpose(temp['X'])))
			Y_testing.append(np.asarray(np.where(temp['Y']==1))[0])


	# Saves pickle file with different variables from each trial
	with open(logdir +'states_train.pkl', 'wb') as handle:
		pickle.dump(states_train, handle, protocol=pickle.HIGHEST_PROTOCOL)
	with open(logdir  + 'states_val.pkl', 'wb') as handle:
		pickle.dump(states_val, handle, protocol=pickle.HIGHEST_PROTOCOL)
	with open(logdir  + 'states_test.pkl', 'wb') as handle:
		pickle.dump(states_test, handle, protocol=pickle.HIGHEST_PROTOCOL)

	np.savetxt(logdir + 'X_train_'+str(task)+'.csv',X_training,fmt="%1.2f", delimiter=',')
	np.savetxt(logdir + 'Y_train_'+str(task)+'.csv',Y_training,fmt="%1.2f", delimiter=',')
	np.savetxt(logdir + 'X_test_'+str(task)+'.csv',X_testing,fmt="%1.2f", delimiter=',')
	np.savetxt(logdir + 'Y_test_'+str(task)+'.csv',Y_testing,fmt="%1.2f", delimiter=',')
	np.savetxt(logdir + 'X_val_'+str(task)+'.csv',X_validation,fmt="%1.2f", delimiter=',')
	np.savetxt(logdir + 'Y_val_'+str(task)+'.csv',Y_validation,fmt="%1.2f", delimiter=',')




	data = {
	'X_training': np.stack(X_training),
	'Y_training' : np.stack(Y_training),
	'X_validation' : np.stack(X_validation),
	'Y_validation' : np.stack(Y_validation),
	'X_testing' : np.stack(X_testing),
	'Y_testing' : np.stack(Y_testing),
	'states_train' : states_train,
	'states_val' : states_val,
	'states_test' : states_test
	}

	return data
