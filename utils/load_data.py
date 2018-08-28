import numpy as np
import glob
import pickle



def load_data(file, task):

	"""
	Load dataset previously generated

		INPUT:
			- file    : a string, the name of the file where the dataset is
			- task	  : an integer or a string, the task(s) present in the dataset

	"""
	
	task = str(task)
	logdir = "./dataset/"+file

	### load training data
	X_training = np.loadtxt(logdir + "/X_train_"+task+'.csv', delimiter = ',')
	Y_training = np.loadtxt(logdir + "/Y_train_"+task+'.csv', delimiter =',')
	### load testing data
	X_testing = np.loadtxt(logdir + "/X_test_"+task+'.csv', delimiter = ',')
	Y_testing = np.loadtxt(logdir + "/Y_test_"+task+'.csv', delimiter = ',')
	### load testing data
	X_validation = np.loadtxt(logdir + "/X_val_"+task+'.csv', delimiter = ',')
	Y_validation = np.loadtxt(logdir + "/Y_val_"+task+'.csv', delimiter = ',')

	states_train = pickle.load( open( logdir + "/states_train.pkl", "rb" ) )
	states_test = pickle.load( open( logdir + "/states_test.pkl", "rb" ))
	states_val = pickle.load( open( logdir + "/states_val.pkl", "rb" ))

	data = {
	"X_training": X_training,
	"Y_training": Y_training,
	"X_testing": X_testing,
	"Y_testing": Y_testing,
	"X_validation": X_validation,
	"Y_validation": Y_validation,
	"states_train": states_train,
	"states_test": states_test,
	"states_val": states_val
	}

	return data
