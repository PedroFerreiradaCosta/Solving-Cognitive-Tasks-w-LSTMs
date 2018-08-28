import numpy as np
import pylab as plt
import load_data as ld
import pickle
import pandas as pd


class study(object):

    """
	Analysis of the model's activations to a given dataset, through a study
	of correlations and variance

    """

	def __init__( self, file_data, file_weights, task, analysis ):
		"""
		INPUT:
			- file_data        : file with the dataset location
			- file_weights     : file with the mdel's weigths location
			- task    		   : task analysed by the dataset
			- analysis 		   : layer 1 or layer 2 analysed
		"""

        self.file_data = file_data

        self.file_weigths = file_weights
        self.task = task
        self.analysis = analysis



	#################
	### LOAD DATA ###
	#################




	#file_data = 'run-201807180919'
	#file_weights = '---run-201805141317t5'

	#task = 'None'

	logdir = "./dataset/"+self.file_data

	X_testing = np.loadtxt(logdir + "/X_test_"+self.task+'.csv', delimiter = ',')
	Y_testing = np.loadtxt(logdir + "/Y_test_"+self.task+'.csv', delimiter = ',')
	states = pickle.load( open( logdir + "/states_test.pkl", "rb" ))


	tst_trials = X_testing.shape[0]
	Tx = 600
	classes_in = 71
	classes_out = 33
	h_units1 = 256
	h_units2 = 128


	layer1 = np.loadtxt('seq_model_logs/'+self.file_weights+'/1stlayer.csv',delimiter=',')
	layer1 = layer1.reshape(tst_trials,Tx, h_units1)
	layer2 = np.loadtxt('seq_model_logs/'+self.file_weights+'/2ndlayer.csv',delimiter=',')
	layer2 = layer2.reshape(tst_trials,Tx, h_units2)
	output = np.loadtxt('seq_model_logs/'+self.file_weights+'/output.csv',delimiter=',')
	output = output.reshape(tst_trials, Tx, classes_out)


	activations1 = []
	activations2 = []
	for i in range(tst_trials):
		if states[i]['task']=='go' or states[i]['task']=='dly_go' or states[i]['task']=='anti' or states[i]['task']=='dly_anti':
			activations1.append(layer1[i, (states[i]['Tgo']+10), :])
			activations2.append(layer2[i, (states[i]['Tgo']+10), :])
		else:
			activations1.append(layer1[i, (states[0]['Tgo']+10), :])
			activations2.append(layer2[i, (states[0]['Tgo']+10), :])
			print(states[1]['task'])


	df1 = np.transpose(np.array(activations1))
	df2 = np.transpose(np.array(activations2))

	if self.analysis == 1:
		layer = h_units1
	elif self.analysis == 2:
		layer = h_units2

	go_i=[]
	dly_go_i=[]
	rt_go_i=[]
	anti_i=[]
	dly_anti_i=[]
	rt_anti_i=[]



	for _ in range(len(states)):
		if states[_]['task']=='go':
			go_i.append(_)
		elif states[_]['task']=='dly_go':
			dly_go_i.append(_)
		elif states[_]['task']=='rt_go':
			rt_go_i.append(_)
		elif states[_]['task']=='anti':
			anti_i.append(_)
		elif states[_]['task']=='dly_anti':
			dly_anti_i.append(_)
		elif states[_]['task']=='rt_anti':
			rt_anti_i.append(_)


	go = []
	dly_go = []
	rt_go = []
	anti = []
	dly_anti = []
	rt_anti = []


	val = []
	var = []

	if self.analysis == 1:
		for _ in range(layer):

			go.append(df1[_][go_i])
			tmp = np.var(df1[_][go_i])
			val.append(tmp)

			dly_go.append(df1[_][dly_go_i])
			tmp = np.var(df1[_][dly_go_i])
			val.append(tmp)

			rt_go.append(df1[_][rt_go_i])
			tmp = np.var(df1[_][rt_go_i])
			val.append(tmp)

			anti.append(df1[_][anti_i])
			tmp = np.var(df1[_][anti_i])
			val.append(tmp)

			dly_anti.append(df1[_][dly_anti_i])
			tmp = np.var(df1[_][dly_anti_i])
			val.append(tmp)

			rt_anti.append(df1[_][rt_anti_i])
			tmp = np.var(df1[_][rt_anti_i])
			val.append(tmp)

			var.append(np.var(df1[_]))

	elif self.analysis == 2:
		for _ in range(layer):

			go.append(df2[_][go_i])
			tmp = np.var(df2[_][go_i])
			val.append(tmp)

			dly_go.append(df2[_][dly_go_i])
			tmp = np.var(df2[_][dly_go_i])
			val.append(tmp)

			rt_go.append(df2[_][rt_go_i])
			tmp = np.var(df2[_][rt_go_i])
			val.append(tmp)

			anti.append(df2[_][anti_i])
			tmp = np.var(df2[_][anti_i])
			val.append(tmp)

			dly_anti.append(df2[_][dly_anti_i])
			tmp = np.var(df2[_][dly_anti_i])
			val.append(tmp)

			rt_anti.append(df2[_][rt_anti_i])
			tmp = np.var(df2[_][rt_anti_i])
			val.append(tmp)

			var.append(np.var(df2[_]))

	else:
		print("analys variable not applied properly. Should be 1 for 1st layer and 2 for 2nd layer")

	val = np.reshape(val,(layer,6))


	go = np.mean(go, axis=1)
	dly_go = np.mean(dly_go, axis=1)
	rt_go = np.mean(rt_go, axis=1)
	anti = np.mean(anti, axis=1)
	dly_anti = np.mean(dly_anti, axis=1)
	rt_anti = np.mean(rt_anti, axis=1)


	set = [go, dly_go, rt_go, anti, dly_anti, rt_anti]#
	tasks = ['go', 'dlygo', 'rtgo', 'anti', 'dlyanti', 'rtanti']


	set = np.stack(set)


	var_tasks = []
	for i in range(layer):
		var_tasks.append(np.var(set[:,i]))



	def corr_units():
		"""
		Plots layer correlations between trials


		"""
		df1 = pd.DataFrame(df1)
		df2 = pd.DataFrame(df2)

		plt.matshow(df1.corr())
		plt.show()

		plt.matshow(df2.corr())
		plt.show()
		return

	def var_check(threshold):
		"""
			returns list of variance above the treshold for the second layer's output
			and prints every variance

					INPUT:
						- threshold	: float, threshold above which variance is reported
						- df2    : string, activations for every unit across trials, layer 2

		"""
		var = []
		val = []
		for _ in range(h_units2):
			tmp = np.var(df2[_])
			val.append(tmp)
			print(str(_) + ' : ' + str(tmp))
			if tmp >= threshold:
				var.append(_)
		return var

	def var_trials():
		"""
		plots the model's variance across all trials, for both layers
		"""


		plt.subplot(211)
		plt.scatter(range(layer), var, marker='^', c='#475f94', edgecolor = 'g')
		plt.title('Variance of 2nd Layer across all trials', fontsize = 20)
		plt.xlabel('Unit')
		plt.ylabel('Variance')
		plt.axis([0, layer, 0, 2])
		plt.subplot(212)
		plt.scatter(range(layer), var_tasks, marker='^', c='#475f94', edgecolor = 'g')
		plt.title('Variance of 2nd Layer', fontsize = 20)
		plt.axis([0, layer, 0, 0.4])
		plt.xlabel('Unit')
		plt.ylabel('Variance')
		plt.show()

		return

	def corr_tasks(file_name):
		"""
		Plots similarity matrices between tasks

				INPUT:
					- file_name    : string, csv file the correlation is saved at

		"""

		set = pd.DataFrame(np.transpose(set))
		correlation = set.corr(method='spearman')
		correlation = np.array(np.arctanh(correlation))   # Fisher's Z Transform
		plt.matshow(correlation)
		plt.xticks([0,1,2,3,4,5],tasks, rotation = 'vertical')
		plt.yticks([0,1,2,3,4,5],tasks)
		plt.colorbar()
		plt.show()

		np.savetxt(file_name, correlation ,delimiter=",",fmt="%1.3f")
		return

	def var_tasks():
		"""
			Plots variance across a given tasks along all trials

		"""

		norm = []
		for _ in range(6):
			val[:,_] = val[:,_] / np.linalg.norm(val[:,_])

		plt.imshow(np.transpose(val), aspect = 10)
		plt.yticks([0,1,2,3,4,5],tasks)
		plt.colorbar()
		plt.show()
