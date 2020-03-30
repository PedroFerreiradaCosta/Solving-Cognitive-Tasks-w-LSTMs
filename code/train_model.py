import numpy as np
import keras
import pylab as plt
import create_dataset as cdt
import load_data as ld
import model
from keras.models import Sequential, Model
from keras.layers import SimpleRNN,Activation, LSTM, Dense
from keras.utils import to_categorical, plot_model
from keras.callbacks import ModelCheckpoint, EarlyStopping
from keras import backend as K
from datetime import datetime
import argparse
import random
import os

def main(is_2layers, load_data, root_logdir=None, weight_path=None):
	"""
	Trains a recurrent model on a given dataset.
	
	INPUT:
	is_2layers - boolean, false for 1 layer model, true for 2 layers
	root_logdir - string, file where model and data is saved at
	load_data - boolean, true if dataset loaded, false if dataset generated
	weight_path - string path to model's weights


	"""
	if root_logdir is not None:
		logdir = "../output/{}/".format(root_logdir)
	else:
		now = datetime.utcnow().strftime("%Y%m%d%H%M")
		# root_logdir = "./seq_model_logs"
		logdir = "../output/run-{}/".format(now)

	# Generates the paths for the model outputs
	if not os.path.exists(logdir):
	    os.makedirs(logdir)
	    os.makedirs(logdir+"outputs")
	    os.makedirs(logdir+"images_output")
	    os.makedirs(logdir+"images_1stlayer")
	    if is_2layers:
		    os.makedirs(logdir+"images_2ndlayer")




	if load_data:
		print('[INFO] Loading Dataset...')
		file_data = root_logdir
		data = ld.load_data(file = file_data, task = 'None')

	else:
		print('[INFO] Generating Dataset...')
		data = cdt.create_data(task = 'None', plot = False, trials = 2000, now=now)



	X_training = data['X_training']
	X_testing = data['X_testing']
	X_validating = data['X_validation']
	Y_training = data['Y_training']
	Y_testing = data['Y_testing']
	Y_validating = data['Y_validation']

	states = data['states_train']
	states_tst = data['states_test']

	# Model's and dataset parameters
	Tx = 600
	classes_in = 71
	classes_out = 33
	h_units1 = 512
	if is_2layers:
		h_units2 = 256
	n_epochs = 250
	n_batch = 64

	trn_trials = X_training.shape[0]
	val_trials = X_validating.shape[0]
	tst_trials = X_testing.shape[0]

	trn_index = np.random.shuffle(np.arange(trn_trials))
	val_index = np.random.shuffle(np.arange(val_trials))

	X_training = X_training[trn_index,:]
	X_validating = X_validating[val_index,:]
	Y_training = Y_training[trn_index]
	Y_validating = Y_validating[val_index]


	Y_training_oh = to_categorical(Y_training, num_classes = classes_out) # changing from indices to one-hot vector
	Y_validating_oh = to_categorical(Y_validating, num_classes = classes_out)
	Y_testing_oh = to_categorical(Y_testing, num_classes = classes_out)

	### reshape
	X_training = np.reshape(X_training,(trn_trials,Tx,classes_in))
	X_validating = np.reshape(X_validating,(val_trials,Tx,classes_in))
	X_testing = np.reshape(X_testing,(tst_trials,Tx,classes_in))

	### reshape
	Y_training_oh = np.reshape(Y_training_oh,(trn_trials,Tx,classes_out))
	Y_validating_oh = np.reshape(Y_validating_oh,(val_trials,Tx,classes_out))
	Y_testing_oh = np.reshape(Y_testing_oh,(tst_trials,Tx,classes_out))


	# Generates a text file with the model's information
	file = open(logdir + "Train_Model.txt", "w")
	file.write('_______________________________________\n')
	file.write('_______________________________________\n')
	file.write('_______________________________________\n')
	file.write("Number of classes of Input: " + str(classes_in)+ '\n')
	file.write("Number of classes of Output: " + str(classes_out)+ '\n')
	file.write("Number of timesteps: " + str(Tx) + '\n')
	file.write("Number of Lstm units - 1st Layer: " + str(h_units1) + '\n')
	if is_2layers:
		file.write("Number of Lstm units - 2nd Layer: " + str(h_units2) + '\n')
	file.write("Number of epochs: " + str(n_epochs)+ '\n')
	file.write("Number of batches: " + str(n_batch)+ '\n')
	file.write("Number of training trials per epoch: " + str(trn_trials)+ '\n')
	file.write("Number of validation trials per epoch: " + str(val_trials)+ '\n')
	file.write("Number of testing trials: " + str(tst_trials)+ '\n')
	if load_data:
		file.write("Data loaded from : " + file_data+ '\n')
	else:
		file.write("Data created on: " + "run-{}".format(now) + '\n')
	file.write('_______________________________________' + '\n')
	file.write('_______________________________________'+ '\n')

	### defines model
	model, first_layer, second_layer= model.cog_lstm(X_training, 
							weight_path, is_2layers,
							Tx, classes_in, classes_out, h_units1,
							h_units2)

	# fit network
	callbacks = [ModelCheckpoint(filepath=logdir + 'weights.hdf5', verbose=1, save_best_only=True)]
	history = model.fit(X_training, Y_training_oh, epochs=n_epochs, batch_size=n_batch,
	validation_data=(X_validating, Y_validating_oh), verbose=2, callbacks=callbacks)#shuffle=False,

	# serialize model to JSON
	model_json = model.to_json()
	with open(logdir + "model.json", "w") as json_file:
	    json_file.write(model_json)
	# serialize weights to HDF5
	print("Saved model to disk")

	model.save(logdir + 'model.h5')

	weights = model.get_weights()
	for m in range(len(weights)):
	        np.savetxt(logdir + 'outputs/output_weights'+str(m)+'.csv',weights[m],fmt="%1.2f",delimiter=',')


	model.save_weights(logdir + 'weights.h5')

	#plot_model(model, to_file=logdir+'model.png', show_shapes=True, show_layer_names=True, rankdir='TB')


	#plot history
	plt.plot(history.history['loss'], label='train')
	plt.plot(history.history['val_loss'], label='test')
	plt.legend()
	plt.savefig(logdir + 'loss.png')
	plt.close()


	scores = model.evaluate(X_testing, Y_testing_oh, verbose=2)

	print("%s: %.2f" % ('Training Loss', history.history['loss'][-1]))
	print("%s: %.2f%%" % ('Training Accuracy', history.history['categorical_accuracy'][-1]*100))
	print("%s: %.2f" % ('Validation Loss', history.history['val_loss'][-1]))
	print("%s: %.2f%%" % ('Validation Accuracy', history.history['val_categorical_accuracy'][-1]*100))
	print("%s: %.2f" % ('Testing Loss', scores[0]))
	print("%s: %.2f%%" % ('Testing Accuracy', scores[1]*100))

	file.write("%s: %.2f" % ('Training Loss', history.history['loss'][-1]) + '\n')
	file.write("%s: %.2f%%" % ('Training Accuracy', history.history['categorical_accuracy'][-1]*100) + '\n')
	file.write("%s: %.2f" % ('Validation Loss', history.history['val_loss'][-1]) + '\n')
	file.write("%s: %.2f%%" % ('Validation Accuracy', history.history['val_categorical_accuracy'][-1]*100) + '\n')
	file.write("%s: %.2f" % ('Test Loss', scores[0]) + '\n')
	file.write("%s: %.2f%%" % ('Test Accuracy', scores[1]*100)+ '\n')


	### Save output to csv file ###
	pred_output = model.predict(X_testing)

	pred_output = np.stack(pred_output)

	pred_output = np.reshape(pred_output,(tst_trials,Tx,classes_out))
	for _ in range(tst_trials):

			plt.subplot(131)
			plt.imshow(np.reshape(X_testing[_,:], (Tx, classes_in)), cmap="viridis")
			plt.title('X')
			plt.subplot(132)
			plt.imshow(np.reshape(Y_testing_oh[_,:],(Tx,classes_out)),cmap="viridis", aspect=0.5)
			plt.title('Y')
			plt.subplot(133)
			plt.imshow(np.reshape(pred_output[_,:],(Tx,classes_out)),cmap="viridis", aspect=0.5)
			plt.title('Y_hat')
			plt.colorbar()
			plt.savefig(logdir + 'images_output/trial_'+str(_)+states_tst[_]['task']+'.png')
			plt.close()

	np.savetxt(logdir + 'output.csv',np.reshape(pred_output,(tst_trials,Tx*classes_out)),fmt="%1.2f", delimiter=',')


	pred_layer1 = first_layer.predict(X_testing)
	pred_layer1 = np.stack(pred_layer1)
	pred_layer1 = np.reshape(pred_layer1,(tst_trials,Tx,h_units1))

	# Plot predictions
	for _ in range(tst_trials):

			plt.subplot(131)
			plt.imshow(np.reshape(X_testing[_,:], (Tx, classes_in)), cmap="viridis")
			plt.title('X')
			plt.subplot(132)
			plt.imshow(np.reshape(Y_testing_oh[_,:],(Tx,classes_out)),cmap="viridis", aspect=0.5)
			plt.title('Y')
			plt.subplot(133)
			plt.imshow(np.reshape(pred_layer1[_,:],(Tx,h_units1)),cmap="viridis")
			plt.title('LSTM1')
			plt.colorbar()
			plt.savefig(logdir + 'images_1stlayer/trial_'+str(_)+states_tst[_]['task']+'.png')
			plt.close()

	np.savetxt(logdir + '1stlayer.csv',np.reshape(pred_layer1,(tst_trials,Tx*h_units1)),fmt="%1.2f", delimiter=',')

	if is_2layers:
		pred_layer2 = second_layer.predict(X_testing)
		pred_layer2 = np.stack(pred_layer2)
		pred_layer2 = np.reshape(pred_layer2,(tst_trials,Tx,h_units2))
		for _ in range(tst_trials):

				plt.subplot(131)
				plt.imshow(np.reshape(X_testing[_,:], (Tx, classes_in)), cmap="viridis")
				plt.title('X')
				plt.subplot(132)
				plt.imshow(np.reshape(Y_testing_oh[_,:],(Tx,classes_out)),cmap="viridis", aspect=0.5)
				plt.title('Y')
				plt.subplot(133)
				plt.imshow(np.reshape(pred_layer2[_,:],(Tx,h_units2)),cmap="viridis")
				plt.title('LSTM2')
				plt.colorbar()
				plt.savefig(logdir + 'images_2ndlayer/trial_'+str(_)+states_tst[_]['task']+'.png')
				plt.close()


		np.savetxt(logdir + '2ndlayer.csv', np.reshape(pred_layer2,(tst_trials,Tx*h_units2)), fmt="%1.2f", delimiter=',')

		return now

if __name__ == '__main__':
	parser = argparse.ArgumentParser()
	parser.add_argument('-2l',
						dest='is_2layers',
						help='If model is 2 layered or just one',
						type=bool,
						required=True)
	parser.add_argument('-load',
						dest='load_data',
						help='If data is loaded or created',
						type=bool,
						required=True)
	parser.add_argument('-logdir',
						dest='root_logdir',
						help='Folder where to save the model',
						type=str,
						required=False)
	parser.add_argument('-weight',
						dest='weight_path',
						help='Path to weights to pre-load to model',
						type=str,
						required=False)

	args = parser.parse_args()

	main(args.is_2layers, args.load_data, args.root_logdir, args.weight_path)

