
from keras.models import Model
from keras.layers import Input
from keras.layers import LSTM, Dense
from keras.optimizers import Adam
from keras.utils import to_categorical
from keras.callbacks import ModelCheckpoint
from keras import backend as K
import matplotlib.pyplot as plt
import numpy as np
import create_dataset as cdt
from sklearn.manifold import TSNE
from sklearn.preprocessing import normalize
from datetime import datetime
import load_data as ld
import time
import os

"""
Analysis a previously trained model by checking accuracy on different datasets,
plotting prediction outputs and the respective cell states and inner layer outputs,
plotting a number of t-SNE plots on the different variables of each trial

file_weights - string, the name of the file where the weights of the saved model
are stored
load_data = boolean, true if dataset is to be loaded, false if is to be generated


"""

now = datetime.utcnow().strftime("%Y%m%d%H%M")
root_logdir = "./model_logs_2l"
logdir = "{}/run-{}-nounit160dlyanti/".format(root_logdir, now)


if not os.path.exists(logdir):
    os.makedirs(logdir)
    os.makedirs(logdir+"/outputs")
    os.makedirs(logdir+"/images_output")
    os.makedirs(logdir+"/images_hidden_c1")
    os.makedirs(logdir+"/images_hidden_c2")
    os.makedirs(logdir+"/images_hidden_lstm")


classes_in = 71 # number of input units
classes_out = 33 # number of output units
Tx = 600 # total timesteps
batch_size = 1
load_data = True

#################
### LOAD DATA ###
#################

file_weights = '---run-201805141317t5'

if load_data:
    file_data = 'run-201808211048'
    data = ld.load_data(file = file_data, task = 0)
else:
    file_data = "run-{}/".format(now)
    data = cdt.create_data(task = 'None', plot = False, trials = 300)

X_training = data['X_training']
X_testing = data['X_testing']
X_validation = data['X_validation']
Y_training = data['Y_training']
Y_testing = data['Y_testing']
Y_validation = data['Y_validation']

states = data['states_train']

X_training = X_training[:200,:]
Y_training = Y_training[:200]
states = states[:200]

trn_trials = X_training.shape[0] # train trials
tst_trials = X_testing.shape[0] # test trials

task = []
stim1_dir = []
stim_mod = []
Tstim1 = []
Tdelay = []
stim2_dir = []
stim2_mod = []
Tgo =[]

# Extraction of information on each trial

for i in range(trn_trials):

    task.append(states[i]['task'])
    stim1_dir.append(states[i]['stim1_dir'])
    stim_mod.append(states[i]['stim1_mod'])
    Tstim1.append(states[i]['Tstim1'])
    Tdelay.append(states[i]['Tdelay'])
    stim2_dir.append(states[i]['stim2_dir'])
    stim2_mod.append(states[i]['stim2_mod'])
    Tgo.append(states[i]['Tgo'])


# Structure of the trained model
h_units1 = 256 # number of LSTM units first layer
h_units2 = 128 # second layer of lstms

n_epoch = 1 # number of epochs
pred_output = np.zeros((trn_trials, Tx, classes_out))
pred_hidden_c1 = np.zeros((trn_trials, Tx, h_units1))
pred_hidden_c2 = np.zeros((trn_trials, Tx, h_units2))
pred_hidden_lstm = np.zeros((trn_trials, Tx, h_units1))

### reshape
X_training = np.reshape(X_training,(trn_trials,Tx,classes_in)) # (m, Tx, classes)
X_testing = np.reshape(X_testing,(tst_trials,Tx,classes_in))

# define model
inputs1 = Input(batch_shape = (1,1,classes_in))
lstm1, state_1h, state_1c = LSTM(h_units1, dropout = 0.2, batch_input_shape=(1, 1, 33), return_state=True, stateful=True, return_sequences=True)(inputs1)
lstm2, state_2h, state_2c = LSTM(h_units2, dropout = 0.2, batch_input_shape=(1, 1, 33), return_state=True, stateful=True)(lstm1)
out = Dense(classes_out, activation = 'softmax')(lstm2)


model_out = Model(inputs= inputs1, outputs = out)
model_lstm1= Model(inputs= inputs1, outputs = lstm1)
model_c1 = Model(inputs= inputs1, outputs = state_1c)
model_c2 = Model(inputs = inputs1, outputs = state_2c)


# Gradient Optimizer
opt = Adam(lr=0.01, beta_1 = 0.9, beta_2 = 0.999, decay = 0.01)
model_out.compile(optimizer = opt, loss = "categorical_crossentropy", metrics=['categorical_accuracy'])

weights = []
weights.append(np.loadtxt('seq_model_logs/'+file_weights+'/outputs/output_weights_nounit160.csv',delimiter=','))
weights.append(np.loadtxt('seq_model_logs/'+file_weights+'/outputs/output_weights1.csv',delimiter=','))
weights.append(np.loadtxt('seq_model_logs/'+file_weights+'/outputs/output_weights2.csv',delimiter=','))
weights.append(np.loadtxt('seq_model_logs/'+file_weights+'/outputs/output_weights3.csv',delimiter=','))
weights.append(np.loadtxt('seq_model_logs/'+file_weights+'/outputs/output_weights4.csv',delimiter=','))
weights.append(np.loadtxt('seq_model_logs/'+file_weights+'/outputs/output_weights5.csv',delimiter=','))
weights.append(np.loadtxt('seq_model_logs/'+file_weights+'/outputs/output_weights6.csv',delimiter=','))
weights.append(np.loadtxt('seq_model_logs/'+file_weights+'/outputs/output_weights7.csv',delimiter=','))

# Setting of the weights extracted from training
model_out.set_weights(weights[0:7])
model_c1.set_weights(weights[0:2])
model_c2.set_weights(weights[0:5])
model_lstm1.set_weights(weights[0:2])

# Generates a textfile with the analysis information
file = open(logdir + "Train_Model.txt", "w")
file.write('___________________________________\n')
file.write('___________________________________\n')
file.write("Number of classes Input: " + str(classes_in)+ '\n')
file.write("Number of classes Output: " + str(classes_out)+ '\n')
file.write("Number of timesteps: " + str(Tx) + '\n')
file.write("Number of Lstm units - 1st Layer: " + str(h_units1) + '\n')
file.write("Number of Lstm units - 2st Layer: " + str(h_units2) + '\n')
file.write("Number of epochs: " + str(n_epoch)+ '\n')
file.write("Number of trials per epoch: " + str(trn_trials)+ '\n')
file.write("Data from " + file_data + '\n')
file.write("Weights from " + file_weights + '\n')
file.write('___________________________________' + '\n')
file.write('___________________________________'+ '\n')


model_loss = []
model_acc = []
for n in range(n_epoch):

    start_time = time.time()
    mean_ep_acc = []
    mean_ep_loss = []

    print("Epoch " + str(n+1) + "/" + str(n_epoch))
    file.write("Epoch " + str(n+1) + "/" + str(n_epoch) + '\n')
    model_out.summary()



    print("\nEvaluating...\n")
    file.write("\nEvaluating...\n")

    for i in range(trn_trials):

        Y_training_oh = to_categorical(Y_training[i,:], num_classes = classes_out)

        mean_tr_acc = []
        mean_tr_loss = []


        for j in range(Tx):



            pred_output[i, j, :] = model_out.predict(X_training[i,j,:].reshape(1,1,classes_in))

            mean_tr_acc.append(np.argmax(pred_output[i, j, :])==Y_training[i,j])

            pred_hidden_c1[i,j,:] = model_c1.predict(X_training[i,j,:].reshape(1,1,classes_in))
            pred_hidden_c2[i,j,:] = model_c2.predict(X_training[i,j,:].reshape(1,1,classes_in))
            pred_hidden_lstm[i,j,:] = model_lstm1.predict(X_training[i,j,:].reshape(1,1,classes_in))

        model_out.reset_states()
        model_c1.reset_states()
        model_c2.reset_states()
        model_lstm1.reset_states()
        if i % 100 == 0:

            print("Trial " + str(i))


            print('___________________________________')

            file.write("Trial " + str(i) + '\n')
            file.write('Accuracy = {0:.2f}'.format(np.mean(mean_tr_acc)) + '\n')
            file.write('___________________________________\n')


        mean_ep_acc.append(np.mean(mean_tr_acc))


    print('Epoch accuracy = {0:.2f}'.format(np.mean(mean_ep_acc)))

    print('___________________________________')
    print('___________________________________')

    file.write("Running time:  %s seconds" % (time.time() - start_time) + '\n')
    file.write('___________________________________\n')
    file.write('___________________________________\n')

    model_acc.append(np.mean(mean_ep_acc))


# Save information on model accuracy
plt.plot(mean_ep_acc)
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('trial')
plt.savefig(logdir + 'model_loss_accuracy.png')
plt.close()
np.savetxt(logdir + 'model_acc.csv',mean_ep_acc,fmt="%1.2f",delimiter=',')


# Save model's prediction output and cell states
for i in range(trn_trials):
    if i % 1 == 0:

        Y_training_oh = to_categorical(Y_training[i,:], num_classes = classes_out)
        plt.subplot(131)
        plt.imshow(np.reshape(X_training[i,:], (Tx, classes_in)), cmap="viridis")
        plt.title('X')
        plt.subplot(132)
        plt.imshow(np.reshape(Y_training_oh,(Tx,classes_out)),cmap="viridis")
        plt.title('Y')
        plt.subplot(133)
        plt.imshow(np.reshape(pred_output[i,:,:],(Tx,classes_out)),cmap="viridis")
        plt.title('Y_hat')
        plt.colorbar()
        plt.savefig(logdir + 'images_output/train_trial_'+str(i)+'_task_' + str(states[i]['task']) + '.png')
        plt.close()

        plt.imshow(np.reshape(pred_hidden_c1[i,:,:],(Tx,h_units1)),cmap="viridis")
        plt.colorbar()
        plt.savefig(logdir + 'images_hidden_c1/train_trial_'+str(i)+'_task_' + str(states[i]['task'])+'.png')
        plt.close()
        np.savetxt(logdir + 'images_hidden_c1/train_trial_'+str(i)+'_task_' + str(states[i]['task'])+'.csv', np.reshape(pred_hidden_c1[i,:,:],(Tx,h_units1)), fmt="%1.2f", delimiter = ',')

        plt.imshow(np.reshape(pred_hidden_c2[i,:,:],(Tx,h_units2)),cmap="viridis")
        plt.colorbar()
        plt.savefig(logdir + 'images_hidden_c2/train_trial_'+str(i)+'_task_' + str(states[i]['task'])+'.png')
        plt.close()
        np.savetxt(logdir + 'images_hidden_c2/train_trial_'+str(i)+'_task_' + str(states[i]['task'])+'.csv', np.reshape(pred_hidden_c2[i,:,:],(Tx,h_units2)), fmt="%1.2f", delimiter = ',')

        plt.imshow(np.reshape(pred_hidden_lstm[i,:,:],(Tx,h_units1)),cmap="viridis")
        plt.colorbar()
        plt.savefig(logdir + 'images_hidden_lstm/train_trial_'+str(i)+'_task_' + str(states[i]['task'])+'.png')
        plt.close()
        np.savetxt(logdir + 'images_hidden_lstm/train_trial_'+str(i)+'_task_' + str(states[i]['task'])+'.csv', np.reshape(pred_hidden_lstm[i,:,:],(Tx,h_units1)), fmt="%1.2f", delimiter = ',')


taski = []
for i in range(len(task)):
    if task[i] == 'go':
        taski.append(0)
    elif task[i] == 'dly_go':
        taski.append(1)
    elif task[i] == 'rt_go':
        taski.append(2)
    elif task[i] == 'anti':
        taski.append(3)
    elif task[i] == 'dly_anti':
        taski.append(4)
    elif task[i] == 'rt_anti':
        taski.append(5)

stim1_dir = np.stack(stim1_dir).reshape(-1)
stim_mod = np.stack(stim_mod).reshape(-1)
Tstim1 = np.stack(Tstim1).reshape(-1)
Tdelay = np.stack(Tdelay).reshape(-1)
stim2_dir = np.stack(stim2_dir).reshape(-1)
stim2_mod = np.stack(stim2_mod).reshape(-1)
Tgo =np.stack(Tgo).reshape(-1)

# Generates t-SNE plots for the different layer outputs for the different variables
pred_hidden_c1 = np.reshape(pred_hidden_c1,(trn_trials,Tx*h_units1))
tsne =TSNE(n_components=2)
encoded_hidden_c1 = tsne.fit_transform(pred_hidden_c1)

plt.scatter(encoded_hidden_c1[:,0],encoded_hidden_c1[:,1], c=taski)
plt.colorbar()
plt.savefig(logdir + 'trn_tsne_hidden_c1_task.png')
plt.show()

plt.scatter(encoded_hidden_c1[:,0],encoded_hidden_c1[:,1], c=stim1_dir)
plt.colorbar()
plt.savefig(logdir + 'trn_tsne_hidden_c1_stim1_dir.png')
plt.show()

plt.scatter(encoded_hidden_c1[:,0],encoded_hidden_c1[:,1], c=stim_mod.ravel())
plt.colorbar()
plt.savefig(logdir + 'trn_tsne_hidden_c1_stim1_mod.png')
plt.show()

plt.scatter(encoded_hidden_c1[:,0],encoded_hidden_c1[:,1], c=Tstim1.ravel())
plt.colorbar()
plt.savefig(logdir + 'trn_tsne_hidden_c1_Tstim1.png')
plt.show()

plt.scatter(encoded_hidden_c1[:,0],encoded_hidden_c1[:,1], c=Tdelay.ravel())
plt.colorbar()
plt.savefig(logdir + 'trn_tsne_hidden_c1_Tdelay.png')
plt.show()

plt.scatter(encoded_hidden_c1[:,0],encoded_hidden_c1[:,1], c=stim2_dir.ravel())
plt.colorbar()
plt.savefig(logdir + 'trn_tsne_hidden_c1_stim2_dir.png')
plt.show()

plt.scatter(encoded_hidden_c1[:,0],encoded_hidden_c1[:,1], c=stim2_mod.ravel())
plt.colorbar()
plt.savefig(logdir + 'trn_tsne_hidden_c1_stim2_mod.png')
plt.show()

plt.scatter(encoded_hidden_c1[:,0],encoded_hidden_c1[:,1], c=Tgo.ravel())
plt.colorbar()
plt.savefig(logdir + 'trn_tsne_hidden_c1_Tgo.png')
plt.show()

pred_hidden_c2 = np.reshape(pred_hidden_c2,(trn_trials,Tx*h_units2))
tsne =TSNE(n_components=2)
encoded_hidden_c2 = tsne.fit_transform(pred_hidden_c2)

plt.scatter(encoded_hidden_c2[:,0],encoded_hidden_c2[:,1], c=taski)
plt.colorbar()
plt.savefig(logdir + 'trn_tsne_hidden_c2_task.png')
plt.show()

plt.scatter(encoded_hidden_c2[:,0],encoded_hidden_c2[:,1], c=stim1_dir)
plt.colorbar()
plt.savefig(logdir + 'trn_tsne_hidden_c2_stim1_dir.png')
plt.show()

plt.scatter(encoded_hidden_c2[:,0],encoded_hidden_c2[:,1], c=stim_mod.ravel())
plt.colorbar()
plt.savefig(logdir + 'trn_tsne_hidden_c2_stim_mod.png')
plt.show()

plt.scatter(encoded_hidden_c2[:,0],encoded_hidden_c2[:,1], c=Tstim1.ravel())
plt.colorbar()
plt.savefig(logdir + 'trn_tsne_hidden_c2_Tstim1.png')
plt.show()

plt.scatter(encoded_hidden_c2[:,0],encoded_hidden_c2[:,1], c=Tdelay.ravel())
plt.colorbar()
plt.savefig(logdir + 'trn_tsne_hidden_c2_Tdelay.png')
plt.show()

plt.scatter(encoded_hidden_c2[:,0],encoded_hidden_c2[:,1], c=stim2_dir.ravel())
plt.colorbar()
plt.savefig(logdir + 'trn_tsne_hidden_c2_stim2_dir.png')
plt.show()

plt.scatter(encoded_hidden_c2[:,0],encoded_hidden_c2[:,1], c=stim2_mod.ravel())
plt.colorbar()
plt.savefig(logdir + 'trn_tsne_hidden_c2_stim2_mod.png')
plt.show()

plt.scatter(encoded_hidden_c2[:,0],encoded_hidden_c2[:,1], c=Tgo.ravel())
plt.colorbar()
plt.savefig(logdir + 'trn_tsne_hidden_c2_Tgo.png')
plt.show()

pred_hidden_lstm = np.reshape(pred_hidden_lstm,(trn_trials,Tx*h_units1))
tsne =TSNE(n_components=2)
encoded_hidden_lstm = tsne.fit_transform(pred_hidden_lstm)

plt.scatter(encoded_hidden_lstm[:,0],encoded_hidden_lstm[:,1], c=taski)
plt.colorbar()
plt.savefig(logdir + 'trn_tsne_hidden_lstm1_task.png')
plt.show()

plt.scatter(encoded_hidden_lstm[:,0],encoded_hidden_lstm[:,1], c=stim1_dir)
plt.colorbar()
plt.savefig(logdir + 'trn_tsne_hidden_lstm1_stim1_dir.png')
plt.show()

plt.scatter(encoded_hidden_lstm[:,0],encoded_hidden_lstm[:,1], c=stim_mod.ravel())
plt.colorbar()
plt.savefig(logdir + 'trn_tsne_hidden_lstm1_stim_mod.png')
plt.show()

plt.scatter(encoded_hidden_lstm[:,0],encoded_hidden_lstm[:,1], c=Tstim1.ravel())
plt.colorbar()
plt.savefig(logdir + 'trn_tsne_hidden_lstm1_Tstim1.png')
plt.show()

plt.scatter(encoded_hidden_lstm[:,0],encoded_hidden_lstm[:,1], c=Tdelay.ravel())
plt.colorbar()
plt.savefig(logdir + 'trn_tsne_hidden_lstm1_Tdelay.png')
plt.show()

plt.scatter(encoded_hidden_lstm[:,0],encoded_hidden_lstm[:,1], c=stim2_dir.ravel())
plt.colorbar()
plt.savefig(logdir + 'trn_tsne_hidden_lstm1_stim2_dir.png')
plt.show()

plt.scatter(encoded_hidden_lstm[:,0],encoded_hidden_lstm[:,1], c=stim2_mod.ravel())
plt.colorbar()
plt.savefig(logdir + 'trn_tsne_hidden_lstm1_stim2_mod.png')
plt.show()

plt.scatter(encoded_hidden_lstm[:,0],encoded_hidden_lstm[:,1], c=Tgo.ravel())
plt.colorbar()
plt.savefig(logdir + 'trn_tsne_hidden_lstm1_Tgo.png')
plt.show()


pred_output = np.reshape(pred_output,(trn_trials,Tx*classes_out))
tsne =TSNE(n_components=2)
encoded_hidden_out = tsne.fit_transform(pred_output)

plt.scatter(encoded_hidden_out[:,0],encoded_hidden_out[:,1], c=taski)
plt.colorbar()
plt.savefig(logdir + 'trn_tsne_hidden_out_task.png')
plt.show()

plt.scatter(encoded_hidden_out[:,0],encoded_hidden_out[:,1], c=stim1_dir)
plt.colorbar()
plt.savefig(logdir + 'trn_tsne_hidden_out_stim1_dir.png')
plt.show()

plt.scatter(encoded_hidden_out[:,0],encoded_hidden_out[:,1], c=stim_mod.ravel())
plt.colorbar()
plt.savefig(logdir + 'trn_tsne_hidden_out_stim_mod.png')
plt.show()

plt.scatter(encoded_hidden_out[:,0],encoded_hidden_out[:,1], c=Tstim1.ravel())
plt.colorbar()
plt.savefig(logdir + 'trn_tsne_hidden_out_Tstim1.png')
plt.show()

plt.scatter(encoded_hidden_out[:,0],encoded_hidden_out[:,1], c=Tdelay.ravel())
plt.colorbar()
plt.savefig(logdir + 'trn_tsne_hidden_out_Tdelay.png')
plt.show()

plt.scatter(encoded_hidden_out[:,0],encoded_hidden_out[:,1], c=stim2_dir.ravel())
plt.colorbar()
plt.savefig(logdir + 'trn_tsne_hidden_out_stim2_dir.png')
plt.show()

plt.scatter(encoded_hidden_out[:,0],encoded_hidden_out[:,1], c=stim2_mod.ravel())
plt.colorbar()
plt.savefig(logdir + 'trn_tsne_hidden_out_stim2_mod.png')
plt.show()

plt.scatter(encoded_hidden_out[:,0],encoded_hidden_out[:,1], c=Tgo.ravel())
plt.colorbar()
plt.savefig(logdir + 'trn_tsne_hidden_out_Tgo.png')
plt.show()

plt.scatter(encoded_hidden_out[:,0],encoded_hidden_out[:,1], c=Tstim1.ravel())
plt.colorbar()
plt.savefig(logdir + 'trn_tsne_hidden_out_Tstim1.png')
plt.show()

file.write('\n\nEnd of Program')
