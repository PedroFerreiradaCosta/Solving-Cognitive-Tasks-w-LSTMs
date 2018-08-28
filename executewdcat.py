# -*- coding: utf-8 -*-
import generate_trial as dt
import numpy as np
import pylab as plt
from keras.utils import to_categorical
from datetime import datetime
import os

task = 'dm'
trials = 500
fix = 100 # stimulus is presented for n iterations
nofix = 250
classes_in = 38
classes_out = 33
input = np.zeros((trials, (fix+nofix) * classes_in)) # lines - trials, columns - flatten input (iterations * units)
output = np.zeros((trials, fix + nofix)) # lines - trials, columns - flatten output (iterations * units)
SNR = np.zeros(trials)
duration = np.zeros(trials)

now = datetime.utcnow().strftime("%Y%m%d%H%M")
root_logdir = "./dataset"
logdir = "{}/run-{}/".format(root_logdir, now+task)

if not os.path.exists(logdir):
    os.makedirs(logdir)
    os.makedirs(logdir + "images")


for i in range(0,trials):
 input[i,:], output[i,:], SNR[i], duration[i] =  dt.decision_making(fix, i) #gotask receives (nr of stimulus iterations, input no, plot = False)

 plt.subplot(121)
 plt.imshow(np.reshape(input[i,:], (fix+nofix, classes_in)), cmap="viridis")
 plt.title('X')
 plt.subplot(122)
 plt.imshow(np.reshape(to_categorical(output[i,:], num_classes = classes_out),(fix+nofix,classes_out)),cmap="viridis")
 plt.title('Y')
 plt.colorbar()
 plt.savefig(logdir + 'images/trial_'+str(i)+'.png')
 plt.close()
 #print("Input: " + str(input[i,0:33]))
 #print("Ouput: " + str(output[i,:]))
 #print("Difficulty: " + str(difficulty[i]))
 #os.system("pause")
# saves a input, output per line of the matrix


#np.savetxt("input_go.csv", input ,delimiter=",",fmt="%1.2f") # save input
#np.savetxt("output_go.csv",output,delimiter=",",fmt="%1.2f") # save output
#np.savetxt("difficulty_go.csv",difficulty,delimiter=",", fmt="%.0f") # save output

train = int(trials*0.8) # 80% of the set is used for training. 20% for testing
input_tr = input[0:train,:]
output_tr = output[0:train,:]
SNR_tr = SNR[0:train]
duration_tr = duration[0:train]
input_tst = input[train:, :]
output_tst = output[train:, :]
SNR_tst = SNR[train:]
duration_tst = duration[train:]
stim_direction_trn = output_tr[:,-1]
stim_direction_tst = output_tst[:,-1]



file = open(logdir + "Data.txt", "w")
file.write('___________________________________\n')
file.write('___________________________________\n')
file.write("Number of trials: " + str(trials)+ '\n')
file.write("Number of training trials: " + str(train)+ '\n')
file.write("Number of timesteps: " + str(fix + nofix) + '\n')
file.write("Fixation Period: " + str(fix) + '\n')
file.write("Reaction Period: " + str(nofix) + '\n')
file.write("Number of classes input: " + str(classes_in)+ '\n')
file.write("Number of classes output: " + str(classes_out)+ '\n')
file.write('___________________________________' + '\n')
file.write('___________________________________'+ '\n')



np.savetxt("X_train_"+task+".csv", input_tr ,delimiter=",",fmt="%1.2f") # save input train
np.savetxt("Y_train_"+task+".csv",output_tr,delimiter=",",fmt="%1.2f") # save output train
np.savetxt("train_SNR_"+task+".csv",SNR_tr,delimiter=",", fmt="%1.1f") # save SNR of train set
np.savetxt("train_duration_"+task+".csv",duration_tr,delimiter=",", fmt="%1.1f") # save stim duration of train set
np.savetxt("X_test_"+task+".csv", input_tst ,delimiter=",",fmt="%1.2f")
np.savetxt("Y_test_"+task+".csv",output_tst,delimiter=",",fmt="%1.2f")
np.savetxt("test_SNR_"+task+".csv",SNR_tst,delimiter=",", fmt="%.1f")
np.savetxt("test_duration_"+task+".csv",duration_tst,delimiter=",", fmt="%.1f")
np.savetxt('stim_direction_trn_'+ task+ '.csv', stim_direction_trn, fmt="%i", delimiter = ',')
np.savetxt('stim_direction_tst_'+task+'.csv', stim_direction_tst,  fmt="%i", delimiter = ',')


np.savetxt(logdir + "X_train_"+task+".csv", input_tr ,delimiter=",",fmt="%1.2f") # save input train
np.savetxt(logdir + "Y_train_"+task+".csv",output_tr,delimiter=",",fmt="%1.2f") # save output train
np.savetxt("train_SNR_"+task+".csv",SNR_tr,delimiter=",", fmt="%1.1f") # save SNR of train set
np.savetxt("train_duration_"+task+".csv",duration_tr,delimiter=",", fmt="%1.1f") # save stim duration of train set
np.savetxt(logdir + "X_test_"+task+".csv", input_tst ,delimiter=",",fmt="%1.2f")
np.savetxt(logdir + "Y_test_"+task+".csv",output_tst,delimiter=",",fmt="%1.2f")
np.savetxt("test_SNR_"+task+".csv",SNR_tst,delimiter=",", fmt="%.1f")
np.savetxt("test_duration_"+task+".csv",duration_tst,delimiter=",", fmt="%.1f")
np.savetxt(logdir + 'stim_direction_trn.csv', stim_direction_trn, fmt="%i", delimiter = ',')
np.savetxt(logdir + 'stim_direction_tst.csv', stim_direction_tst,  fmt="%i", delimiter = ',')#for i in range(0,trials):
# input[i,:], output[i,:], difficulty[i] =  dt.antitask(250, i) # saves a input, output per line of the matrix


#np.savetxt("input_anti.csv", input ,delimiter=",",fmt="%1.2f") # save input
#np.savetxt("output_anti.csv",output,delimiter=",",fmt="%1.2f") # save output
#np.savetxt("difficulty_anti.csv",difficulty,delimiter=",", fmt="%.0f") # save output


#for i in range(0,trials):
# input[i,:], output[i,:], difficulty[i] =  dt.gonotask(250, i) # saves a input, output per line of the matrix


#np.savetxt("input_gono.csv", input ,delimiter=",",fmt="%1.2f") # save input
#np.savetxt("output_gono.csv",output,delimiter=",",fmt="%1.2f") # save output
#np.savetxt("difficulty_gono.csv",difficulty,delimiter=",", fmt="%.0f") # save output


file.write('\n\n End of Program')
