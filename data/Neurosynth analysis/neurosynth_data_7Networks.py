import os
import numpy as np
import nibabel as nib
from nilearn.plotting import plot_stat_map
import pandas as pd
import pylab as plt
import math

epsilon = 0.00001

###################################
#      Romy's Proposed Terms      #
###################################


react = nib.load('./Neurosynth/reaction time_pAgF_z.nii') # rt go
react = np.array(react.dataobj) #

react2 = nib.load('./Neurosynth/reaction times_pAgF_z.nii') # rt go
react2 = np.array(react2.dataobj) #

incongruent = nib.load('./Neurosynth/incongruent_pAgF_z.nii') # memory anti
incongruent = np.array(incongruent.dataobj) #

wm = nib.load('./Neurosynth/wm task_pAgF_z.nii')
wm = np.array(wm.dataobj) # Bad correlation with our tasks # memory go

wm2 = nib.load('./Neurosynth/working memory_pAgF_z.nii') # memory go
affine = wm2.affine # We save the affine matrix to use it to create similar nifty files
wm2 = np.array(wm2.dataobj)#

motor = nib.load('./Neurosynth/motor task_pAgF_z.nii') # go
motor = np.array(motor.dataobj)#

finger = nib.load('./Neurosynth/finger tapping_pAgF_z.nii')  #go
finger = np.array(finger.dataobj)#

switch = nib.load('./Neurosynth/switching_pAgF_z.nii') # anti rt
switch = np.array(switch.dataobj)#

switch2 = nib.load('./Neurosynth/switch_pAgF_z.nii') # anti rt
affine = switch2.affine
switch2 = np.array(switch2.dataobj)#

nogo = nib.load('./Neurosynth/nogo_pAgF_z.nii') # anti(?)
nogo = np.array(nogo.dataobj)#



#############################################
# LOADING CORRELATION MATRIX FROM OUR TASKS #
#############################################

model = np.loadtxt('task6_corr.csv', delimiter = ',')
model = model[np.triu_indices(6, k = 1)]	# extracting the upper triangle
#print(model)
model = np.reshape(model, (model.shape[0],1))
#model = pd.DataFrame(model)	#transform it into DataFrame type to correlate



###################################
#      YEO's MASKS - 7 NETS       #
###################################

mask = nib.load('./Yeo2011_7Networks_MNI152_FreeSurferConformed_2mm.nii')
mask = np.array(mask.dataobj)#

"""
We are able to try any combinations of the
desired terms by inputting them to the dictionary below

To add a new term, choose the compatible task and add
it to the dictionary.
"""

# Go Task Proxys
go = finger + motor
# Memory Go Tasks Task Proxys
dly = wm + wm2
# Reaction Go Task Proxys
rt = react + react2
# Anti Task
anti = nogo
#Anti Memory Task
antimem = incongruent
# Anti Task Proxys
antirt = switch + switch2




n_network = 7
corr2 = np.zeros(n_network)
count = 0
marks = []



for i in range(n_network):

	go_i = go[mask == i+1]
	mem_1_i = dly[mask == i+1]
	rt_i = rt[mask == i+1]
	nogo_i = anti[mask == i+1]
	inco_i = antimem[mask == i+1]
	antirt_i = antirt[mask == i+1]





	df1 = np.transpose([go_i, mem_1_i, rt_i, nogo_i, inco_i, antirt_i])
	#[go, dly_go, rt_go, anti, dly_anti, rt_anti]
	df1 = pd.DataFrame(df1)
	corr = df1.corr(method = 'pearson')
	corr = np.arctanh(corr) # Fischer's Z Transform



	#plt.matshow(corr)
	#plt.colorbar()
	#plt.show()

	corr = np.array(corr)
	corr = corr[np.triu_indices(6, k = 1)]	# extracting the upper triangle
	corr = np.reshape(corr,(corr.shape[0],1))
	#print(corr)


	#print('*'*70)
	frames = (np.concatenate((corr, model), axis = 1))
	df2 = pd.DataFrame(frames)
	temp_corr2 = df2[0].corr(df2[1], method = 'pearson')
	corr2[i] = np.arctanh(temp_corr2) # Fischer's Z Transform



print(corr2)
