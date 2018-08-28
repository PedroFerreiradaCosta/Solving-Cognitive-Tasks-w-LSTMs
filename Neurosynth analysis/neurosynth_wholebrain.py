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

model = np.loadtxt('task_corr_Spearman.csv', delimiter = ',')
model = model[np.triu_indices(6, k = 1)]	# extracting the upper triangle
#print(model)
model = np.reshape(model, (model.shape[0],1))
#model = pd.DataFrame(model)	#transform it into DataFrame type to correlate
#np.savetxt('corr_model_Pearson_tri_7.csv', model)




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


n_network = 1
corr2 = np.zeros(n_network)
count = 0
marks = []
dist = []


set = [go, dly, rt, anti, antimem, antirt]
maximum = []
ind = []


go_i = set[0]
mem_1_i = set[1]
rt_i = set[2]
nogo_i = set[3]
inco_i = set[4]
antirt_i = set[5]




df1 = np.transpose([go_i.flatten(), mem_1_i.flatten(), rt_i.flatten(), nogo_i.flatten(), inco_i.flatten(), antirt_i.flatten()])

df1 = pd.DataFrame(df1)
corr = df1.corr(method = 'spearman')
corr = np.arctanh(corr) # Fischer's Z Transform

# Neurosynth correlation for the whole brain
corr = np.array(corr)
plt.imshow(corr)
plt.title('Spearman Correlation')
plt.colorbar()
plt.yticks([0,1,2,3,4,5],['go','dlygo','rtgo','anti','dlyanti','rtanti'])
plt.xticks([0,1,2,3,4,5],['go','dlygo','rtgo','anti','dlyanti','rtanti'], rotation = 'vertical')
plt.show()
corr = corr[np.triu_indices(6, k = 1)]	# extracting the upper triangle
corr = np.reshape(corr,(corr.shape[0],1))

# Correaltions of correlations
frames = (np.concatenate((corr, model), axis = 1))
df2 = pd.DataFrame(frames)
temp_corr2 = df2[0].corr(df2[1], method = 'spearman')
corr2 = np.arctanh(temp_corr2) # Fischer's Z Transform

print(corr2)
