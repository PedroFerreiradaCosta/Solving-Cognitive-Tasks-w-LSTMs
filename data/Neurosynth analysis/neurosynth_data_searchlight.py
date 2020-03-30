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
motor = np.array(motor.dataobj)# s

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
model = pd.DataFrame(model)	#transform it into DataFrame type to correlate

# Initialize an array to add up the correlations
brain_corr = np.zeros((91,109,91))
#Initialize an array to count how many times region wass scoped for correlations
brain_count = np.zeros((91,109,91))



######################
#      SETTINGS      #
######################

# Steps taken per iteration
stride = 1

# X size
dim1 = 10
# Y size
dim2 = 5
# Z size
dim3 = 10

i = 0
j = 0
k = 0


# Anti Task Proxys
antirt = switch + switch2
# Memory Go Tasks Task Proxys
dly = wm + wm2
# Reaction Go Task Proxys
rt = react + react2
# Go Task Proxys
go = finger + motor


marks = []



while i+dim1 < go.shape[0]:

	while j+dim2 < go.shape[1]:

		while k+dim3 < go.shape[2]:


			go_i = go[i:i+dim1, j:j+dim2, k:k+dim3]
			rt_i = rt[i:i+dim1, j:j+dim2, k:k+dim3]
			mem_1_i = dly[i:i+dim1, j:j+dim2, k:k+dim3]
			antirt_i = antirt[i:i+dim1, j:j+dim2, k:k+dim3]
			inco_i = incongruent[i:i+dim1, j:j+dim2, k:k+dim3]
			nogo_i = nogo[i:i+dim1, j:j+dim2, k:k+dim3]


			go_i = np.reshape(go_i,(go_i.shape[0]*go_i.shape[1]*go_i.shape[2]))
			rt_i = np.reshape(rt_i,(rt_i.shape[0]*rt_i.shape[1]*rt_i.shape[2]))
			mem_1_i = np.reshape(mem_1_i,(mem_1_i.shape[0]*mem_1_i.shape[1]*mem_1_i.shape[2]))
			antirt_i = np.reshape(antirt_i,(antirt_i.shape[0]*antirt_i.shape[1]*antirt_i.shape[2]))
			inco_i = np.reshape(inco_i,(inco_i.shape[0]*inco_i.shape[1]*inco_i.shape[2]))
			nogo_i = np.reshape(nogo_i,(nogo_i.shape[0]*nogo_i.shape[1]*nogo_i.shape[2]))

			brain_count[i:i+dim1, j:j+dim2, k:k+dim3] += 1

			df1 = np.transpose([go_i, mem_1_i, rt_i, nogo_i, inco_i, antirt_i])
			#[go, dly_go, rt_go, anti, dly_anti, rt_anti]
			df1 = pd.DataFrame(df1)
			corr = df1.corr()
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
			corr2 = df2[0].corr(df2[1])
			corr2 = np.arctanh(corr2) # Fischer's Z Transform
			brain_corr[i:i+dim1, j:j+dim2, k:k+dim3] += corr2


			#print('Correlation of Correlations: ' + str(corr2))
			#print('*'*70)
			"""
			if corr2 > 0.9:

				mark = {
				'correlation':corr2,
				'i':i,
				'j':j,
				'k':k,
				'l':l,
				'm':m,
				'n':n,
				'o':o}
				#print(mark)
				marks.append(mark)
			"""
			k = k + stride
		k = 0
		j = j + stride
	j = 0
	i = i + stride
i = 0




# The desired image is obtained by dividing the correlation cube by the number
# of times the correlation was executed in a given region of the brain
#print(np.where(brain_count == 0))
brain_final = brain_corr / (brain_count + epsilon)
np.set_printoptions(threshold=np.nan)
#print(brain_final[90,:,:])
#print(in_ctrl[2][(np.where(brain_count==0))])

new_image = nib.Nifti1Image(brain_final, affine=affine)

# We can adjust the threshold for the plotted image (threshold = x)
plot_stat_map(new_image, output_file = None, colorbar = True, threshold = 'auto')
plt.savefig('fullimage_dim_10_thrshld_auto_stride_1.pdf')
nib.save(new_image, 'fullimage_dim_10_stride_1_brain_correlation.nii')
