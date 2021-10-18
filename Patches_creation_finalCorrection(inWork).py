#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Fri Jan 22 16:14:36 2021
https://github.com/hmanko/NN_sr
@author: hannamanko
"""
#####################  Import the required libraries  ###################
import numpy as np
import pandas as pd
from tifffile import imread, imsave
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from numpy import unique
from numpy import where
import time
import math
import random
from skimage.morphology import disk
from scipy.ndimage.morphology import white_tophat
from joblib import Parallel, delayed

### Defining Patches Size
image_height = 16  ## image height in pixels
loc_image_width = 16  ## image width in pixels
image_chanels = 1   ##
loc_sp_distance = 230  ## distance between localization and its spectral peak
                        #   measured from original image / in pixels
sp_image_width = 48 #### spectral image width
half_width = int(loc_image_width/2)    ##  

##################### DEFINING THE  FUNCTIONS ###################
###
##### #######    Useful  function to plot on one figure multiple patchs
def show_images(patches):
    n = int(math.sqrt(len(patches))+2)
    m = int(math.sqrt(len(patches)))
    plt.figure(figsize = (50,40))
    for i in range (0, len(patches)):
        plt.subplot(n,m,i+1)
        plt.title(i, fontsize=25)
        plt.imshow(patches[i])
    plt.show()

#######
def set_coordinates(i, half_width, loc_sp_distance, sp_image_width):
    row = np.array(where(cluster == i))
    ## coordinates for localization part
    #shift_y = random.randint(-6,6)   ##    shift for x and y coordinates. To move the box around localization
   # shift_x = random.randint(-6,6)   ##
    shift_y = 0   ##      if shift is not requires. When we need to create patches 
    shift_x = 0   ##
    y1 = int(dat[np.array(row)[0,0],0]-half_width-shift_y)   #   start y-coordinate of the box around localization
    y2 = int(dat[np.array(row)[0,0],0]+half_width-shift_y)   #   end y-coordinate of the box around localization
    x1 = int(dat[np.array(row)[0,0],1]-half_width-shift_x)   #   start x-coordinate of the box around localization
    x2 = int(dat[np.array(row)[0,0],1]+half_width-shift_x)   #   end y-coordinate of the box around localization
    ## coordinates for spectral part
    y11 = int(dat[np.array(row)[0,0],0]+loc_sp_distance-shift_y)    #   start y-coordinate of the box around spectra
    y21 = int(dat[np.array(row)[0,0],0]+loc_sp_distance + sp_image_width-shift_y) #   end y-coordinate of the box around spectra
    return(y1,y2,x1,x2,y11,y21,row)

####   The code was first writen to be able to create patches of different types
##   Function that returns image shape depending on patch type chosen
def image_type(image_height, sp_image_width,loc_image_width): 
    if only_loc == True:                                       ##  to have only localization part
        im_shape = (image_height, loc_image_width)             
    if only_spectra == True:                                   ##  to have only spectral part
        im_shape = (image_height, sp_image_width)     
    if combined == True:                                       ##  to have both localization and spectral parts
        im_shape = (image_height, sp_image_width+loc_image_width)
    return (im_shape)

#   Function that forms the patches depending on patch type chosen
sp_cor = 0   ###  the number of pixels in x-axis to corect position of the box around spectral part 
def patches_formation(row, stack):
    #sp_cor = random.randint(-2,8)  ##    to introduce random shift for the box around spectral part
    frame = dat[:,2][row]
    if only_loc == True:
        patch = np.float64(stack[frame, x1:x2, y1:y2])
        patch = patch.reshape(row.shape[1],16,16)
    if only_spectra == True:
        patch = np.float64(stack[frame, x1:x2, y11:y21])
        patch = patch.reshape(row.shape[1],16,48)        
    if combined == True:
        patch = np.concatenate((np.float64(stack[frame, x1:x2, y1:y2]), 
                                np.float64(stack[frame, x1-sp_cor:x2-sp_cor, y11:y21])), axis=3)
        patch = patch.reshape(row.shape[1],16,64)
    return(patch)

#######   Function to calculate Spatial Frecuency as described in (Shutao Li et al. 
# 'Combination of images with diverse focuses using the spatial frequency',  Information Fusion
# https://doi.org/10.1016/S1566-2535(01)00038-0.
def SF_calculator(patch):
    MN = image_height*sp_image_width
    rf = np.sqrt((1/MN)*np.sum([np.abs((patch[:,n]-patch[:,n-1])**2) for n in range(1,sp_image_width-1)]))
    cf = np.sqrt((1/MN)*np.sum([np.abs((patch[m,:]-patch[m-1,:])**2) for m in range(1,image_height-1)]))
    SF = np.sqrt(rf**2 + cf**2)
    return SF     

# Two functions to denoise spectral and localization  part
selem = disk(10)
def d_spectra(img, tempIm, ampFact = .5):   ## ampFact is introduced to increase signal for the spectral part  
    imgMean1 = tempIm[0, 0:img.shape[2]]/(ampFact*tempIm[0:img.shape[1],0:img.shape[2]].max())
    imgMean1 = white_tophat(imgMean1, footprint =  selem)
    return(imgMean1)
def d_loc(img, tempIm):
    imgMean2 = tempIm[0, 0:img.shape[2]]/tempIm[0:img.shape[1],0:img.shape[2]].max()
    imgMean2 = white_tophat(imgMean2, footprint = selem)
    return(imgMean2)

#######    /// Spatial Frequency calculation and image denoisin  \\\
def SF_Image(patch, only_loc=False, only_spectra=False, combined=False, denoising = True):
    SF=[]
    SF_sum = 0
    cumIm = np.zeros((patch[1:].shape))
    for i in range(0,len(patch[1:])):
        SF.append(SF_calculator(patch[i]))
        cumIm = cumIm + patch[i]*SF[i]
    SF_sum = np.sum(SF)
    tempIm = cumIm/SF_sum
    if denoising ==True:
        if only_spectra == True:
            imgMean = d_spectra(patch[:,:,loc_image_width:], tempIm[:,:,loc_image_width:])
        elif only_loc == True:
            imgMean = d_loc(patch[:,:,0:loc_image_width], tempIm[:,:,0:loc_image_width])
        elif combined == True:
            imgMean = np.concatenate((d_loc(patch[:,:,0:loc_image_width], 
                                            tempIm[:,:,0:loc_image_width]),(d_spectra(patch[:,:,loc_image_width:], 
                                                                                       tempIm[:,:,loc_image_width:]))),axis=1)
        return(imgMean)
    else:
        return(tempIm)
####
#   function that exclude one frame and apply SF_Image function to all othe frames
def SF_part(n, patch_x, pat_s):
    fr = [*range(0,len(patch_x))]
    fr.pop(n)
    pat_s = SF_Image(patch_x[((fr)),:,:], combined=True)
    return (pat_s)    
 
    
############################################################### 3
############################################################### 2
############################################################### 1 ... 
##########################   ## Put the type of images that you want to obtain equal True
###///                          Here we will create images containing both spectral and localization parts
only_spectra = False
only_loc = False
combined = True
###\\\
##########################
# Reading the localisation file created in one of the next plugins (line , line)
data = pd.read_csv('D:/1/Data/GataQuand__15-150mW/100mW_5_MMStack_Pos0.ome.tif.results.csv')

####    localization file from ThunderSTORM (Fiji plugin)  ###
# ////
data = data[data["Intensity [photon]"]>10000]  ## you can filter the data by intensity to discard noise localizations   
markerSize = 0.05   
plt.plot(data["x [nm]"], data["y [nm]" ], 'o', markersize=markerSize, color='red')

datat = pd.concat([pd.DataFrame(data["x [nm]"]),pd.DataFrame(data["y [nm]"]),pd.DataFrame(data["frame"])], axis = 1, ignore_index=False)
datat['y [nm]'] = (datat['y [nm]']/21.36752)/5
datat['x [nm]'] = (datat['x [nm]']/21.36752)/5
### \\\\
    
####    localization file from PeakFit (part of GDSC SMLM2 plugin in Fiji)  ###
# /////
data = data[data["Signal"]>50000]
plt.plot(data["origX"], data["origY" ], 'o', markersize=markerSize, color='red')
datat = pd.concat([pd.DataFrame(data["origX"]),pd.DataFrame(data["origY"]),pd.DataFrame(data["Frame"])], axis = 1, ignore_index=False)
data_n = datat.to_numpy()
### \\\\\

data_n= datat.to_numpy()
d = data_n[(data_n[:,0]>8)&(data_n[:,0]<220)&(data_n[:,1]>8)&(data_n[:,1]<200)]
plt.plot(d[:,0], d[:,1], 'o', markersize=markerSize, color='red')

#####
##############################################################################
#####       Next we cluster the localizations on consecutive frames to create 'clean' images from the same localization
##          For this we are using DBSCAN

cl = DBSCAN(eps=0.5,min_samples=3)    # clustering   ## eps need to be chosed depending on the data (see line )
cluster = cl.fit_predict(d[:, 0:2])   # table with indexes of all clusters
clusters = unique(cluster)            # tacking unique indexes of clusters 
clusters = clusters[clusters>-1]      # discarding noise
#---
def consecutive(data, stepsize=1):             # small function that helps to find consecutive frames for cluster
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)
#--- 
dat = pd.concat([pd.DataFrame(d), pd.DataFrame(cluster)], axis = 1)  # Adding cluster indexes as additional column to our data
dat = dat.to_numpy()  # Converting to numpy

cluster_2 = np.int64(np.full((cluster.shape), -1))  # Creation array full of '-1'
new_cluster = 0  
for i in clusters:
    ar = consecutive(dat[:,2][(where(dat[:,3] == i))])  # geting consecutive indexes in data for each cluster
    for j in ar:             # for ezch of the consecutive index sets
        if len(j) > 8:       # if this set is longer then 8
            for ii in range(0, len(j)):   
                cluster_2[where((dat[:,2]==j[ii])&(dat[:,3] == i))] = int(new_cluster)  # put new cluster index to cluster_2 
            print(new_cluster)
            new_cluster = new_cluster+1  # increase new_cluster index by 1
            
clusters_2 = unique(cluster_2)  #  again finding unique clusters
clusters_2 = clusters_2[clusters_2>-1]  # discarding the noise            
dat[:,3] = cluster_2   #  put cluster indexes as column 3 in data
clusters = clusters_2  #  rewrite clusters
del cluster_2, clusters_2  # deleting additional variables 

for clust in clusters:         ##  Using this loop we can plot data by coloring different clusters
	row_ix = where(cluster == clust)
	plt.scatter(dat[row_ix, 0], dat[row_ix, 1], s=markerSize)
plt.show()

#  Reading the original stack of aquired images
stack = imread('D:/1/Data/GataQuand__15-150mW/100mW_5/100mW_5_MMStack_Pos0.ome.tif')

start_time = time.time()
print("--- %s seconds ---" ,(time.time() - start_time))   # printing start time
### Creating the training set 
########## 
im_shape = image_type(image_height, sp_image_width,loc_image_width)   # Reading image shape
count = 0   
print("Please, wait until the end")
start_time = time.time()
for i in clusters:      #  for each cluster index
    y1,y2,x1,x2,y11,y21,row = set_coordinates(i,half_width,loc_sp_distance, sp_image_width)  #  defining the cordinates of the box
    patch_x = np.zeros((np.shape(row[0])[0], im_shape[0], im_shape[1]))  # creating the empty patch of defined size
    try:
        patch = patches_formation(row, stack)  #  forming the patches for one cluster (comsecutive localizations)
    except:
        patch.shape != im_shape  
    patch_x = patch
    pat_s = np.zeros(patch_x.shape)
    SF_p = []
    SF_sum = []
    pat_s = Parallel(n_jobs=5)(delayed(SF_part)(n, patch_x, pat_s) for n in range(0, len(patch_x)))  # creating 'clean' images
    pat_s = np.array(pat_s).reshape(patch_x.shape)  # reshaping 
    SF_sum = [SF_calculator(pat_s[n]) for n in range(0, len(pat_s))]   # saving calculated SF value
    SF_p = [SF_calculator(patch_x[n]) for n in range(0, len(patch_x))]  # calculating SF value for one excluded pacth
    ind = where(np.array(SF_sum)/np.array(SF_p) > ((np.array(SF_sum)/np.array(SF_p)).max()-    #
                                                   (np.array(SF_sum)/np.array(SF_p)).min())/2) #    
    patch_ = patch_x[ind]
    patch_sum  = pat_s[ind]
    if count == 0:                                                     #   gethering all the patches in one big set
        patch__ = patch_                                               #   set of original images      
        patch_sum_  = patch_sum                                        #   set of created 'clean' images
    else:                                                              #    
        patch__ = np.concatenate((patch__, patch_), axis=0)            #   set of original images 
        patch_sum_ = np.concatenate((patch_sum_, patch_sum), axis=0)   #   set of created 'clean' images
    count = count+1
    print(count)
print("--- %s seconds ---" % (time.time() - start_time))    #  printing  end time 

#  Now we can save created raw and 'clean' sets of patches
imsave('D:/My Library_2/U-net/For_Network/pstches_to_test_nn/100_5_patch_sum.tif', patch_sum_)
imsave('D:/My Library_2/U-net/For_Network/pstches_to_test_nn//100_5_patch.tif', patch__) 



#################################
#########   to create noisy patches
#################################
#  We can take any of the stacks used to create patches for the training set
stack = imread('D:/1/Data/GataQuand__15-150mW/100mW_1/100mW_1_MMStack_Pos0.ome.tif')
stack = stack/60000
#########    To create big set of noisy patches this proces need to be repeated several times 
#########    and obtained sets need to be concatenated 
###   :::::::::
im_shape = image_type(image_height, sp_image_width,loc_image_width)
count = 0
for i in range(0, len(stack)):
    for _ in range(0, 40):
            rand_x = random.randint(13,215)
            rand_y = random.randint(10,220)
            y1 = rand_y-half_width
            y2 = rand_y+half_width
            x1 = rand_x-half_width
            x2 = rand_x+half_width
            ## coordinates for spectral part
            y11 = rand_y+loc_sp_distance
            y21 = rand_y+loc_sp_distance + sp_image_width
            patch = np.concatenate((np.float64(stack[i, x1:x2, y1:y2]), 
                            np.float64(stack[i, x1:x2, y11:y21])), axis=1)
            try:
                patch = patch
            except:
                patch.shape != im_shape
            pat_s = SF_Image(np.concatenate((patch.reshape(1, 16, 64), patch.reshape(1, 16, 64)), axis = 0),combined=True)
            patch_ = np.concatenate((patch.reshape(1, 16, 64), patch.reshape(1, 16, 64)), axis = 0)
            patch_sum  = pat_s
            if count == 0:
                patch__ = patch_[0].reshape(1,16,64)
                patch_sum_  = patch_sum.reshape(1,16,64)
            else:
                patch__ = np.concatenate((patch__, patch_[0].reshape(1,16,64)), axis=0) 
                patch_sum_ = np.concatenate((patch_sum_, patch_sum.reshape(1,16,64)), axis=0)   
            count = count+1
            print(count)
                       
patch__ = imread('D:/My Library_2/U-net/For_Network/xx/Noise3.tif') 
patch_sum_ = imread('D:/My Library_2/U-net/For_Network/yy/Noise3.tif') 


#  Filtering and deleting the patches that randomly got localization
i_list = []
for i in range(0, len(patch_sum_)):
    if patch_sum_[i,:,:16].max() > 0.3:
        i_list.append(i)
patch_sum_ = np.delete(patch_sum_, [i_list], axis=0)
patch__ = np.delete(patch__, [i_list], axis=0)


###    To concattenate  several sets of patches:
p = patch__   #  writing set of raw pacthes to new variable
ps = patch_sum_  #  writing set of 'clean' pacthes to new variable
## SOmetimes to obtain required number of images in patch set 
patch__ = np.concatenate((patch__, p), axis = 0)      #  gathering all the patch sets
patch_sum_ = np.concatenate((patch_sum, ps), axis=0)

###   :::::::::
#########  



##############################         
##############################
###  Creating the same type of patches but only the 'raw', so basicaly we just cut the images in the correct form. 
###  These patches are created to be treated with our model and get images with high SNR at the output
##############################   
x1_list = []  #  to save original x-coordinates
y1_list = []  #  to save original y-coordinates
count = 0
for i in range(0,len(d)):
    y1 = int(d[i][:1][0])-half_width    
    y2 = y1+2*half_width
    x1 = int(d[i][1:2][0])-half_width
    x2 = x1+2*half_width
    ## coordinates for spectral part
    y11 = int(d[i][:1][0])+loc_sp_distance
    y21 = int(d[i][:1][0])+loc_sp_distance + sp_image_width
    try: 
        patch = np.concatenate(((stack[int(d[i][2]), x1:x2, y1:y2]), 
           (stack[int(d[i][2]), x1:x2, y11:y21])), axis=1)
        if (patch[8,1]+patch[8,8]+patch[8,15]) == 0:
            continue
        y1_list.append(d[i][1:2][0])
        x1_list.append(d[i][:1][0])
        patch = patch.reshape(1,16,64)
    except:
        continue
    if count == 0:
        patch_x = patch
    else:
        patch_x = np.concatenate((patch_x, patch), axis=0) 
    count = count+1
    print(count)
 

coordinates = np.zeros((len(x1_list),2))   # creatingg array
coordinates[:,0]= x1_list                  # adding x coordinates to 0 column
coordinates[:,1]= y1_list                  # adding y coordinates to 1st column
    
plt.scatter(coordinates[:,0],coordinates[:,1],s=markerSize,cmap=plt.cm.jet)# we can plot the coordinates to check if everything is ok
#  Saving the original coordinates as *.csv file and created pacth as *.tif
pd.DataFrame(coordinates).to_csv('/Users/hannamanko/Desktop/diff_st/My Library_2/U-net/for figures/coordinates_5.0.csv',index = False)
imsave('/Users/hannamanko/Desktop/diff_st/My Library_2/U-net/for figures/50mW_4_patch_m5.0.tif', patch_x)
##############################  


###################################################
#################     To draw rectangles 
###################################################

from matplotlib.patches import Rectangle
stack = imread('D:/1/Data/GattaQuant/Gqaunt_ Fluoro75mW_sPAINT__7/Gqaunt_ Fluoro75mW_sPAINT__7_MMStack_Pos0.ome.tif')


 
dd = stack[980]  # Choosing one frame from the images stack
plt.imshow(dd)   # showing this image
datt = data.iloc[where(data['Frame'] == 980)]  # choosing localizations only from this frame
datt = datt[datt['origX']<250]
datt = datt[datt['Signal']>60000]  # Filtering by intensity
 
cord = pd.concat((datt['origX'], datt['origY']), axis = 1, ignore_index=False)
#plt.plot(d[:,0], d[:,1], 'o',markersize = markerSize)

ax = plt.gca()

rect = Rectangle((x1,y1),16,16,linewidth=1,edgecolor='r',facecolor='none') # Creating a Rectangle 
# Add the patch to the Axes
ax.add_patch(rect)
for i in range(0, len(cord)):
    y1 = cord.iloc[i,1]-half_width
    x1 = cord.iloc[i,0]-half_width
    y11 = cord.iloc[i,0]+loc_sp_distance
    plt.imshow(dd)
    plt.gca().add_patch(Rectangle((x1,y1),16,16,linewidth=1,edgecolor='r',facecolor='none'))
    plt.gca().add_patch(Rectangle((y11,y1),48,16,linewidth=1,edgecolor='green',facecolor='none'))
plt.show()


