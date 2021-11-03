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
    loc_sp_dist_shift = random.randint(-15,5)  
    shift_y = random.randint(-5,5)   ##    shift for x and y coordinates. To move the box around localization
    shift_x = random.randint(-5,5)   ##
   # shift_y = 0   ##      if shift is not requires. When we need to create patches 
   # shift_x = 0   ##
    y1 = int(dat[np.array(row)[0,0],0]-half_width-shift_y)   #   start y-coordinate of the box around localization
    y2 = int(dat[np.array(row)[0,0],0]+half_width-shift_y)   #   end y-coordinate of the box around localization
    x1 = int(dat[np.array(row)[0,0],1]-half_width-shift_x)   #   start x-coordinate of the box around localization
    x2 = int(dat[np.array(row)[0,0],1]+half_width-shift_x)   #   end y-coordinate of the box around localization
    ## coordinates for spectral part
    x = dat[np.array(row)[0,0],0]
    y = dat[np.array(row)[0,0],1]
    y11 = int(dat[np.array(row)[0,0],0]+loc_sp_distance+loc_sp_dist_shift-shift_y)    #   start y-coordinate of the box around spectra
    y21 = int(dat[np.array(row)[0,0],0]+loc_sp_distance+loc_sp_dist_shift + sp_image_width-shift_y) #   end y-coordinate of the box around spectra
    return(y1,y2,x1,x2,y11,y21,row,x,y,loc_sp_dist_shift, shift_y, shift_x)

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
    sp_cor = random.randint(-1,1)  ##    to introduce random shift for the box around spectral part
    frame = dat[:,2][row]
    if only_loc == True:
        patch = np.float64(stack[frame, x1:x2, y1:y2])
        patch = patch.reshape(row.shape[1],16,16)
    if only_spectra == True:
        patch = np.float64(stack[frame, x1:x2, y11:y21])
        patch = patch.reshape(row.shape[1],16,48)        
    if combined == True:
        patch = np.concatenate((np.float64(stack[frame, x1:x2, y1:y2]), 
                                np.float64(stack[frame, x1-2-sp_cor:x2-2-sp_cor, y11:y21])), axis=3)
        patch = patch.reshape(row.shape[1],16,64)
    return(patch, sp_cor)

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
'''
start_time = time.time()
print("--- %s seconds ---" % (time.time() - start_time))

### Downloading PeakFit file in csv format with deleted first 7 rows and # before Frame
data = pd.read_csv('/Users/hannamanko/Desktop/diff_st/My Library_2/U-net/patches/Gqaunt_100mW_9.csv')
data = pd.read_csv('D:/My Library_2/U-net/for figures/50mW_4_MMStack_Pos0.ome-2.tif.results.csv')
'''

# Reading the localisation file created in one of the next plugins (line , line)
#data = pd.read_csv('/Users/hannamanko/Desktop/diff_st/My Library_2/U-net/for figures/50mW_4_THSt_5.0.csv')

data = pd.read_csv('/Users/hannamanko/Desktop/NN_data/100mW_2_MMStack_Pos0.ome.tif.results.csv')

data = pd.read_csv('/Users/hannamanko/Desktop/diff_st/Our_Papers/Unet_SRsmlm/data/50mW_1_MMStack_Pos0.ome.tif.results.csv')

markerSize = 0.05   
####    localization file from ThunderSTORM (Fiji plugin)  ###
# ////_______
data = data[data["Intensity [photon]"]>10000]  ## you can filter the data by intensity to discard noise localizations   

plt.plot(data["x [nm]"], data["y [nm]" ], 'o', markersize=markerSize, color='red')

datat = pd.concat([pd.DataFrame(data["x [nm]"]),pd.DataFrame(data["y [nm]"]),pd.DataFrame(data["frame"])], axis = 1, ignore_index=False)
datat['y [nm]'] = (datat['y [nm]']/21.36752)/5
datat['x [nm]'] = (datat['x [nm]']/21.36752)/5
### \\\\_______
  

####    localization file from PeakFit (part of GDSC SMLM2 plugin in Fiji)  ###
# /////_____
data = data[data["Signal"]>5000]
plt.plot(data["origX"], data["origY" ], 'o', markersize=markerSize, color='red')
datat = pd.concat([pd.DataFrame(data["origX"]),pd.DataFrame(data["origY"]),pd.DataFrame(data["Frame"])], axis = 1, ignore_index=False)

### \\\\\_______

data_n= datat.to_numpy()
d = data_n[(data_n[:,0]>10)&(data_n[:,0]<220)&(data_n[:,1]>10)&(data_n[:,1]<250)]
plt.plot(d[:,0], d[:,1], 'o', markersize=markerSize, color='red')

data = data[(data['origX']>10)&(data['origX']<220)&(data['origY']>10)&(data['origY']<250)]
data = data.reset_index()


#####
##############################################################################
#####       Next we cluster the localizations on consecutive frames to create 'clean' images from the same localization
##          For this we are using DBSCAN

cl = DBSCAN(eps=0.001,min_samples=3)    # clustering   ## eps need to be chosed depending on the data (see line )
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
cluster = cluster_2
del cluster_2, clusters_2  # deleting additional variables 

for clust in clusters:         ##  Using this loop we can plot data by coloring different clusters
	row_ix = where(cluster == clust)
	plt.scatter(dat[row_ix, 0], dat[row_ix, 1], s=markerSize)
plt.show()

data['clusterID'] = pd.DataFrame(cluster)


#  Reading the original stack of aquired images
#stack = imread("/Users/hannamanko/Desktop/diff_st/My Library_2/U-net/for figures/50mW_4_MMStack_Pos0.ome-1.tif")

stack = imread("/Users/hannamanko/Desktop/NN_data/100mW_2/100mW_2_MMStack_Pos0.ome.tif")

stack = imread("/Users/hannamanko/Desktop/diff_st/Our_Papers/Unet_SRsmlm/data/150mW_2/150mW_2_MMStack_Pos0.ome.tif")
stack = stack/60000

start_time = time.time()
print("--- %s seconds ---" ,(time.time() - start_time))   # printing start time
### Creating the training set 
########## 
im_shape = image_type(image_height, sp_image_width,loc_image_width)   # Reading image shape
count = 0   
print("Please, wait until the end")
start_time = time.time()
SF_p_list = []
SF_sum_list = []
loc_sp_dist_shift_list = []
shift_y_list = []
shift_x_list = []
sp_cor_list = []
data_v = pd.DataFrame()
for i in clusters:  
    #print('i = ',i)    #  for each cluster index
    y1,y2,x1,x2,y11,y21,row,x,y,loc_sp_dist_shift,shift_y,shift_x = set_coordinates(i,half_width,loc_sp_distance, sp_image_width)  #  defining the cordinates of the box
    patch_x = np.zeros((np.shape(row[0])[0], im_shape[0], im_shape[1]))  # creating the empty patch of defined size
    try:
        patch, sp_cor = patches_formation(row, stack) 
    except:
        continue 
    patch_x = patch
    pat_s = np.zeros(patch_x.shape)
    SF_p = []
    SF_sum = []
    pat_s = Parallel(n_jobs=8)(delayed(SF_part)(n, patch_x, pat_s) for n in range(0, len(patch_x)))  # creating 'clean' images
    pat_s = np.array(pat_s).reshape(patch_x.shape)  # reshaping 
    SF_sum = [SF_calculator(pat_s[n]) for n in range(0, len(pat_s))]   # saving calculated SF value
    SF_p = [SF_calculator(patch_x[n]) for n in range(0, len(patch_x))]  # calculating SF value for one excluded pacth
    ind = where(np.array(SF_sum)/np.array(SF_p) > ((np.array(SF_sum)/np.array(SF_p)).max()-    #
                                                   (np.array(SF_sum)/np.array(SF_p)).min())/2) #    
    patch_ = patch_x[ind]
    patch_sum  = pat_s[ind]
    SF_p_list.extend(np.array(SF_p)[ind[0]])
    SF_sum_list.extend(np.array(SF_sum)[ind[0]])
    loc_sp_dist_shift_list.extend([loc_sp_dist_shift]*len(ind[0]))
    shift_y_list.extend([shift_y]*len(ind[0]))
    shift_x_list.extend([shift_x]*len(ind[0]))
    sp_cor_list.extend([sp_cor]*len(ind[0]))
    data_v = data_v.append(data.iloc[list(row[0][np.array(ind)][0])], ignore_index = True)
    if count == 0:                                                     #   gethering all the patches in one big set
        patch__ = patch_                                               #   set of original images      
        patch_sum_  = patch_sum                                        #   set of created 'clean' images
    else:                                                              #    
        patch__ = np.concatenate((patch__, patch_), axis=0)            #   set of original images 
        patch_sum_ = np.concatenate((patch_sum_, patch_sum), axis=0)   #   set of created 'clean' images
    count = count+1
    print(count)
print("--- %s seconds ---" % (time.time() - start_time))    #  printing  end time 


data_v['SF_raw'] = SF_p_list
data_v['SF_clean'] = SF_sum_list
data_v['shift_x'] = shift_x_list
data_v['shift_y'] = shift_y_list
data_v['loc_sp_dist_shift'] = loc_sp_dist_shift_list
data_v['sp_cor'] = sp_cor_list

patch__v = patch__.reshape(len(patch__),64*16)
patch_sum_v = patch_sum_.reshape(len(patch_sum_),64*16)

data_vv = pd.concat((data_v,pd.DataFrame(patch__v), pd.DataFrame(patch_sum_v)), axis = 1)

###  This is additional filtering
##   To discard patches with too low intensity in localisation part,
#    These are mostly kind of noisy patches
ind = []
for i in range(0, len(patch__)):
    if patch__[i,:16,:].max() <0.09:   
        ind.append(i)
patch___ = np.delete(patch__, [ind], axis=0)
patch_sum__ = np.delete(patch_sum_, [ind], axis=0)
data_vv = data_vv.drop(data_vv.index[ind])


data_vv.to_csv('/Users/hannamanko/Desktop/NN_data/data_100_2.csv', sep = ';', decimal='.', index = False,encoding="utf-8")


dat = pd.read_csv('/Users/hannamanko/Desktop/NN_data/data_100_2.csv')
####   Also patches can be saved as *.tif files
imsave('/Users/hannamanko/Desktop/NN_data/25_2_patch_sum.tif', patch_sum__)
imsave('/Users/hannamanko/Desktop/NN_data/25_2_patch.tif', patch___) 

#////

'''
pred = imread('/Users/hannamanko/Desktop/NN_data/150_2_pred_25.tif')
data = pd.read_csv('/Users/hannamanko/Desktop/diff_st/Our_Papers/Unet_SRsmlm/data/data_pred_150_2.csv', sep = ';', decimal='.')

data_ = data.drop(data.columns[list(range(16,data.shape[1]))], axis=1)
pred_v = pred.reshape(len(pred),64*16)
data_ = pd.concat((data_,pd.DataFrame(pred_v)), axis = 1)

data_.to_csv('/Users/hannamanko/Desktop/NN_data/data_pred_150_2.csv', sep = ';', decimal='.', index = False,encoding="utf-8")
'''
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
        print('x1 = '+str(x1_list[i])+ 'y1 = '+str(y1_list[i]) )
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


#################################
#########   to create noisy patches
#################################
#  We can take any of the stacks used to create patches for the training set
stack = imread('D:/1/Data/GattaQuant/Gqaunt_ Fluoro150mW_sPAINT__11/Gqaunt_ Fluoro150mW_sPAINT__11_MMStack_Pos0.ome.tif')
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


###################################################
#################     To draw rectangles 
###################################################

from matplotlib.patches import Rectangle
d= imread('D:/1/Data/GattaQuant/Gqaunt_ Fluoro75mW_sPAINT__7/Gqaunt_ Fluoro75mW_sPAINT__7_MMStack_Pos0.ome.tif')


loc = pd.read_csv('C:/Users/hmanko/Desktop/1.csv')
dd = d[64]

cord = pd.concat((loc['origX'], loc['origY']), axis = 1, ignore_index=False)
plt.scatter(loc['origX'], loc['origY'])

y1 = cord.iloc[1,1]-half_width
x1 = cord.iloc[1,0]-half_width

plt.imshow(dd)
ax = plt.gca()

# Create a Rectangle patch
rect = Rectangle((x1,y1),16,16,linewidth=1,edgecolor='r',facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect)

for i in range(15,25):
    y1 = cord.iloc[i,1]-half_width
    x1 = cord.iloc[i,0]-half_width
    y11 = cord.iloc[i,0]+loc_sp_distance
    plt.imshow(dd)
    plt.gca().add_patch(Rectangle((x1,y1),16,16,linewidth=1,edgecolor='r',facecolor='none'))
    plt.gca().add_patch(Rectangle((y11,y1),48,16,linewidth=1,edgecolor='green',facecolor='none'))
plt.show()





 
##############################   
one_p = patch__ 
one_ps = patch_sum_ 
 

patch__ = np.concatenate((one_p,two_p), axis=0)
   # one_p[:1500],two_p[:1500],tre_p[:1500], four_p[:1500], five_p[:1500],six_p[:1500], sev_p[:1500]), axis=0)
patch_sum_ = np.concatenate((one_ps,two_ps), axis=0)

plot_value(86)
##########     //%% ploting and discarding patches with shitty profile   %%\\
#########    DEpending on kind of images this step need to be adjusted
spectra = patch__
image_n_list = []
y = list(range(0,spectra.shape[2]))
for n in range(0,len(spectra)):
    sum_sp = np.sum(spectra[n,3:12], axis=0)
    fitter = modeling.fitting.LevMarLSQFitter()
    model = Gaussian1D(100,0.1,0.1)   # depending on the data you need to give some initial values
    fitted_model = fitter(model, y, sum_sp)
    #print("amp: ",fitted_model.amplitude.value, "mean: ", fitted_model.mean.value, "stdev: ",fitted_model.stddev.value, "__",n)
    if int(fitted_model.mean.value) not in range(10,48):
        image_n_list.append(n)
        #print("da", n)
spectra_f = np.delete(spectra, image_n_list, 0)    
patch__ = np.delete(patch__, image_n_list,0)
patch_sum_ = np.delete(patch_sum_, image_n_list,0)

imsave('/Users/hannamanko/Desktop/spectra.tif', spectra)

####### Lets try to threshold a bit ?
#
im = imread('D:/My Library/U-net/patches/yy/patch_sp_sum_5000.tif')

num = random.randint(0,im.shape[0])
plt.imshow(im[num,:,:])
img = im[num,:,:]

plt.hist(img)
imax = img.max()
imin = img.min()
for i in range(0, img.shape[0]):
    for j in range(0, img.shape[1]-1):
        if img[i,j] > imax-(imax/5):
            print(i, j)
            img[i,j] = img[i,j]*1.2
plt.imshow(img)

plt.plot(img[7,:])


#################################
#########   to create noisy patches
#################################

stack = imread('D:/1/Data/GataQuand__15-150mW/75mW_2/75mW_2_MMStack_Pos0.ome.tif')
stack = stack/stack.max()

     

count = 0
for i in range(0, len(patch)):   
    pat_s = SF_Image(np.concatenate((patch[i].reshape(1, 16, 64), patch[i].reshape(1, 16, 64)), axis = 0),combined=True)
    patch_sum  = pat_s
    if count == 0:
        patch_sum_  = patch_sum.reshape(1,16,64)
    else:
        patch_sum_ = np.concatenate((patch_sum_, patch_sum.reshape(1,16,64)), axis=0)   
    count = count+1
    
pp = np.concatenate((patch__, pp), axis = 0)  
pss = np.concatenate((patch_sum_, pss), axis=0)
patch__ = np.concatenate((patch__, patch_), axis = 0)
patch_sum_ = np.concatenate((patch_sum, patch_sum_), axis=0)

patch_sum = patch_sum_
patch_ = patch__


imsave('D:/My Library_2/U-net/For_Network/yy/Noise3.tif', patch_sum_)
imsave('D:/My Library_2/U-net/For_Network/xx/Noise3.tif', patch__)

patch_sum_ = np.concatenate((ps,ps2,ps3), axis=0)
patch__ = np.concatenate((p,p2,p3), axis=0)

noise_y = imread('D:/My Library_2/U-net/patches/yy/Noise.tif')
noise_x = imread('D:/My Library_2/U-net/patches/xx/Noise.tif')


###################################################
#################     To draw rectangles 
###################################################

from matplotlib.patches import Rectangle
d= imread('D:/1/Data/GattaQuant/Gqaunt_ Fluoro75mW_sPAINT__7/Gqaunt_ Fluoro75mW_sPAINT__7_MMStack_Pos0.ome.tif')


loc = pd.read_csv('C:/Users/hmanko/Desktop/1.csv')
dd = d[64]

cord = pd.concat((loc['origX'], loc['origY']), axis = 1, ignore_index=False)
plt.scatter(loc['origX'], loc['origY'])

y1 = cord.iloc[1,1]-half_width
x1 = cord.iloc[1,0]-half_width

plt.imshow(dd)
ax = plt.gca()

# Create a Rectangle patch
rect = Rectangle((x1,y1),16,16,linewidth=1,edgecolor='r',facecolor='none')

# Add the patch to the Axes
ax.add_patch(rect)

for i in range(15,25):
    y1 = cord.iloc[i,1]-half_width
    x1 = cord.iloc[i,0]-half_width
    y11 = cord.iloc[i,0]+loc_sp_distance
    plt.imshow(dd)
    plt.gca().add_patch(Rectangle((x1,y1),16,16,linewidth=1,edgecolor='r',facecolor='none'))
    plt.gca().add_patch(Rectangle((y11,y1),48,16,linewidth=1,edgecolor='green',facecolor='none'))
plt.show()

def consecutive(data, stepsize=1):
    return np.split(data, np.where(np.diff(data) != stepsize)[0]+1)


