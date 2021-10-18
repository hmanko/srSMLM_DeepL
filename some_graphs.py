# -*- coding: utf-8 -*-
"""
Created on Wed Sep  8 14:55:05 2021

@author: hmanko
"""


import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN
from matplotlib import pyplot as plt
from numpy import unique
from numpy import where
from tifffile import imread, imsave
import time
import math

import random




th = pd.read_csv('C:/Users/hmanko/Desktop/Gataq_50_raw.csv')
th = pd.read_csv('/Users/hannamanko/Desktop/diff_st/My Library_2/U-net/for figures/50mW_4_patch_m5.0.csv')


origI = pd.read_csv('/Users/hannamanko/Desktop/diff_st/My Library_2/U-net/for figures/gata_rule1.results.csv')
origI = origI[(origI['x [nm]']>5*21.39815*5)&(origI['x [nm]']<14*21.39815*5)&(origI['y [nm]']>9*21.39815*5)&(origI['y [nm]']<19*21.39815*5)]
origI = origI[origI['x [nm]']<20000]
plt.scatter(origI['x [nm]'], origI['y [nm]'], s = 0.05)
plt.xlim((0,1000))
plt.ylim((0,1000))

d = data_n[(data_n[:,0]>5)&(data_n[:,0]<14)&(data_n[:,1]>9)&(data_n[:,1]<19)]

g = th.copy()
g = g.sort_values('intensity [photon]').drop_duplicates('frame', keep='last')
g = g.sort_values('frame')
g = g.reset_index(drop=True)

plt.scatter(g['x [nm]'], g['y [nm]'], s = 0.1)

coordinates = pd.read_csv('/Users/hannamanko/Desktop/diff_st/My Library_2/U-net/for figures/coordinates_1_m5.0.csv')
th_p = pd.read_csv('/Users/hannamanko/Desktop/diff_st/My Library_2/U-net/for figures/gata_rule2_thST_pred_25.csv')
gp = th_p.copy()
gp = gp.sort_values('intensity [photon]').drop_duplicates('frame', keep='last')
gp = gp.sort_values('frame')
gp = gp.reset_index(drop=True)

markerSize = 0.1

gg = pd.DataFrame(np.zeros((g.shape[0], 2)),columns = ['x','y'])

ggp = pd.DataFrame(np.zeros((g.shape[0], 2)),columns = ['x','y'])
for frame in g['frame']:
    #print(frame)
    k = where(g['frame'] == frame)[0]
    try:
        gg['x'][k] = g['x [nm]'].iloc[k[0]] + coordinates['0'][k[0]]
        gg['y'][k] = g['y [nm]'].iloc[k[0]] + coordinates['1'][k[0]]
        print('gg = '+str(gg['x'].iloc[k[0]])+ ',  g = ' +str(g['x [nm]'].iloc[k[0]])+ 
              ', coordinates = '+str(coordinates['0'][k[0]])+',  original = ' +str(origI['x [nm]'].iloc[k[0]]))
    except:
        continue
 
gg = g.copy()    

for frame in g['frame']:
    #print(frame)
    k = where(g['frame'] == frame)[0]
    gg['x [nm]'][k] = (coordinates['0'][k[0]]*5)*21.36752
    gg['y [nm]'][k] = coordinates['1'][k[0]]*
  
plt.plot(gg['x [nm]'], gg['y [nm]'], 'o', markersize=markerSize, color = 'red')      


gg.to_csv('/Users/hannamanko/Desktop/diff_st/My Library_2/U-net/for figures/gg.csv')
    
 
    
ggp = ggp[ggp >0]
    
plt.plot(th['x [nm]'], th['y [nm]'], 'o', markersize=1)
plt.xlim((700, 1050))
plt.ylim((700, 1050))
plt.subplots(figsize = (20,14))
plt.plot(ggp['x'], ggp['y'], 'o', markersize=markerSize,color = 'black')  
plt.plot(gg['x'], gg['y'], 'o', markersize=markerSize, color = 'red')  

 
g['x [nm]'] = gg['x']
g['y [nm]'] = gg['y']

g.to_csv('/Users/hannamanko/Desktop/diff_st/My Library_2/U-net/for figures/reconstructed_ruler_patch_25.csv')

imsave('/Users/hannamanko/Desktop/diff_st/My Library_2/U-net/for figures/gata_rule2_patch.tif',
       imread('/Users/hannamanko/Desktop/diff_st/My Library_2/U-net/for figures/gata_rule2_patch.tif')/60000)


g.to_csv('C:/Users/hmanko/Desktop/gata_ruler_1_true_pred.csv')

plt.scatter( th['intensity [photon]'],th['uncertainty_xy [nm]'], s = 1.)


ph = ( 5.8*(g['intensity [photon]'] - g['bkgstd [photon]']))/(1*400*0.95)
php = ( 5.8*(gp['intensity [photon]'] - gp['bkgstd [photon]']))/(1*400*0.95)


plt.scatter(gp['uncertainty_xy [nm]'], php)
plt.scatter(g['uncertainty_xy [nm]'], ph)

#############################################################################################################
###################    Precision(photons)  graph
#############################################################################################################
dat = pd.read_csv('/Users/hannamanko/Desktop/diff_st/My Library_2/50_4_patch-1.tif.results.csv')
datP = pd.read_csv('/Users/hannamanko/Desktop/diff_st/My Library_2/50_4_pred_25_norm-1.tif.results.csv')

ph = ( 5.8*(dat['Signal'] - dat['Background']))/(1*400*0.95)
php = ( 5.8*(datP['Signal'] - datP['Background']))/(1*400*0.95)

plt.scatter(dat['Precision'], ph, s = 0.5)
plt.scatter(datP['Precision'], php, s = 0.5)

data = pd.DataFrame()
d = dat.copy()
d = d.sort_values('Signal').drop_duplicates('Frame', keep='last')
d = d.sort_values('Frame')
d = d.reset_index(drop=True)
dP = datP.copy()
dP = dP.sort_values('Signal').drop_duplicates('Frame', keep='last')
dP = dP.sort_values('Frame')
dP = dP.reset_index(drop=True)

precision_pred = []
precision = []
signal = []
backgr = []
for frame in d['Frame']:
    k = where(dP['Frame'] == frame)[0]
    kk = where(d['Frame'] == frame)[0]
    #print(k)
    try:
        precision_pred.append(dP['Precision'][[k][0][0]])
        precision.append(d['Precision'][[kk][0][0]])
        signal.append(d['Signal'][[kk][0][0]])
        backgr.append(d['Background'][[kk][0][0]])
    except:
        continue
 
photons = (5.8*(pd.DataFrame(signal) - pd.DataFrame(backgr)))/(1*400*0.95)
ph = (5.8*(d['Signal'] - d['Background']))/(1*400*0.95)

plt.subplots(figsize = (11,7))
r = plt.scatter(ph, d['Precision'],  s = 0.3, alpha = 0.5)
p = plt.scatter(photons,precision_pred,  s = 0.3, alpha = 0.5)
plt.xlabel('#photons')
plt.ylabel('Precision, ')
plt.legend((r,p), ('raw_data', 'Predicted_data'))


data['Precision'] = pd.DataFrame(precision)
data['Precision_pred'] = pd.DataFrame(precision_pred)
data['N_photons'] = pd.DataFrame(photons)
d['Signal'] = ph
count = 0
plt.subplots(figsize = (10,7))
for phot in range(int(photons.min()), int(photons.max()), 200):
    da = data[(data['N_photons']> phot)&(data['N_photons']<phot+200)]
    daa = d[(d['Signal']>phot)&(d['Signal']<phot+200)]
    plt.boxplot(daa['Precision'],patch_artist=True, positions = [count],widths = 1,boxprops=dict(facecolor='darkblue'))
    plt.boxplot(da['Precision_pred'],patch_artist=True, positions = [count+1],widths = 1,boxprops=dict(facecolor='darkgreen'))
    count = count+4



precision_pred[where()]












