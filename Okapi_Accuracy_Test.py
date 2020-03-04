#!/usr/bin/env python
# coding: utf-8

# In[12]:


import numpy as np
import csv #for loading


# In[13]:


def load_csv(filename):
    file = open(filename, 'r')
    lines = csv.reader(file)
    dataset = list(lines)
    return dataset


# In[14]:


#import as numpy array which is really useful for matrix multiplication, etc.
predictions = np.array(load_csv('predictions.csv'), dtype=np.float64)
last20 = predictions[:, -20:]
first20 = predictions[:, :20]
predictions = np.append(first20, last20, axis=1)
print "Finished loading non-pca predictions!"
    
pca_predictions = np.array(load_csv('predict_test_new_pc_coeff.csv'), dtype=np.float64)
print "Finished loading pca predictions!"
    
methylation_data = np.array(load_csv('methylation_data_no_labels.csv'), dtype=np.float64)
last20 = methylation_data[:, -20:]
first20 = methylation_data[:, :20]
methylation_data = np.append(first20, last20, axis=1)
print "Finished loading methylation data!"


# In[7]:


# Accuracy tests for non-PCA predictions

difference_no_pca = np.absolute(np.subtract(methylation_data, predictions))

num_flags_10 = 0
num_flags_5 = 0
    
for i in range(difference_no_pca.shape[0]): # number of rows
    for j in range(difference_no_pca.shape[1]): # number of cols
        if difference_no_pca[i][j] > 0.1: # 0.1 methylation margin of error
            num_flags_10 += 1
        if difference_no_pca[i][j] > 0.05: # 0.05 methylation margin of error
            num_flags_5 += 1            
    if i%100000 == 0:
        print i
        
print (1 - num_flags_10/float(difference_no_pca.shape[0]*difference_no_pca.shape[1]))
print (1 - num_flags_5/float(difference_no_pca.shape[0]*difference_no_pca.shape[1]))


# In[15]:


# Accuracy tests for PCA predictions

difference_pca = np.absolute(np.subtract(methylation_data, pca_predictions))

num_flags_10 = 0
num_flags_5 = 0
    
for i in range(difference_pca.shape[0]): # number of rows
    for j in range(difference_pca.shape[1]): # number of cols
        if difference_pca[i][j] > 0.1: # 0.1 methylation margin of error
            num_flags_10 += 1
        if difference_pca[i][j] > 0.05: # 0.05 methylation margin of error
            num_flags_5 += 1            
    if i%100000 == 0:
        print i
        
print (1 - num_flags_10/float(difference_pca.shape[0]*difference_pca.shape[1]))
print (1 - num_flags_5/float(difference_pca.shape[0]*difference_pca.shape[1]))


# In[8]:


errors_by_site_no_pca = np.mean(difference_no_pca, axis=1)
print errors_by_site_no_pca[:10]


# In[16]:


errors_by_site_pca = np.mean(difference_pca, axis=1)
print errors_by_site_pca[:10]


# In[17]:


chromosomes = np.array(load_csv('chromosome_col.csv'), dtype=np.int)


# In[11]:


import matplotlib.pyplot as plt

col = np.where(chromosomes%2==0, 'darkblue', 'black')
col = np.squeeze(col)
plt.figure(figsize=(20,10))
plt.margins(0,0)
plt.ylim((0,0.3))
plt.xlim((0,400000))
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.title('Average Error by Methylation Site, color seperated by chromosome (no PCA)', fontsize=30)
plt.ylabel('Average Error', fontsize=20)
plt.xlabel('Methylation Site', fontsize=20)
plt.scatter(range(errors_by_site_no_pca.shape[0]), errors_by_site_no_pca, s=5, marker='o', alpha=0.2, c=col)


# In[18]:


import matplotlib.pyplot as plt

col = np.where(chromosomes%2==0, 'darkblue', 'black')
col = np.squeeze(col)
plt.figure(figsize=(20,10))
plt.margins(0,0)
plt.ylim((0,0.3))
plt.xlim((0,400000))
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.title('Average Error by Methylation Site, color seperated by chromosome (PCA)', fontsize=30)
plt.ylabel('Average Error', fontsize=20)
plt.xlabel('Methylation Site', fontsize=20)
plt.scatter(range(errors_by_site_pca.shape[0]), errors_by_site_pca, s=5, marker='o', alpha=0.2, c=col)


# In[ ]:




