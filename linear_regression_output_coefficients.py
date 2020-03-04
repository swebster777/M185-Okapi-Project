#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import csv
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Ridge


# In[2]:


def load_csv(filename):
    file = open(filename, 'r')
    lines = csv.reader(file)
    dataset = list(lines)
    return dataset


# In[3]:


#import methylation
#import meta data
meta_data = np.array(load_csv('meta_data_no_labels_6var.csv'), dtype=np.float64)[20:-20, :]
#transpose meta data
methylation_data = np.array(load_csv('methylation_data_no_labels.csv'), dtype=np.float64)[:, 20:-20]


# In[8]:


print meta_data[:20]
print methylation_data[:20]


# In[4]:


coefficient_matrix = []
    
for i in range(np.size(methylation_data[:,0])):
    y_true = methylation_data[i,].T
    reg = Ridge()
    reg = reg.fit(meta_data, y_true)
    intercept_and_coef = list(reg.coef_)
    intercept_and_coef.insert(0, float(reg.intercept_))
    coefficient_matrix.append(intercept_and_coef)
    if i%100000 == 0:
        print i


# In[5]:


with open("intercept_and_coefficients.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(coefficient_matrix)
