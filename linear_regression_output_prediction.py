#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import csv


# In[2]:


def load_csv(filename):
    file = open(filename, 'r')
    lines = csv.reader(file)
    dataset = list(lines)
    return dataset


# In[4]:

#load meta data and coefficients
meta_data = np.array(load_csv('meta_data_no_labels_6var.csv'), dtype=np.float64)
coefficient_matrix = np.array(load_csv('intercept_and_coefficients.csv'), dtype=np.float64)


# In[7]:

predictions = []

for i in range(np.size(coefficient_matrix[:,0])):
    if i%100000 == 0: #progress report
        print i
    intercept = coefficient_matrix[i][0] #extract intercept and coefficients for each site
    co_female = coefficient_matrix[i][1]
    co_cancer = coefficient_matrix[i][2]
    co_male_no_cancer = coefficient_matrix[i][3]
    co_male_cancer = coefficient_matrix[i][4]
    co_female_no_cancer = coefficient_matrix[i][5]
    co_female_cancer = coefficient_matrix[i][6]
    predictions_for_site = []
    for k in range(np.size(meta_data[:,0])): #make predictions for each patients based on their meta data
        predictions_for_site.append(intercept + co_female*meta_data[k][0] + co_cancer*meta_data[k][1] + co_male_no_cancer*meta_data[k][2] + co_male_cancer*meta_data[k][3] + co_female_no_cancer*meta_data[k][4] + co_female_cancer*meta_data[k][5])
    predictions.append(predictions_for_site)                   
            
with open("predictions.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(predictions)
