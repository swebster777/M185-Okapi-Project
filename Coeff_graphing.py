#!/usr/bin/env python
# coding: utf-8

# In[1]:


import numpy as np
import csv #for loading
import matplotlib.pyplot as plt


# In[2]:


def load_csv(filename):
    file = open(filename, 'r')
    lines = csv.reader(file)
    dataset = list(lines)
    return dataset


# In[3]:


#import as numpy array which is really useful for matrix multiplication, etc.
coeffs = np.array(load_csv('update_new_pc_coeff.csv'), dtype=np.float64)[:, :9]
coeffs = coeffs.T
print "Finished loading coefficients!"
    
chromosomes = np.array(load_csv('chromosome_col.csv'), dtype=np.int)
print "Finished loading chromosomes!"


# In[4]:


col = np.where(chromosomes%2==0, 'darkblue', 'black') #seperate chromosomes by color
col = np.squeeze(col)
plt.figure(figsize=(20,10))
plt.margins(0,0)
plt.ylim((0,1.0))
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.title('Linear Regression Intercepts', fontsize=30)
plt.ylabel('Intercept', fontsize=20)
plt.xlabel('Methylation Site', fontsize=20)
plt.scatter(range(coeffs.shape[1]), coeffs[0], s=5, marker='o', alpha=0.2, c=col)


# In[12]:


col = np.where(chromosomes%2==0, 'darkblue', 'black') #seperate chromosomes by color
col = np.squeeze(col)
plt.figure(figsize=(20,10))
plt.margins(0,0)
plt.ylim((-0.30,0.30))
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.title('Linear Regression Coefficients - Female', fontsize=30)
plt.ylabel('Coefficient', fontsize=20)
plt.xlabel('Methylation Site', fontsize=20)
plt.scatter(range(coeffs.shape[1]), coeffs[1], s=5, marker='o', alpha=0.2, c=col)


# In[13]:


col = np.where(chromosomes%2==0, 'darkblue', 'black') #seperate chromosomes by color
col = np.squeeze(col)
plt.figure(figsize=(20,10))
plt.margins(0,0)
plt.ylim((-0.30,0.30))
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.title('Linear Regression Coefficients - Cancer', fontsize=30)
plt.ylabel('Coefficient', fontsize=20)
plt.xlabel('Methylation Site', fontsize=20)
plt.scatter(range(coeffs.shape[1]), coeffs[2], s=5, marker='o', alpha=0.2, c=col)


# In[14]:


col = np.where(chromosomes%2==0, 'darkblue', 'black') #seperate chromosomes by color
col = np.squeeze(col)
plt.figure(figsize=(20,10))
plt.margins(0,0)
plt.ylim((-0.30,0.30))
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.title('Linear Regression Coefficients - Female without Cancer', fontsize=30)
plt.ylabel('Coefficient', fontsize=20)
plt.xlabel('Methylation Site', fontsize=20)
plt.scatter(range(coeffs.shape[1]), coeffs[3], s=5, marker='o', alpha=0.2, c=col)


# In[15]:


col = np.where(chromosomes%2==0, 'darkblue', 'black') #seperate chromosomes by color
col = np.squeeze(col)
plt.figure(figsize=(20,10))
plt.margins(0,0)
plt.ylim((-0.30,0.30))
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.title('Linear Regression Coefficients - Female with Cancer', fontsize=30)
plt.ylabel('Coefficient', fontsize=20)
plt.xlabel('Methylation Site', fontsize=20)
plt.scatter(range(coeffs.shape[1]), coeffs[4], s=5, marker='o', alpha=0.2, c=col)


# In[16]:


col = np.where(chromosomes%2==0, 'darkblue', 'black') #seperate chromosomes by color
col = np.squeeze(col)
plt.figure(figsize=(20,10))
plt.margins(0,0)
plt.ylim((-0.30,0.30))
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.title('Linear Regression Coefficients - Male without Cancer', fontsize=30)
plt.ylabel('Coefficient', fontsize=20)
plt.xlabel('Methylation Site', fontsize=20)
plt.scatter(range(coeffs.shape[1]), coeffs[5], s=5, marker='o', alpha=0.2, c=col)


# In[17]:


col = np.where(chromosomes%2==0, 'darkblue', 'black') #seperate chromosomes by color
col = np.squeeze(col)
plt.figure(figsize=(20,10))
plt.margins(0,0)
plt.ylim((-0.30,0.30))
plt.rc('xtick', labelsize=15)
plt.rc('ytick', labelsize=15)
plt.title('Linear Regression Coefficients - Male with Cancer', fontsize=30)
plt.ylabel('Coefficient', fontsize=20)
plt.xlabel('Methylation Site', fontsize=20)
plt.scatter(range(coeffs.shape[1]), coeffs[6], s=5, marker='o', alpha=0.2, c=col)


# In[33]:


significance_value = 0.05
abs_coeffs = np.absolute(coeffs)

num_flags_female = 0
num_flags_cancer = 0
num_flags_female_no_cancer = 0
num_flags_female_cancer = 0
num_flags_male_no_cancer = 0
num_flags_male_cancer = 0
num_flags_sig_both = 0
    
for i in range(abs_coeffs.shape[1]):
    sig_cancer = 0
    if abs_coeffs[1][i] > significance_value:
        num_flags_female += 1
    if abs_coeffs[2][i] > significance_value:
        num_flags_cancer += 1
        sig_cancer = 1
    if abs_coeffs[3][i] > significance_value:
        num_flags_female_no_cancer += 1
        if sig_cancer:
            num_flags_sig_both += 1
    if abs_coeffs[4][i] > significance_value:
        num_flags_female_cancer += 1
        if sig_cancer:
            num_flags_sig_both += 1
    if abs_coeffs[5][i] > significance_value:
        num_flags_male_no_cancer += 1
        if sig_cancer:
            num_flags_sig_both += 1
    if abs_coeffs[6][i] > significance_value:
        num_flags_male_cancer += 1
        if sig_cancer:
            num_flags_sig_both += 1
            
print "Significance level: 0.05\n"            
print "Proportion of sites where sex alone was significant:"
print num_flags_female/float(abs_coeffs.shape[1])
print "Proportion of sites where cancer alone was significant:"
print num_flags_cancer/float(abs_coeffs.shape[1])
print "Proportion of sites where female without cancer was significant:"
print num_flags_female_no_cancer/float(abs_coeffs.shape[1])
print "Proportion of sites where female with cancer was significant:"
print num_flags_female_cancer/float(abs_coeffs.shape[1])
print "Proportion of sites where male without cancer was significant:"
print num_flags_male_no_cancer/float(abs_coeffs.shape[1])
print "Proportion of sites where male with cancer was significant:"
print num_flags_male_cancer/float(abs_coeffs.shape[1])
print "Proportion of sites where cancer alone was significant but sex and cancer status was not:"
print (num_flags_cancer - (num_flags_female_no_cancer + num_flags_female_cancer + num_flags_male_no_cancer + num_flags_male_cancer))/float(abs_coeffs.shape[1])
print "Proportion of sites where sex and cancer status was significant but cancer alone was not:"
print (1 - num_flags_sig_both/float(num_flags_female_no_cancer + num_flags_female_cancer + num_flags_male_no_cancer + num_flags_male_cancer))

