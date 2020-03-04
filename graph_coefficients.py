import numpy as np
import csv #for loading
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def load_csv(filename, sep=',', training_indices=(20,-20), runningTest=True):
    with open(filename, 'r') as file:
    	for line in file:
        	line = line.split(sep)

        	if training_indices != None:
        		if runningTest == True:
        			del line[training_indices[0]: training_indices[1]]
        		else:
        			line = line[training_indices[0]: training_indices[1]]

        	yield line

def main():
    #import as numpy array which is really useful for matrix multiplication, etc.
    print('testing')
    meta_data = np.array([line for line in load_csv('meta_data_no_labels.csv', training_indices=None)], dtype=np.float64)[20:-20]
    print('Finished loading predictions')

    coefficient_matrix = []
    for i, line in enumerate(load_csv('methylation_data_no_labels.csv', runningTest=False)):
    	y_true = np.array(line, dtype=np.float64)
    	reg = LinearRegression().fit(meta_data, y_true)
    	intercept_and_coef = list(reg.coef_)
    	intercept_and_coef.insert(0, reg.intercept_)
    	coefficient_matrix.append(intercept_and_coef)
    	
    	if i % 10000 == 0:
    		print(i)

    coefficient_matrix = np.array(coefficient_matrix, dtype=np.float64)

    # plt.figure(figsize=(20,10))
    # plt.margins(0,0)
    # plt.rc('xtick', labelsize=15)
    # plt.rc('ytick', labelsize=15)
    # plt.title('Coefficients graphed by location on the genome', fontsize=30)
    # plt.ylabel('Coefficient value', fontsize=20)
    # plt.xlabel('Genome location', fontsize=20)

    chrom_col = np.array([line[0] for line in load_csv('chromosome_col.csv', runningTest=False, training_indices=None)], dtype=np.float64)[:1000]
    print(chrom_col.shape)

    col = np.squeeze(np.where(chrom_col % 2 == 0, 'darkblue', 'black'))
    coefficient_names = ['Female', 'Male', 'Cancer', 'Healthy', 'Female with Cancer', 'Healthy Female', 'Female with Cancer', 'Healthy Male', 'Male with Cancer']
    for i in xrange(coefficient_matrix.shape[1]):
    	plt.figure(figsize=(20,10))
    	plt.margins(0,0)
    	plt.rc('xtick', labelsize=15)
    	plt.rc('ytick', labelsize=15)
    	plt.title('Methylation site vs %s Coefficient' % (coefficient_names[i],), fontsize=30)
    	plt.ylabel('Coefficient value', fontsize=20)
    	plt.xlabel('Genome location', fontsize=20)
    	plt.scatter(range(coefficient_matrix.shape[0]), list(coefficient_matrix[:,i]), marker='o', s=5, c=col)
    	plt.show()

    plt.show()

if __name__ == '__main__':
    main()
