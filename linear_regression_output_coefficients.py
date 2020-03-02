import numpy as np
import csv
from sklearn.linear_model import LinearRegression

def load_csv(filename):
	file = open(filename, 'r')
	lines = csv.reader(file)
	dataset = list(lines)
	return dataset

def main():
	#import methylation
	#import meta data
	meta_data = np.array(load_csv('meta_data_no_labels.csv'), dtype=np.float64)[20:-20, :]
	#transpose meta data
	methylation_data = np.array(load_csv('methylation_data_no_labels.csv'), dtype=np.float64)[:, 20:-20]
	
	coefficient_matrix = []
    
	for i in range(np.size(methylation_data[:,0])):
            y_true = methylation_data[i,].T
            reg = LinearRegression().fit(meta_data, y_true)
            coefficient_matrix.append(list(reg.coef_))
            print i

	with open("coefficients.csv", "wb") as f:
		writer = csv.writer(f)
		writer.writerows(coefficient_matrix)

if __name__ == '__main__':
	main()