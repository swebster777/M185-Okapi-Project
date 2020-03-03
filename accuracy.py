import numpy as np
import csv #for loading
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt

def load_csv(filename):
    file = open(filename, 'r')
    lines = csv.reader(file)
    dataset = list(lines)
    return dataset

def main():
    #import as numpy array which is really useful for matrix multiplication, etc.
    predictions = np.array(load_csv('C:\\Users\\minah\\OneDrive\\Documents\\UCLA\\Junior Year\\Winter\\CASB_185\\predictions.csv'), dtype=np.float64)
    last20 = predictions[:, -20:]
    first20 = predictions[:, :20]
    predictions = np.append(first20, last20, axis=1)
    print('Finished loading predictions')
    
    methylation_data = np.array(load_csv('C:\\Users\\minah\\OneDrive\\Documents\\UCLA\\Junior Year\\Winter\\CASB_185\\methylation_data_no_labels.csv'), dtype=np.float64)
    last20 = methylation_data[:, -20:]
    first20 = methylation_data[:, :20]
    methylation_data = np.append(first20, last20, axis=1)
    print('Finished loading methylation_data')

    difference = np.absolute(np.subtract(methylation_data, predictions))
    error_margin = np.divide(difference, np.absolute(methylation_data))

    num_flags = 0

    for i in range(error_margin.shape[0]): # number of rows
    	for j in range(error_margin.shape[1]): # number of cols
        	if error_margin[i,j] > 0.1: # 10% margin of error
        		num_flags += 1
    	if i%100000 == 0:
        	print i
        
	print num_flags/float(error_margin.shape[0]*error_margin.shape[1]) # cast denominator to float so that result is float

	errors_by_site = np.mean(error_margin, axis=1)

	plt.scatter(range(errors_by_site.shape[0]), errors_by_site)
	plt.show()

	true_values = np.mean(methylation_data, axis=1)
	predicted_values = np.mean(predictions, axis=1)

	predicted_values_above = predicted_values + np.std(predictions, axis=1)
	predicted_values_below = predicted_values - np.std(predictions, axis=1)

	plt.scatter(list(true_values), list(predicted_values))
	plt.scatter(list(true_values), list(predicted_values_above))
	plt.scatter(list(true_values), list(predicted_values_below))
	plt.show()

if __name__ == '__main__':
	main()