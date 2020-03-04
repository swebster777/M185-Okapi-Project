import numpy as np
import csv #for loading
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
import matplotlib.pyplot as plt

def load_csv(filename, sep=',', training_indices=(20,-20), runningTest=True):
    csv = []
    with open(filename, 'r') as file:
    	for line in file:
        	line = line.split(sep)

        	if training_indices != None:
        		if runningTest == True:
        			del line[training_indices[0]: training_indices[1]]
        		else:
        			line = line[training_indices[0]: training_indices[1]]

        	csv.append([float(j) for j in line])
    return np.array(csv, dtype=np.float64)

def main():
    #import as numpy array which is really useful for matrix multiplication, etc.
    print('testing')
    predictions = load_csv('predictions.csv')
    print('Finished loading predictions')
    print(predictions.shape)
    methylation_data = load_csv('methylation_data_no_labels.csv')
    print(methylation_data.shape)
    print('Finished loading methylation_data')

    difference = np.absolute(np.subtract(methylation_data, predictions))
    error_margin = np.divide(difference, np.absolute(methylation_data))

    num_flags = 0

    for i in range(error_margin.shape[0]): # number of rows
    	for j in range(error_margin.shape[1]): # number of cols
    		if error_margin[i,j] > 0.1: # 10% margin of error
    			num_flags += 1
    				
    	if i%100000 == 0:
    		print(i)

    print(num_flags / float(error_margin.shape[0] * error_margin.shape[1]))

    errors_by_site = np.mean(error_margin, axis=1)

    plt.scatter(range(errors_by_site.shape[0]), errors_by_site)
    plt.show()
    
    true_values = np.mean(methylation_data, axis=1)
    predicted_values = np.mean(predictions, axis=1)

    predicted_values_above = predicted_values + np.std(predictions, axis=1)
    predicted_values_below = predicted_values - np.std(predictions, axis=1)

    print(r2_score(true_values, predicted_values))

    plt.figure(figsize=(20,10))
    plt.margins(0,0)
    plt.ylim((0,1))
    plt.xlim((0,1))
    plt.rc('xtick', labelsize=15)
    plt.rc('ytick', labelsize=15)
    plt.title('Average Error by Methylation Site, color seperated by chromosome (no PCA)', fontsize=30)
    plt.ylabel('True Methylation Value', fontsize=20)
    plt.xlabel('Predicted Methylation Value', fontsize=20)

    plt.scatter(list(true_values), list(predicted_values_above),s=5, alpha=0.2)
    plt.scatter(list(true_values), list(predicted_values_below),s=5, alpha=0.2)
    plt.scatter(list(true_values), list(predicted_values),s=5, alpha=0.2)
    plt.show()

if __name__ == '__main__':
    main()
