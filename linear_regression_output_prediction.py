import numpy as np
import csv

def load_csv(filename):
	file = open(filename, 'r')
	lines = csv.reader(file)
	dataset = list(lines)
	return dataset

def main():
	#import methylation
	#import meta data
    meta_data = np.array(load_csv('meta_data_no_labels.csv'), dtype=np.float64)
	#transpose meta data
    #methylation_data = np.array(load_csv('methylation_data_no_labels.csv'), dtype=np.float64)
    coefficient_matrix = np.array(load_csv('intercept_and_coefficients.csv'), dtype=np.float64)
    predictions = []
    
    for i in range(np.size(coefficient_matrix[:,0])):
            print i
            intercept = coefficient_matrix[i][0]
            co_female = coefficient_matrix[i][1]
            co_male = coefficient_matrix[i][2]
            co_cancer = coefficient_matrix[i][3]
            co_no_cancer = coefficient_matrix[i][4]
            co_male_no_cancer = coefficient_matrix[i][5]
            co_male_cancer = coefficient_matrix[i][6]
            co_female_no_cancer = coefficient_matrix[i][7]
            co_female_cancer = coefficient_matrix[i][8]
            predictions_for_site = []
            for k in range(np.size(meta_data[:,0])):
                    predictions_for_site.append(intercept + co_female*meta_data[k][0] + co_male*meta_data[k][1] + co_cancer*meta_data[k][2] + co_no_cancer*meta_data[k][3] + co_male_no_cancer*meta_data[k][4] + co_male_cancer*meta_data[k][5] + co_female_no_cancer*meta_data[k][6] + co_female_cancer*meta_data[k][7])
            predictions.append(predictions_for_site)
                    
            
    with open("predictions.csv", "wb") as f:
		writer = csv.writer(f)
		writer.writerows(predictions)

if __name__ == '__main__':
	main()