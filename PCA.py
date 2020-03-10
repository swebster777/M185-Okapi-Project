import numpy as np
import csv #for loading
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

def load_csv(filename):
	file = open(filename, 'r')
	lines = csv.reader(file)
	dataset = list(lines)
	return dataset

def main():
    #import as numpy array which is really useful for matrix multiplication, etc.

	data = np.array(load_csv('methylation_data_no_labels.csv'))[20:-20].astype(np.float64)

	mu = np.mean(data, axis=0)
	print('data loaded')
	
	pca = PCA()
	pca.fit(data - mu)
	variance = [np.sum(pca.explained_variance_ratio_[:i]) for i in range(1, np.size(pca.explained_variance_ratio_))]
	variance.insert(0, 0)

	plt.figure(figsize=(20,10))
	plt.margins(0,0)
	plt.rc('xtick', labelsize=15)
	plt.rc('ytick', labelsize=15)
	plt.ylim((0, 1))
	plt.title('PCA variance', fontsize=30)
	plt.ylabel('% Variance explained', fontsize=20)
	plt.xlabel('Number of components used', fontsize=20)
	plt.plot(range(len(variance)), variance, color='darkblue')
	plt.savefig('PCA_variance.png')
	plt.show()

	plt.figure(figsize=(20,10))
	plt.margins(0,0)
	plt.rc('xtick', labelsize=15)
	plt.rc('ytick', labelsize=15)
	plt.title('PCA variance zoomed in', fontsize=30)
	plt.ylim((0, 1))
	plt.ylabel('% Variance explained', fontsize=20)
	plt.xlabel('Number of components used', fontsize=20)
	plt.plot(range(0, 11), variance[:11], color='darkblue')
	plt.savefig('PCA_variance_zoom.png')
	plt.show()
	
	pca = PCA(n_components=2)
	data_transform = pca.fit_transform(data - mu)

	plt.figure(figsize=(20,10))
	plt.margins(0,0)
	plt.rc('xtick', labelsize=15)
	plt.rc('ytick', labelsize=15)
	plt.title('PCA visualization', fontsize=30)
	plt.ylabel('PCA 2', fontsize=20)
	plt.xlabel('PCA 1', fontsize=20)
	plt.ylim((-10, 10))
	plt.xlim((-10, 10))
	plt.scatter(data_transform[:,0], data_transform[:,1], s=2, alpha=0.1, c='darkblue')
	plt.savefig('PCA_visualization.png')
	plt.show()

if __name__ == '__main__':
    main()
