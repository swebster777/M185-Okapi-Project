import os
import numpy as np
import sklearn as sk

def main():
    #read healthy metadata
    dir = os.path.dirname(__file__)
    f = os.path.join(dir, 'GSE101764_metadata.csv')

    with open(f, 'r') as fid:
        meta_data_healthy = np.loadtxt(fid, skiprows=2).T

    #add new cols (isMale, hasCancer, and doesntHaveCancer)
    meta_data_healthy = np.append(meta_data_healthy, np.logical_not(meta_data_healthy[:,1:2]), axis=1)
    meta_data_healthy = np.append(meta_data_healthy, np.zeros(shape=(meta_data_healthy.shape[0], 1)), axis=1)
    meta_data_healthy = np.append(meta_data_healthy, np.ones(shape=(meta_data_healthy.shape[0], 1)), axis=1)

    #read unhealthy metadata
    f = os.path.join(dir, '<unhealthy_metadata>.csv')

    with open(f, 'r') as fid:
        meta_data_unhealthy = np.loadtxt(fid, skiprows=2).T

    #add new cols(isMale, hasCancer, and doesntHaveCancer)
    meta_data_unhealthy = np.append(meta_data_unhealthy, np.logical_not(meta_data_unhealthy[:,1:2]), axis=1)
    meta_data_unhealthy = np.append(meta_data_unhealthy, np.ones(shape=(meta_data_unhealthy.shape[0], 1)), axis=1)
    meta_data_unhealthy = np.append(meta_data_unhealthy, np.zeros(shape=(meta_data_unhealthy.shape[0], 1)), axis=1)

    #combine meta_data from both datasets into one array
    meta_data = np.append(meta_data_healthy, meta_data_unhealthy, axis=0)

    #read PCs dataset
    f = os.path.join(dir, '<principal_components>.csv')
    with open(f, 'r') as fid:
        ncols=len(fid.readline().split('\t'))

    with open(f, 'r') as fid:
        principal_components = np.loadtxt(fid, usecols=range(1, ncols-1), skiprows=2).T

    #combine PCs with metadata
    final_array = np.append(meta_data, principal_components, axis=1)

    # #read healthy methylation data
    # f = os.path.join(dir, 'GSE101764_filtered_methylation_data.csv')

    # with open(f, 'r') as fid:
    #     ncols = len(fid.readline().split(','))

    # with open(f, 'r') as fid:
    #     methylation_data_healthy = np.loadtxt(fid, delimiter=',', usecols=range(1, ncols-1), skiprows=1).T

    # #read unhealthy methylation data
    # f = os.path.join(dir, 'GSE101764_filtered_methylation_data.csv')

    # with open(f, 'r') as fid:
    #     ncols = len(fid.readline().split(','))

    # with open(f, 'r') as fid:
    #     methylation_data_unhealthy = np.loadtxt(fid, delimiter=',', usecols=range(1, ncols-1), skiprows=1).T

    # #combine methylation datasets
    # methylation_data = np.append(methylation_data_healthy, methylation_data_unhealthy, axis=0)

    # #standardize methylation data to mean=0, var=1
    # methylation_data = sk.preprocessing.StandardScaler().fit_transform(methylation_data)

    # #run pca on healthy methylation data
    # pca = sk.decomposition.PCA(n_components=5)

    # principal_components = pca.fit_transform(methylation_data)


    # final_array = np.append(meta_data, principle_components, axis=1)
    
    np.savetxt('final_matrix.csv', final_array, delimeter=',')

if __name__ == "__main__":
    main()
