# python libraries
import os

# numpy libraries
import numpy as np

import csv
 

# Load a CSV file
def load_csv(filename):
	file = open(filename, "r")
	lines = csv.reader(file)
	dataset = list(lines)
	return dataset

print "loading data files..."
methylation_data = load_csv('GSE101764_filtered_methylation_data.csv')
cg_names_data = load_csv('HumanMethylation450_15017482_v1-2.csv')
print "Done\n"

XY_cg_names = []

print "extracting methylation site chromesome locations..."
for i in range(len(cg_names_data)):
    if len(cg_names_data[i]) > 11:
        if cg_names_data[i][11] == 'X' or cg_names_data[i][11] == 'Y':
            XY_cg_names.append(cg_names_data[i][0])
print "Done\n"
#print XY_cg_names

indexes_to_remove = []

print "finding methylation data from X and Y chromosomes..."
for i in range(len(methylation_data)):
    if 'ch.X' in methylation_data[i][0] or methylation_data[i][0] in XY_cg_names:
        indexes_to_remove.append(i)
print "Done\n"
#print indexes_to_remove

print "removing methylation data from X and Y chromosomes..."
XY_filtered_methylation_data = np.delete(methylation_data, indexes_to_remove, 0)
print "Done\n"
#print XY_filtered_methylation_data

print "writing filtered data to new csv file..."
with open("GSE101764_XY_filtered_methylation_data.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(XY_filtered_methylation_data)
print "Done\n"