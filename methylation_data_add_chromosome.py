import csv

# Load a CSV file
def load_csv(filename):
	file = open(filename, "r")
	lines = csv.reader(file)
	dataset = list(lines)
	return dataset

print "loading data files..."
methylation_data = load_csv('combined_methylation_data_with_chromosome_coordinate_blank.csv')
cg_names_data = load_csv('sorted_HumanMethylation450_15017482_v1-2.csv')
print "Done\n"

print "finding site's chromosome location..."
for i in range(len(methylation_data)):
    if i != 0:
        print i
        for k in range(len(cg_names_data)):
            if (methylation_data[i][0] == cg_names_data[k][0]):
                methylation_data[i][1] = cg_names_data[k][11]
                methylation_data[i][2] = cg_names_data[k][12]
                cg_names_data.pop(k)
                break
print "Done\n"

print "writing data to new csv file..."
with open("combined_methylation_data_with_chromosome_coordinate.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(methylation_data)
print "Done\n"