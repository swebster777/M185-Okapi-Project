import csv

# Load a CSV file
def load_csv(filename):
	file = open(filename, "r")
	lines = csv.reader(file)
	dataset = list(lines)
	return dataset

print "loading data files..."
unhealthy_methylation_data = load_csv('GSE101764_XY_filtered_methylation_data_combining.csv')
healthy_methylation_data = load_csv('GSE69270_XY_methylation_data.csv')
print "Done\n"

print "combining data files..."
for i in range(len(unhealthy_methylation_data)):
    if i != 0:
        print i
        for k in range(len(healthy_methylation_data)):
            if (unhealthy_methylation_data[i][0] == healthy_methylation_data[k][0]):
                for q in range(len(healthy_methylation_data[0]) - 1):
                   unhealthy_methylation_data[i][len(unhealthy_methylation_data[0]) - len(healthy_methylation_data[0]) + q + 1] = healthy_methylation_data[k][q + 1]
                healthy_methylation_data.pop(k)
                break
print "Done\n"

print "writing combined data to new csv file..."
with open("combined_methylation_data.csv", "wb") as f:
    writer = csv.writer(f)
    writer.writerows(unhealthy_methylation_data)
print "Done\n"