translate_index_dict = {}

for k in index_dict:
	translate_index_dict[index_dict[k]] = k

import pickle
import numpy

f = open("translate_index_dict.pkl")
index_dict = pickle.load(f)
f.close()

path = "scikit-learn/scikit_LR_matrix.csv"

dataset = numpy.loadtxt(path, delimiter=",", dtype="str")
# dataset=numpy.delete(dataset,len(dataset[0,:])-1,1)

for i, header in enumerate(dataset[0,:]):
	if i > 0:
		dataset[0,i] = index_dict[int(header)]

for i, header in enumerate(dataset[:,0]):
	if i > 0:
		dataset[i,0] = index_dict[int(header)]

f = open(path, "w")

for line in dataset:
	string = ""
	for element in line:
		string = string + str(element) + ","
	f.write(string + "\n")

f.close()

