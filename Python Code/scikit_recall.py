from operator import itemgetter
from sklearn.cross_validation import train_test_split, Bootstrap
from sklearn.metrics import accuracy_score, precision_score, recall_score
from scipy.sparse import csr_matrix
import numpy
import time

from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn import svm
from sklearn.decomposition import RandomizedPCA

""" Select the top_n features"""
def selectTopNFeatures(top_n):
	data = sc.textFile("test", 80)
	columns = data.flatMap(lambda line : line.split("\t")).map(lambda col : (col, 1)).reduceByKey(lambda x, y : x + y)
	sortedFeatures = sorted(columns.collect(), key=itemgetter(1), reverse=True)
	topFeatures = list(feature[0] for feature in sortedFeatures[0 : top_n]) # select & filter out the word count
	return topFeatures

def indexColumn(line):
	values = [int(x) for x in line.split("\t")]
	values.sort()
	return values

def indexRow(input):
	line = input[0]
	features = [int(x) for x in line.split("\t")]
	row_num = input[1]
	row_index = numpy.empty(len(features))
	row_index.fill(row_num)
	return row_index

def getXY(old_x, col_to_delete):
	all_cols = numpy.arange(old_x.shape[1])
	cols_to_keep = numpy.where(numpy.logical_not(numpy.in1d(all_cols, col_to_delete)))[0]
	X = old_x[:, cols_to_keep]
	y = old_x[:, col_to_delete].toarray()
	return (X, y.reshape(len(y)))

def findCoveragePercent(socredLabel, recall):
	if recall > 1:
		return 1.0
	sum = socredLabel.sum()
	accum = 0
	for i in range(socredLabel.size):
		accum += socredLabel[i]
		if accum >= recall*sum:
			return i / float(socredLabel.size)

#select_n = 30
#topFeatures = selectTopNFeatures(select_n)
topFeatures = [711]

n_ys = 2
data = sc.textFile("test", 80)
row_num = data.count()
col_num = data.flatMap(lambda line : line.split("\t")).distinct().count()
column_index = data.flatMap(indexColumn).collect()
row_index = data.zipWithIndex().flatMap(indexRow).collect()

complete_matrix = csr_matrix((numpy.ones(len(column_index)), (row_index, column_index)), shape=(row_num, col_num))
pca = RandomizedPCA(n_components=5)

#input_matrix = sc.broadcast(complete_matrix)

for targetColumn in topFeatures:
	parsedData = getXY(complete_matrix, int(targetColumn))
	X_train, X_test, y_train, y_test = train_test_split(parsedData[0], parsedData[1], train_size=0.9)
	n_test = X_test.shape[0]
	indexes = sc.parallelize([numpy.arange(X_train.shape[0])],80)
	##########
	BNB_model = BernoulliNB()
	BNB_prob = indexes.flatMap(lambda index : BNB_model.fit(X_train[index], y_train[index]).predict_proba(X_test)).map(lambda values : values[1]).collect()
	BNB_result = numpy.concatenate((numpy.array(BNB_prob).reshape((-1,1)), y_test.reshape((-1,1))), axis=1)
	BNB_sorted_result = BNB_result[BNB_result[:,0].argsort()[::-1]][:,1] #sort the result by probabilities and retain the second column
	##########
	MNB_model = MultinomialNB()
	MNB_prob = indexes.flatMap(lambda index : MNB_model.fit(X_train[index], y_train[index]).predict_proba(X_test)).map(lambda values : values[1]).collect()
	MNB_result = numpy.concatenate((numpy.array(MNB_prob).reshape((-1,1)), y_test.reshape((-1,1))), axis=1)
	MNB_sorted_result = MNB_result[MNB_result[:,0].argsort()[::-1]][:,1] #sort the result by probabilities and retain the second column	
	##########
	LR_model = linear_model.LogisticRegression()
	LR_prob = indexes.flatMap(lambda index : LR_model.fit(X_train[index], y_train[index]).predict_proba(X_test)).map(lambda values : values[1]).collect()
	LR_result = numpy.concatenate((numpy.array(LR_prob).reshape((-1,1)), y_test.reshape((-1,1))), axis=1)
	LR_sorted_result = LR_result[LR_result[:,0].argsort()[::-1]][:,1] #sort the result by probabilities and retain the second column
	##########
	SVM_model = svm.LinearSVC()
	SVM_prob = indexes.flatMap(lambda index : SVM_model.fit(X_train[index], y_train[index]).predict_proba(X_test)).map(lambda values : values[1]).collect()
	SVM_result = numpy.concatenate((numpy.array(SVM_prob).reshape((-1,1)), y_test.reshape((-1,1))), axis=1)
	SVM_sorted_result = SVM_result[SVM_result[:,0].argsort()[::-1]][:,1] #sort the result by probabilities and retain the second column


findCoveragePercent(BNB_sorted_result, 0.4)
findCoveragePercent(BNB_sorted_result, 0.8)
findCoveragePercent(BNB_sorted_result, 1)
findCoveragePercent(MNB_sorted_result, 0.4)
findCoveragePercent(MNB_sorted_result, 0.8)
findCoveragePercent(MNB_sorted_result, 1)
findCoveragePercent(LR_sorted_result, 0.4)
findCoveragePercent(LR_sorted_result, 0.8)
findCoveragePercent(LR_sorted_result, 1)

for i in range(2):
	print i
	print numpy.mean([precisions[x*2+i] for x in range(30)])
	print numpy.mean([recalls[x*2+i] for x in range(30)])