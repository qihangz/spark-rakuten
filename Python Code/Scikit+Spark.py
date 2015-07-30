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
	return (X, y.reshape(-1))

def vote_increment(y_est):
	y_est = [int(i) for i in y_est]
	increment = zero_matrix(len(y_est), n_ys)
	increment[numpy.arange(len(y_est)), y_est] = 1
	return increment # test point x class matrix with 1s marking the estimator prediction

def zero_matrix(n, m):
	return numpy.zeros(n*m, dtype = int).reshape(n, m)

select_n = 30
topFeatures = selectTopNFeatures(select_n)

n_ys = 2
data = sc.textFile("test", 80)
row_num = data.count()
col_num = data.flatMap(lambda line : line.split("\t")).distinct().count()
column_index = data.flatMap(indexColumn).collect()
row_index = data.zipWithIndex().flatMap(indexRow).collect()

complete_matrix = csr_matrix((numpy.ones(len(column_index)), (row_index, column_index)), shape=(row_num, col_num))

precisions = []
recalls = []

model_start = time.time()
for targetColumn in topFeatures:
	parsedData = getXY(complete_matrix, int(targetColumn))
	X_train, X_test, y_train, y_test = train_test_split(parsedData[0], parsedData[1], train_size=0.9)
	n_test = X_test.shape[0]
	model = BernoulliNB()
	#model = MultinomialNB()
	#model = KNeighborsClassifier()
	#model = linear_model.SGDClassifier()
	#model = linear_model.LogisticRegressionCV()
	#model = svm.LinearSVC()
	samples = sc.parallelize(Bootstrap(X_train.shape[0], n_iter=19, train_size=0.5), 8)
	vote_result = samples.map(lambda (index, _) : model.fit(X_train[index], y_train[index]).predict(X_test)).map(vote_increment).fold(zero_matrix(n_test, n_ys), numpy.add)
	y_estimate_vote = numpy.argmax(vote_result, axis = 1)
	precisions.append(precision_score(y_test, y_estimate_vote))
	recalls.append(recall_score(y_test, y_estimate_vote))
	samples = sc.parallelize([numpy.arange(X_train.shape[0])])
	vote_result = samples.map(lambda index : model.fit(X_train[index], y_train[index]).predict(X_test)).map(vote_increment).fold(zero_matrix(n_test, n_ys), numpy.add)
	y_estimate_vote = numpy.argmax(vote_result, axis = 1)
	precisions.append(precision_score(y_test, y_estimate_vote))
	recalls.append(recall_score(y_test, y_estimate_vote))

end = time.time()

print (end - model_start) / 60
numpy.mean(precisions)
numpy.mean(recalls)

for i in range(2):
	print i
	print numpy.mean([precisions[x*2+i] for x in range(30)])
	print numpy.mean([recalls[x*2+i] for x in range(30)])