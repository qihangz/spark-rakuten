import numpy

from sklearn.naive_bayes import BernoulliNB, MultinomialNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn import linear_model
from sklearn import svm
from sklearn.decomposition import RandomizedPCA

data = numpy.load("dense_data.npy")
top_features = numpy.load("/home/qihangz/Desktop/small_test/top_features.npy")
LR_model = linear_model.LogisticRegression()

result = []

for feature in top_features:
	LR_result = LR_model.fit(numpy.delete(data, feature, 1), data[:,feature])
	result.append([feature, numpy.insert(LR_result.coef_, feature, None)[topFeatures]])

f = open("scikit_LR_matrix.csv", "w")

title = "column\t"
for item in result:
	title = title + str(item[0]) + "\t"

f.write(title+"\n")

for item in result:
	string = ""
	for element in item[1]:
		string = string + str(element) + "\t"
	f.write(str(item[0]) + "\t" + string+"\n")

f.close()



LR_model.fit(X_train[index], y_train[index])

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