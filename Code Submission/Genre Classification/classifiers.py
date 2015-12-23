from sets import Set
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.mixture import GMM
from sklearn import svm
import numpy as np
from scipy import sparse
import math
import re
import string
import codecs
import math
import pdb
from sklearn import datasets
from sklearn.decomposition import PCA

#data is in hw1 format, return list of list of words format for word2vec use
def getData(filename, vocab, intEntry= False):
	matrix = []
	labels = []
	f = open(filename, 'r')
	lines = f.readlines()
	for song in lines[3: ]:
		song = song.split()
		labels.append(int(song[0]))
		words = song[1: : 2]
		times = song[2: : 2]
		if intEntry:
			temp = np.zeros([1, len(vocab)])
			for i in range(len(words)):
				temp[0][int(words[i]) - 1] = times[i]
			matrix.append(temp[0])
		else:
			temp = []
			for i in range(len(words)):
				for j in range(int(times[i])):
					temp.append(vocab[int(words[i]) - 1])
			matrix.append(temp)
	if intEntry:
		matrix = np.array(matrix)
	return matrix, labels

# input train_text, vali_text, test_text: each being a list of strings
#       train_labels, vali_labels: each being a list of labels
def useWord2vec(train_text, train_labels, test_text, test_labels, get = False):

    from gensim.models import Word2Vec

    sentence = []
    sentence.extend([i for i in train_text])
    sentence.extend([i for i in test_text])

    # train your word2vec here
    model = Word2Vec(sentence, size = 100, window = 5, min_count = 1, workers = 4)

    # train your classifiers here
    trainMatrix = []
    for song in train_text:
        matrix = [model[word] for word in song]
        matrix = np.array(matrix).mean(0)
        trainMatrix.append(matrix)

    testMatrix = []
    for song in test_text:
        matrix = [model[word] for word in song]
        matrix = np.array(matrix).mean(0)
        testMatrix.append(matrix)

    ftrain = open('word2VecTrain', 'w')
    for song in trainMatrix:
    	for entry in song:
    		ftrain.write(str(entry))
    		ftrain.write(' ')
    	ftrain.write('\n')
    ftest = open('word2VecTest', 'w')
    for song in testMatrix:
    	for entry in song:
    		ftest.write(str(entry))
    		ftest.write(' ')
    	ftest.write('\n')
    if get:
    	return np.array(trainMatrix), np.array(testMatrix)

    C = [0.001, 0.01, 0.1, 1, 10, 100]
    accuracySVM = []
    accuracyLR = []
    for i in C:
        modelSVM = LinearSVC(C = i)
        modelLR = LogisticRegression(C = i)
        modelSVM.fit(trainMatrix, train_labels)
        modelLR.fit(trainMatrix, train_labels)
        predictionsSVM = modelSVM.predict(testMatrix)
        predictionsLR = modelLR.predict(testMatrix)
        accuracySVM.append(1.0 * sum(np.equal(predictionsSVM, test_labels)) / len(test_labels))
        accuracyLR.append(1.0 * sum(np.equal(predictionsLR, test_labels)) / len(test_labels))

    print accuracySVM
    print accuracyLR

def main():
	
	f = open('revised_dataset_test_7Genres.txt')
	vocab = f.readlines()[2].split()
	
	#(test_matrix, test_labels) = getData('reviesd_dataset_test_4Genres.txt', vocab)
	#(train_matrix, train_labels) = getData('reviesd_dataset_train_4Genres.txt', vocab)
	#useWord2vec(train_matrix, train_labels, test_matrix, test_labels)
	
	
	#integers matrix
	print 'fetch data'
	(test_matrix, test_labels) = getData('revised_dataset_test_7Genres.txt', vocab, True)
	(train_matrix, train_labels) = getData('revised_dataset_train_7Genres.txt', vocab, True)
	transformer = TfidfTransformer(norm = False)
	total = np.concatenate((train_matrix, test_matrix), axis = 0)
	
	print 'calculate tfidf\n'
	total = transformer.fit_transform(total).toarray()
	train_matrix = sparse.csr_matrix(total[0: train_matrix.shape[0], :])
	test_matrix = sparse.csr_matrix(total[train_matrix.shape[0]: total.shape[0], :])
	
	
	#multinomial naive bayes
	print 'multinomial naive bayes'
	modelNB = MultinomialNB()
	modelNB.fit(train_matrix, train_labels)
	predictionsNB = modelNB.predict(train_matrix)
	print 'train accuracy: ', 1.0 * sum(np.equal(predictionsNB, train_labels)) / len(train_labels)
	predictionsNB = modelNB.predict(test_matrix)
	print 'test accuracy: ', 1.0 * sum(np.equal(predictionsNB, test_labels)) / len(test_labels)
	total2 = [0, 0, 0, 0, 0, 0, 0]
	correct = [0, 0, 0, 0, 0, 0, 0]
	for i in range(len(predictionsNB)):
		total2[test_labels[i] - 1] += 1
		correct[test_labels[i] - 1] += (predictionsNB[i] == test_labels[i])
	print total2
	print correct
	for j in range(7):
		print 1.0 * correct[j] / total2[j]

	#linear svm
	print 'linear svm'
	C = [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
	
	i = 1e-5
	modelSVM = svm.SVC(kernel = 'linear', gamma = i)
	print i
	modelSVM.fit(train_matrix, train_labels)
	predictionsSVM = modelSVM.predict(test_matrix)
	accuracySVM = 1.0 * sum(np.equal(predictionsSVM, test_labels)) / len(test_labels)
	total = [0, 0, 0, 0, 0, 0, 0]
	correct = [0, 0, 0, 0, 0, 0, 0]
	for i in range(len(prediction)):
		total[test_labels[i] - 1] += 1
		correct[test_labels[i] - 1] += (predictionsSVM[i] == test_labels[i])
	for j in range(7):
		print 1.0 * correct[j] / total[j]
	print i, accuracySVM
	
	#gaussian svm
	
	print 'gaussian kernel svm'
	i = 0.001
	modelSVM = svm.SVC(kernel = 'rbf', C = i)
	modelSVM.fit(train_matrix[0: 2500, :], train_labels[0: 2500])
	predictionsSVM = modelSVM.predict(test_matrix)
	accuracySVM = 1.0 * sum(np.equal(predictionsSVM, test_labels)) / len(test_labels)
	print i, accuracySVM
	
	#gaussian naive bayes
	
	modelNB = GaussianNB()
	modelNB.fit(train_matrix.toarray(), train_labels)
	predictionsNB = modelNB.predict(test_matrix.toarray())
	print 'test accuracy: ', 1.0 * sum(np.equal(predictionsNB, test_labels)) / len(test_labels)
	predictionsNB = modelNB.predict(train_matrix.toarray())
	print 'train accuracy: ', 1.0 * sum(np.equal(predictionsNB, train_labels)) / len(train_labels)
	
	#logistic regression
	
	print 'doing feature selection with PCA\nextract :',
	featureNum = 3500
	print featureNum, ' features'
	pca = PCA(n_components = featureNum)
	total = pca.fit_transform(total)
	
	#get training data
	train_matrix = total[0: train_matrix.shape[0], :]
	test_matrix = total[train_matrix.shape[0]: total.shape[0], :]

	#concatenate constant
	train_matrix = np.concatenate((train_matrix, np.ones([train_matrix.shape[0], 1])), axis = 1)
	test_matrix = np.concatenate((test_matrix, np.ones([test_matrix.shape[0], 1])), axis = 1)

	print 'logistic regression'
	C = [10, 1, 0.1, 0.01, 0.001, 0.0001, 0.00001]
	for i in C:
		print i,
		modelLR = modelLR = LogisticRegression(C = i)
		modelLR.fit(train_matrix, train_labels)
		predictionsLR = modelLR.predict(test_matrix)
		accuracyLR = (1.0 * sum(np.equal(predictionsLR, test_labels)) / len(test_labels))
		print 'test accuracy: ', accuracyLR
		predictionsLR = modelLR.predict(train_matrix)
		accuracyLR = (1.0 * sum(np.equal(predictionsLR, train_labels)) / len(train_labels))
		print 'train accuracy: ', accuracyLR
	
	

if __name__ == '__main__':
    main()