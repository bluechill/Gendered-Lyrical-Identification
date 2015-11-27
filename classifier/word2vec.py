from stanford_corenlp_pywrapper import CoreNLP
from sets import Set
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, TfidfTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.mixture import GMM
from sklearn.svm import LinearSVC
import numpy as np
from scipy import sparse
import math
import re
import string
import codecs
import math
import pdb
from sklearn import datasets

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
	
	f = open('dataset_5Genres.test')
	vocab = f.readlines()[2].split()
	'''
	(test_matrix, test_labels) = getData('dataset_5Genres.test', vocab)
	(train_matrix, train_labels) = getData('dataset_5Genres.train', vocab)
	useWord2vec(train_matrix, train_labels, test_matrix, test_labels)
	'''
	
	#integers matrix
	(test_matrix, test_labels) = getData('dataset_5Genres.test', vocab, True)
	(train_matrix, train_labels) = getData('dataset_5Genres.train', vocab, True)
	#multinomial naive bayes
	'''
	modelNB = MultinomialNB()
	modelNB.fit(train_matrix, train_labels)
	predictionsNB = modelNB.predict(test_matrix)
	print 1.0 * sum(np.equal(predictionsNB, test_labels)) / len(test_labels)
	'''

	#gaussian naive bayes
	'''
	transformer = TfidfTransformer(norm = None)
	tfidf_train = transformer.fit_transform(train_matrix.tolist())
	tfidf_test = transformer.fit_transform(test_matrix.tolist())

	tfidf_train = tfidf_train.toarray()
	tfidf_test = tfidf_test.toarray()
	modelNB = GaussianNB()
	modelNB.fit(tfidf_train, train_labels)
	predictionsNB = modelNB.predict(tfidf_test)
	print 1.0 * sum(np.equal(predictionsNB, test_labels)) / len(test_labels)
	'''
	#use GMM as classifier
	(test_matrix, test_labels) = getData('dataset_5Genres.test', vocab)
	(train_matrix, train_labels) = getData('dataset_5Genres.train', vocab)
	(train_matrix, test_matrix) = useWord2vec(train_matrix, train_labels, test_matrix, test_labels, True)
	classifiers = dict((covar_type, GMM(n_components=4,
                    covariance_type=covar_type, init_params='wc', n_iter=20))
                   for covar_type in ['spherical', 'diag', 'tied', 'full'])
	for index, (name, classifier) in enumerate(classifiers.items()):
		classifier.means_ = np.array([train_matrix[range(2500 * (i - 1), 2500 * i), :].mean(axis=0)
                                  for i in range(4)])
		classifier.fit(train_matrix.tolist())
		predictions = classifier.predict(test_matrix)
		f = open('predictions.txt', 'w')
		for i in range(len(test_labels)):
			predictions[i] += 1
		for i in range(len(test_labels)):
			if test_labels[i] == 5:
				test_labels[i] == 4
		print name, 1.0 * sum(np.equal(predictions, test_labels)) / len(test_labels)
	

if __name__ == '__main__':
    main()