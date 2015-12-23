import numpy as np
from keras.models import Sequential
from keras.layers import containers
from keras.layers.core import Dense, Dropout, Activation, AutoEncoder, Layer
from keras.optimizers import SGD, RMSprop
from keras.regularizers import l2
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.decomposition import PCA
import string
import re

#data is in hw1 format, return list of list of words format for word2vec use
def getData(filename, vocab, intEntry= False):
	matrix = []
	labels = []
	classes = []
	f = open(filename, 'r')
	lines = f.readlines()
	for song in lines[3: ]:
		song = song.split()
		if song[0] == '1':
			labels.append([1, 0, 0, 0, 0, 0, 0])
			classes.append(1)
		elif song[0] == '2':
			labels.append([0, 1, 0, 0, 0, 0, 0])
			classes.append(2)
		elif song[0] == '3':
			labels.append([0, 0, 1, 0, 0, 0, 0])
			classes.append(3)
		elif song[0] == '4':
			labels.append([0, 0, 0, 1, 0, 0, 0])
			classes.append(4)
		elif song[0] == '5':
			labels.append([0, 0, 0, 0, 1, 0, 0])
			classes.append(5)
		elif song[0] == '6':
			labels.append([0, 0, 0, 0, 0, 1, 0])
			classes.append(6)
		elif song[0] == '7':
			labels.append([0, 0, 0, 0, 0, 0, 1])
			classes.append(7)
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
	return matrix, labels, classes

def preprocess(directory, vocab):
	f = open(directory, 'r')
	lines = f.readlines()
	songs = []
	temp = []
	for line in range(len(lines)):
		if len(lines[line]) > 0 and lines[line][0] != '$':
			temp.append(lines[line])
		elif len(lines[line]) > 0 and lines[line][0] == '$':
			if len(temp) != 0:
				songs.append(' '.join(temp))
			temp = []
		if line == len(lines) - 1:
			songs.append(' '.join(temp))
	
	lyrics = []
	for song in songs:
		for c in string.punctuation:
			song = song.replace(c, " ")
		song = song.split()
		lyrics.append(song)
	matrix = []
	for lyric in lyrics:
		vector = np.zeros([1, len(vocab)])[0]
		for word in lyric:
			if word in vocab:
				vector[vocab.index(word)] += 1
		matrix.append(vector)
	return np.array(matrix)
	


def main():
	f = open('revised_dataset_test_7Genres.txt')
	vocab = f.readlines()[2].split()
	rap_songs = preprocess('generated_50_cent_songs.txt', vocab)
	country_songs = preprocess('generated_johnny_cash_songs.txt', vocab)
	#import raw data
	print 'fetching data'
	(test_matrix, test_labels, test_class) = getData('revised_dataset_test_7Genres.txt', vocab, True)
	(train_matrix, train_labels, train_class) = getData('revised_dataset_train_7Genres.txt', vocab, True)
	transformer = TfidfTransformer(norm = False)
	total = np.concatenate((train_matrix, test_matrix, rap_songs, country_songs), axis = 0)

	print 'calculate tfidf\n'
	total = transformer.fit_transform(total).toarray()

	
	print 'doing feature selection with PCA\nextract :',
	featureNum = 3500
	print featureNum, ' features'
	pca = PCA(n_components = featureNum)
	total = pca.fit_transform(total)
	
	#get training data
	train_matrix = total[0: train_matrix.shape[0], :]
	test_matrix = total[train_matrix.shape[0]: train_matrix.shape[0] + test_matrix.shape[0], :]
	rap_songs = total[train_matrix.shape[0] + test_matrix.shape[0]: train_matrix.shape[0] + test_matrix.shape[0] + rap_songs.shape[0], :]
	country_songs = total[train_matrix.shape[0] + test_matrix.shape[0] + rap_songs.shape[0]: total.shape[0], :]

	#concatenate constant
	train_matrix = np.concatenate((train_matrix, np.ones([train_matrix.shape[0], 1])), axis = 1)
	test_matrix = np.concatenate((test_matrix, np.ones([test_matrix.shape[0], 1])), axis = 1)
	rap_songs = np.concatenate((rap_songs, np.ones([rap_songs.shape[0], 1])), axis = 1)
	country_songs = np.concatenate((country_songs, np.ones([country_songs.shape[0], 1])), axis = 1)
	#pretrain with autoencoder
	'''
	print 'pretrain with autoencoder\n'
	featureSize1 = 1500
	encoder1 = containers.Sequential([Dense(featureSize1, input_dim = train_matrix.shape[1], init = 'lecun_uniform', activation='sigmoid')])
	decoder1 = containers.Sequential([Dense(train_matrix.shape[1], input_dim = featureSize1, init = 'lecun_uniform', activation = 'sigmoid')])
	ae1 = Sequential()
	ae1.add(AutoEncoder(encoder = encoder1, decoder = decoder1, output_reconstruction = False))
	ae1.compile(loss = 'mean_squared_error', optimizer = SGD(lr = 0.0001, decay = 1e-6))
	ae1.fit(train_matrix, train_matrix, batch_size = 5, nb_epoch = 1)

	FirstAeOutput = ae1.predict(train_matrix)
	featureSize2 = 1000
	encoder2 = containers.Sequential([Dense(featureSize2, input_dim = FirstAeOutput.shape[1], init = 'lecun_uniform', activation = 'sigmoid')])
	decoder2 = containers.Sequential([Dense(FirstAeOutput.shape[1], input_dim = featureSize2, init = 'lecun_uniform', activation = 'sigmoid')])
	ae2 = Sequential()
	ae2.add(AutoEncoder(encoder = encoder2, decoder = decoder2, output_reconstruction = False))
	ae2.compile(loss = 'mean_squared_error', optimizer = SGD(lr = 0.01, decay = 1e-6))
	ae2.fit(FirstAeOutput, FirstAeOutput, batch_size = 5, nb_epoch = 1)

	SecondAeOutput = ae2.predict(FirstAeOutput)
	featureSize3 = 600
	encoder3 = containers.Sequential([Dense(featureSize3, input_dim = SecondAeOutput.shape[1], init = 'lecun_uniform', activation = 'sigmoid')])
	decoder3 = containers.Sequential([Dense(SecondAeOutput.shape[1], input_dim = featureSize3, init = 'lecun_uniform', activation = 'sigmoid')])
	ae3 = Sequential()
	ae3.add(AutoEncoder(encoder = encoder3, decoder = decoder3, output_reconstruction = False))
	ae3.compile(loss = 'mean_squared_error', optimizer = SGD(lr = 0.01, decay = 1e-6))
	ae3.fit(SecondAeOutput, SecondAeOutput, batch_size = 5, nb_epoch = 1)
	
	ThirdAeOutput = ae3.predict(SecondAeOutput)
	featureSize4 = 400
	encoder4 = containers.Sequential([Dense(featureSize4, input_dim = ThirdAeOutput.shape[1], init = 'lecun_uniform', activation = 'sigmoid')])
	decoder4 = containers.Sequential([Dense(ThirdAeOutput.shape[1], input_dim = featureSize4, init = 'lecun_uniform', activation = 'sigmoid')])
	ae4 = Sequential()
	ae4.add(AutoEncoder(encoder = encoder4, decoder = decoder4, output_reconstruction = False))
	ae4.compile(loss = 'mean_squared_error', optimizer = SGD(lr = 0.01, decay = 1e-6))
	ae4.fit(ThirdAeOutput, ThirdAeOutput, batch_size = 5, nb_epoch = 1)
	'''
	#build up DNN
	print 'set up neural network\n'
	
	model = Sequential()
	print "feature of layers:"
	featureSize1 = 1500
	featureSize2 = 1100
	featureSize3 = 700
	featureSize4 = 500
	featureSize5 = 100
	print featureSize1, featureSize2, featureSize3, featureSize4, featureSize5
	
	model.add(Dense(featureSize1, input_dim = train_matrix.shape[1], init = 'lecun_uniform', W_regularizer = l2(0.001), b_regularizer = l2(0.001)))
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))

	model.add(Dense(featureSize2, init = 'lecun_uniform', W_regularizer = l2(0.001), b_regularizer = l2(0.001)))
	model.add(Activation('tanh'))
	model.add(Dropout(0.5))
	
	model.add(Dense(featureSize3, init = 'lecun_uniform', W_regularizer = l2(0.001), b_regularizer = l2(0.001)))
	model.add(Activation('tanh'))
	model.add(Dropout(0.2))
	
	
	model.add(Dense(featureSize4, init = 'lecun_uniform', W_regularizer = l2(0.001), b_regularizer = l2(0.001)))
	model.add(Activation('tanh'))
	model.add(Dropout(0.1))
	
	
	model.add(Dense(featureSize5, init = 'lecun_uniform', W_regularizer = l2(0.001), b_regularizer = l2(0.001)))
	model.add(Activation('tanh'))
	model.add(Dropout(0.1))
	
	
	'''
	model = Sequential()
	model.add(ae1.layers[0].encoder)
	model.add(ae2.layers[0].encoder)
	model.add(ae3.layers[0].encoder)
	model.add(ae4.layers[0].encoder)
	'''
	model.add(Dense(7, init = 'lecun_uniform', W_regularizer = l2(0.001)))
	model.add(Activation('softmax'))

	sgd = SGD(lr = 0.01, decay = 1e-6)
	model.compile(loss = 'mean_squared_error', optimizer = sgd)
	print 'train neural network'
	model.fit(train_matrix, train_labels, nb_epoch = 20, batch_size = 10)
	score = model.evaluate(test_matrix, test_labels, batch_size = 10, show_accuracy = True)
	print 'testing error: ',
	print score
	score = model.evaluate(train_matrix, train_labels, batch_size = 10, show_accuracy = True)
	print 'trainning error:'
	print score
	prediction = model.predict_classes(test_matrix, batch_size = 1)
	prediction = np.array(prediction) + 1
	total = [0, 0, 0, 0, 0, 0, 0]
	correct = [0, 0, 0, 0, 0, 0, 0]
	for i in range(len(prediction)):
		total[test_class[i] - 1] += 1
		correct[test_class[i] - 1] += (prediction[i] == test_class[i])

	for j in range(7):
		print 1.0 * correct[j] / total[j]

	predictionR = model.predict_classes(rap_songs, batch_size = 1)
	print np.array(predictionR) + 1
	predictionC = model.predict_classes(country_songs, batch_size = 1)
	print np.array(predictionC) + 1
	return


if __name__ == '__main__':
	main()