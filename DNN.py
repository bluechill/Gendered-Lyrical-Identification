import numpy as np
from keras.models import Sequential
from keras.layers.core import Dense, Dropout, Activation
from keras.optimizers import SGD
from keras.regularizers import l2
from sklearn.feature_extraction.text import TfidfTransformer

#data is in hw1 format, return list of list of words format for word2vec use
def getData(filename, vocab, intEntry= False):
	matrix = []
	labels = []
	f = open(filename, 'r')
	lines = f.readlines()
	for song in lines[3: ]:
		song = song.split()
		if song[0] == '1':
			labels.append([1, 0, 0, 0, 0, 0, 0, 0])
		elif song[0] == '2':
			labels.append([0, 1, 0, 0, 0, 0, 0, 0])
		elif song[0] == '3':
			labels.append([0, 0, 1, 0, 0, 0, 0, 0])
		elif song[0] == '4':
			labels.append([0, 0, 0, 1, 0, 0, 0, 0])
		elif song[0] == '5':
			labels.append([0, 0, 0, 0, 1, 0, 0, 0])
		elif song[0] == '6':
			labels.append([0, 0, 0, 0, 0, 1, 0, 0])
		elif song[0] == '7':
			labels.append([0, 0, 0, 0, 0, 0, 1, 0])
		elif song[0] == '8':
			labels.append([0, 0, 0, 0, 0, 0, 0, 1])
		else:
			continue
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


def main():
	f = open('revised_dataset_test_8Genres.txt')
	vocab = f.readlines()[2].split()
	#import data
	(test_matrix, test_labels) = getData('revised_dataset_test_8Genres.txt', vocab, True)
	(train_matrix, train_labels) = getData('revised_dataset_train_8Genres.txt', vocab, True)
	
	transformer = TfidfTransformer(norm = False)
	total = np.concatenate((train_matrix, test_matrix), axis = 0)
	total = transformer.fit_transform(total)
	labelTotal = np.concatenate((train_labels, test_labels), axis = 0)
	train_matrix = total[0: total.shape[0] - 4000, :]
	train_labels = labelTotal[0: total.shape[0] - 4000, :]
	test_matrix = total[total.shape[0] - 4000: total.shape[0], :]
	test_labels = labelTotal[total.shape[0] - 4000: total.shape[0], :]
	#concatenate constant
	train_matrix = np.concatenate((train_matrix.toarray(), np.ones([train_matrix.shape[0], 1])), axis = 1)
	test_matrix = np.concatenate((test_matrix.toarray(), np.ones([test_matrix.shape[0], 1])), axis = 1)
	#build up DNN
	model = Sequential()
	model.add((Dense(1500, input_dim = train_matrix.shape[1], init = 'lecun_uniform', W_regularizer = l2(0.005),b_regularizer = l2(0.005))))
	model.add(Activation('tanh'))
	model.add(Dropout(0.25))
	model.add(Dense(1000, init = 'lecun_uniform', W_regularizer = l2(0.001)))
	model.add(Activation('tanh'))
	model.add(Dropout(0.25))
	model.add(Dense(600, init = 'lecun_uniform', W_regularizer = l2(0.001)))
	model.add(Activation('tanh'))
	model.add(Dropout(0.25))
	model.add(Dense(300, init = 'lecun_uniform', W_regularizer = l2(0.001)))
	model.add(Activation('tanh'))
	model.add(Dropout(0.25))
	model.add(Dense(8, init = 'lecun_uniform', W_regularizer = l2(0.001)))
	model.add(Activation('softmax'))

	sgd = SGD(lr = 0.01, decay = 1e-6)
	model.compile(loss = 'mean_squared_error', optimizer = sgd)

	model.fit(train_matrix, train_labels, nb_epoch = 25, batch_size = 100, verbose=2)
	score = model.evaluate(test_matrix, test_labels, batch_size = 25, show_accuracy = True, verbose=0)
	print 'testing error: ',
	print score
	score = model.evaluate(train_matrix, train_labels, batch_size = 25, show_accuracy = True, verbose=0)
	print 'trainning error:'
	print score
	return


if __name__ == '__main__':
	main()