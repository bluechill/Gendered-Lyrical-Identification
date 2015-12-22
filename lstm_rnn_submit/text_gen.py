from __future__ import print_function
from keras.models import Sequential
from keras.layers.core import Dense, Activation, Dropout
from keras.layers.recurrent import LSTM
from keras.datasets.data_utils import get_file
import numpy as np
import random
import sys

from keras.models import model_from_json

# tutorial is here:
# https://github.com/fchollet/keras/blob/master/examples
# note the one that says lstm_text_generation

###
# fifty cent lyrics generation

path = 'lyrics_fifty_cent.txt'
text = open(path).read().lower()

all_characters = set(text)
character_is = dict((c, i) for i, c in enumerate(all_characters))
is_character = dict((i, c) for i, c in enumerate(all_characters))

# cut text into subsections
maximum_length = 20
step = 3
extracted_sents = []
subseq_characters = []
for i in range(0, len(text) - maximum_length, step):
    extracted_sents.append(text[i: i + maximum_length])
    subseq_characters.append(text[i + maximum_length])

# throw characters into vectors
X = np.zeros((len(extracted_sents), maximum_length, len(all_characters)), dtype=np.bool)
y = np.zeros((len(extracted_sents), len(all_characters)), dtype=np.bool)
for i, sentence in enumerate(extracted_sents):
    for t, char in enumerate(sentence):
        X[i, t, character_is[char]] = 1
    y[i, character_is[subseq_characters[i]]] = 1

# make the model
lstm_rnn = Sequential()
lstm_rnn.add(LSTM(512, return_sequences=True, input_shape=(maximum_length, len(all_characters))))
lstm_rnn.add(Dropout(0.2))
lstm_rnn.add(LSTM(512, return_sequences=False))
lstm_rnn.add(Dropout(0.2))
lstm_rnn.add(Dense(len(all_characters)))
lstm_rnn.add(Activation('softmax'))

lstm_rnn.compile(loss='categorical_crossentropy', optimizer='rmsprop')

#this is how you load weights
lstm_rnn = model_from_json(open('fifty_cent_architecture.json').read())
lstm_rnn.load_weights('fifty_cent_weights.h5')

def unwind_text(char, diversity=1.0):
    char = np.log(char) / diversity
    char = np.exp(char) / np.sum(np.exp(char))
    return np.argmax(np.random.multinomial(1, char, 1))

# train the model, output generated text after each iteration
for iteration in range(1, 2):
    print()

    begin_i = random.randint(0,len(text)-maximum_length-1)

    diversity = 0.5
    print('rap music generation')
    print('temperature:', diversity)

    created_text = ''
    char_string = text[begin_i: begin_i + maximum_length]
    created_text += char_string
    print('seed: "' + char_string + '"')
    sys.stdout.write(created_text)

    for iteration in range(2500):
        x = np.zeros((1, maximum_length, len(all_characters)))
        for t, char in enumerate(char_string):
            x[0, t, character_is[char]] = 1.

        predicted = lstm_rnn.predict(x, verbose=0)[0]
        subseq_i = unwind_text(predicted, diversity)
        subseq_char = is_character[subseq_i]

        created_text += subseq_char
        char_string = char_string[1:] + subseq_char

        sys.stdout.write(subseq_char)
        sys.stdout.flush()
    print()

####
# Johnny cash generation

path = 'lyrics_johnny_cash.txt'
text = open(path).read().lower()

all_characters = set(text)
character_is = dict((c, i) for i, c in enumerate(all_characters))
is_character = dict((i, c) for i, c in enumerate(all_characters))

# cut text into subsections
maximum_length = 20
step = 3
extracted_sents = []
subseq_characters = []
for i in range(0, len(text) - maximum_length, step):
    extracted_sents.append(text[i: i + maximum_length])
    subseq_characters.append(text[i + maximum_length])

# throw characters into vectors
X = np.zeros((len(extracted_sents), maximum_length, len(all_characters)), dtype=np.bool)
y = np.zeros((len(extracted_sents), len(all_characters)), dtype=np.bool)
for i, sentence in enumerate(extracted_sents):
    for t, char in enumerate(sentence):
        X[i, t, character_is[char]] = 1
    y[i, character_is[subseq_characters[i]]] = 1

# make the model
lstm_rnn = Sequential()
lstm_rnn.add(LSTM(512, return_sequences=True, input_shape=(maximum_length, len(all_characters))))
lstm_rnn.add(Dropout(0.2))
lstm_rnn.add(LSTM(512, return_sequences=False))
lstm_rnn.add(Dropout(0.2))
lstm_rnn.add(Dense(len(all_characters)))
lstm_rnn.add(Activation('softmax'))

lstm_rnn.compile(loss='categorical_crossentropy', optimizer='rmsprop')

#this is how you load weights
lstm_rnn = model_from_json(open('johnny_cash_architecture.json').read())
lstm_rnn.load_weights('johnny_cash_weights.h5')

def unwind_text(char, diversity=1.0):
    char = np.log(char) / diversity
    char = np.exp(char) / np.sum(np.exp(char))
    return np.argmax(np.random.multinomial(1, char, 1))

# train the model, output generated text after each iteration
for iteration in range(1, 2):
    print()
    
    begin_i = random.randint(0,len(text)-maximum_length-1)

    diversity = 0.5
    print('country music generation')
    print('temperature:', diversity)

    created_text = ''
    char_string = text[begin_i: begin_i + maximum_length]
    created_text += char_string
    print('seed: "' + char_string + '"')
    sys.stdout.write(created_text)

    for iteration in range(2500):
        x = np.zeros((1, maximum_length, len(all_characters)))
        for t, char in enumerate(char_string):
            x[0, t, character_is[char]] = 1.

        predicted = lstm_rnn.predict(x, verbose=0)[0]
        subseq_i = unwind_text(predicted, diversity)
        subseq_char = is_character[subseq_i]

        created_text += subseq_char
        char_string = char_string[1:] + subseq_char

        sys.stdout.write(subseq_char)
        sys.stdout.flush()
    print()
    
###
# combined johnny cash and 50 cent generation

path = 'lyrics_combined_50_cent_johnny_cash.txt'
text = open(path).read().lower()

all_characters = set(text)
character_is = dict((c, i) for i, c in enumerate(all_characters))
is_character = dict((i, c) for i, c in enumerate(all_characters))

# cut text into subsections
maximum_length = 20
step = 3
extracted_sents = []
subseq_characters = []
for i in range(0, len(text) - maximum_length, step):
    extracted_sents.append(text[i: i + maximum_length])
    subseq_characters.append(text[i + maximum_length])

# throw characters into vectors
X = np.zeros((len(extracted_sents), maximum_length, len(all_characters)), dtype=np.bool)
y = np.zeros((len(extracted_sents), len(all_characters)), dtype=np.bool)
for i, sentence in enumerate(extracted_sents):
    for t, char in enumerate(sentence):
        X[i, t, character_is[char]] = 1
    y[i, character_is[subseq_characters[i]]] = 1

# make the model
lstm_rnn = Sequential()
lstm_rnn.add(LSTM(512, return_sequences=True, input_shape=(maximum_length, len(all_characters))))
lstm_rnn.add(Dropout(0.2))
lstm_rnn.add(LSTM(512, return_sequences=False))
lstm_rnn.add(Dropout(0.2))
lstm_rnn.add(Dense(len(all_characters)))
lstm_rnn.add(Activation('softmax'))

lstm_rnn.compile(loss='categorical_crossentropy', optimizer='rmsprop')

#this is how you load weights
lstm_rnn = model_from_json(open('combined_architecture.json').read())
lstm_rnn.load_weights('combined_weights.h5')

def unwind_text(char, diversity=1.0):
    char = np.log(char) / diversity
    char = np.exp(char) / np.sum(np.exp(char))
    return np.argmax(np.random.multinomial(1, char, 1))

# train the model, output generated text after each iteration
for iteration in range(1, 2):
    print()

    begin_i = random.randint(0,len(text)-maximum_length-1)

    diversity = 0.5
    print('rap and country mixed generation')
    print('temperature:', diversity)

    created_text = ''
    char_string = text[begin_i: begin_i + maximum_length]
    created_text += char_string
    print('seed: "' + char_string + '"')
    sys.stdout.write(created_text)

    for iteration in range(2500):
        x = np.zeros((1, maximum_length, len(all_characters)))
        for t, char in enumerate(char_string):
            x[0, t, character_is[char]] = 1.

        predicted = lstm_rnn.predict(x, verbose=0)[0]
        subseq_i = unwind_text(predicted, diversity)
        subseq_char = is_character[subseq_i]

        created_text += subseq_char
        char_string = char_string[1:] + subseq_char

        sys.stdout.write(subseq_char)
        sys.stdout.flush()
    print()
