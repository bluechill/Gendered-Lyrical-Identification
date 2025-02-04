import numpy as np
from string import punctuation

def extract_words(input_string):
    for c in punctuation:
        input_string = input_string.replace(c, ' ')
    array = input_string.split()
    array = array[2:]
    occurences = array[1::2]
    words = array[0::2]
    return words, occurences

def extract_track_id(input_string):
    for c in punctuation:
        input_string = input_string.replace(c, ' ')
    array = input_string.split()
    track_id = array[0]
    return track_id

# files are in format:
# field0 field1
# track_id genre/location
def extract_field1s(input_string):
    for c in punctuation:
        input_string = input_string.replace(c, ' ')
    array = input_string.split()
    field1 = array[1]
    return field1
    
def get_track_ids(infile):
    num_lines = sum(1 for line in open(infile,'rU'))
    track_ids = ['Empty']*num_lines
    
    f = open(infile, 'rU')
    idx = 0
    for line in f:
        track_id = extract_track_id(line)
        track_ids[idx] = track_id
        idx = idx + 1
    track_ids = np.array(track_ids, dtype=object)
    np.save(infile+'_track_ids', track_ids)
    f.close()
    return track_ids

# files are in format:
# field0 field1
# track_id genre/location
def get_field1s(infile):
    num_lines = sum(1 for line in open(infile,'rU'))
    field_1s = ['Empty']*num_lines
    
    f = open(infile, 'rU')
    idx = 0
    for line in f:
        field_1 = extract_field1s(line)
        field_1s[idx] = field_1
        idx = idx + 1
    field_1s = np.array(field_1s)
    np.save(infile+'_field_1s', field_1s)
    f.close()
    return field_1s

def extract_feature_vectors(infile, num_words):
    num_lines = sum(1 for line in open(infile,'rU'))
    feature_matrix = np.zeros([num_lines, num_words])

    f = open(infile, 'rU')
    idx = 0
    for line in f:
        words, occurences = extract_words(line)
        word_idx = 0
        for word in words:
            # for some reason the database is 1 indexed
            word = int(word)-1
            feature_matrix[idx, word] = occurences[word_idx]
            word_idx = word_idx + 1
        print(feature_matrix[idx])
        idx = idx + 1
    np.save(infile+'_matrix', feature_matrix)
    f.close()
    return feature_matrix

def find_num_words(infile):
    f = open(infile, 'rU')
    for line in f:
        for c in punctuation:
            line = line.replace(c, ' ')
        array = line.split()
    return len(array)

def main():
    # num_words = find_num_words('song_words_train.txt')
    # print(num_words)
    # track_ids = get_track_ids('songs_features_train.txt')
    # print track_ids
    num_words = 5001
    field_1s = get_field1s('song_location_labels_train.txt')
    dictionary = extract_feature_vectors('song_location_features_train.txt', num_words)


    
main()
