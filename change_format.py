import sys,os 
import re 


revised_file = open('revised_dataset_test.txt','r')
output_file = open('revised_dataset_test_7Genres.txt','w')
trackid_to_genre = open('trackid_to_genre.txt','r')

trackid_to_genre_map = {}

for line in trackid_to_genre.readlines():
	line = line.split()
	if line[1] == 'Pop_Rock':
		trackid_to_genre_map[line[0]] = 1
		continue
	if line[1] == 'Electronic':
		trackid_to_genre_map[line[0]] = 2
		continue
	if line[1] == 'Rap':
		trackid_to_genre_map[line[0]] = 3
		continue
	if line[1] == 'Country':
		trackid_to_genre_map[line[0]] = 4
		continue
	if line[1] == 'Latin':
		trackid_to_genre_map[line[0]] = 5
		continue
	if line[1] == 'RnB':
		trackid_to_genre_map[line[0]] = 6
		continue
	if line[1] == 'Jazz':
		trackid_to_genre_map[line[0]] = 7
		continue
	if line[1] == 'International':
		trackid_to_genre_map[line[0]] = 8
		continue
	continue
trackid_to_genre.close()

total_count = 0
document = []
words = []
count_1 = 0

for line in revised_file.readlines():
	if line[0] == '#':
		continue
	if line[0] == '%':
		word_line = line[1:]
		total_count = len(word_line.split(','))
		words = ' '.join(word_line.split(','))
		continue
	comma_index1 = line.index(',',0)
	comma_index2 = line.index(',',comma_index1+1)

	track_id = line[0:comma_index1]
	
	try:
		genre_id = trackid_to_genre_map[track_id]
		if genre_id == 1:
			if count_1 > 800:
				continue
			else:
				count_1 +=1
	except KeyError:
		continue

	genre = str(genre_id)+' '
	word_frequency = line[comma_index2+1:]
	context = re.sub('(\:)|(\,)',' ',word_frequency)
	new_line = genre + context
	document.append(new_line)

revised_file.close()
output_file.write('DOC_WORD_MATRIX_TRAIN'+'\n')
total_entry = len(document)
output_file.write(str(total_entry)+' '+str(total_count)+'\n')
output_file.write(words)


for line in document:
	output_file.write(line)
output_file.close()