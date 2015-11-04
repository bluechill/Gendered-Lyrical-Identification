import sys 


def generateTrackSongArtist(input_track_data_file,output_data_file,output_label_file):
	#read in the files 
	input_track = open(input_track_data_file,'r')
	trackid_file = open('mxm/unique_tracks.txt','r')
	artist_file = open('mxm/unique_artists.txt', 'r')

	print 'create artist_name to id'
	artist_ids = {}

	for line in artist_file.readlines():
		sep_index1 = line.index('<SEP>',0)
		sep_indexr = line.rindex("<SEP>")

		artist_id = line[0:sep_index1]
		artist_name = line[sep_indexr+5:len(line)-1]

		artist_ids[artist_name] = artist_id
	artist_file.close()


	print 'create track id to song mapping'

	track_song = {}
	track_artist = {}
	# map for song names -> song ids
	song_ids = {}

	for line in trackid_file.readlines():
		sep_index1 = line.index('<SEP>',0)
		sep_index2 = line.index('<SEP>',sep_index1+5)
		sep_index3 = line.index('<SEP>',sep_index2+5)

		track_id = line[0:sep_index1]
		song_id = line[sep_index1+5:sep_index2]
		artist_name = line[sep_index2+5:sep_index3]
		song_name = line[sep_index3+5:]

		track_song[track_id] = song_id
		track_artist[track_id] = artist_name
		song_ids[song_name] = song_id

	trackid_file.close()

	track_song_artist_list = []

	output_data_f = open(output_data_file,'w')
	for line in input_track.readlines():
		if(line[0] == '#'):
 			output_data_f.write(line)
			continue
		if(line[0] == '%'):
			output_data_f.write(line)
			continue
		track_id = line[0:line.index(',')]

		if(track_id in track_song):
			if(track_id in track_artist):
				if(track_artist[track_id] in artist_ids):
					track_song_artist_list.append((track_id,track_song[track_id],artist_ids[track_artist[track_id]]))
					output_data_f.write(line)
	output_data_f.close()

	output_label_f = open(output_label_file,'w')
	for entry in track_song_artist_list:
		output_label_f.write(entry[0]+' '+entry[1]+' '+entry[2]+'\n')
	output_label_f.close()


def main(argv):
	if(len(argv)<2):
		print 'need to specify output_label_file'
		return 
	generateTrackSongArtist(argv[1],argv[2],argv[3])

if __name__ == '__main__':
	main(sys.argv)


