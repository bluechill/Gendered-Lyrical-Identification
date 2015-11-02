import sys
import lat_long_zone_fns as llz


def generateTrackLocationFile(input_data_file_name, output_data_file_name, output_label_file_name, filter_func, label_func, num_data_points_needed):
   # Read in the necessary files
   track_fd = open(input_data_file_name, 'r')
   matches_fd = open('mxm/mxm_779k_matches.txt', 'r')
   artist_fd = open('mxm/unique_artists.txt', 'r')
   location_fd = open('mxm/artist_location.txt', 'r')


   # Create artist id -> lat long zone map with locations file
   print 'creating locations map...'
   artist_locations = {}
   for line in location_fd.readlines():
      # Grab the artist id, lat, and long
      sep1_idx = line.index("<SEP>", 0)
      sep2_idx = line.index("<SEP>", sep1_idx+5)
      sep3_idx = line.index("<SEP>", sep2_idx+5)

      artist_id = line[0:sep1_idx]
      latitude = float(line[sep1_idx+5:sep2_idx])
      longitude = float(line[sep2_idx+5:sep3_idx])

      # Only keep track of this location info based on passed in functions
      if (filter_func(latitude, longitude)):
         artist_locations[artist_id] = label_func(latitude, longitude)


   # Create artist name -> artist id map w/ unique artist file
   duplicate_artist_names = {}
   print 'creating artist map...'
   artist_ids = {}
   for line in artist_fd.readlines():
      # Grab the artist id, and name
      sep1_idx = line.index("<SEP>", 0)
      seplast_idx = line.rindex("<SEP>")

      artist_id = line[0:sep1_idx]
      artist_name = line[seplast_idx+5:len(line) - 1]

      # Detect duplicate artists
      if (artist_name in artist_ids):
         if (artist_name in duplicate_artist_names):
            duplicate_artist_names[artist_name] += 1
         else:
            duplicate_artist_names[artist_name] = 2
      else:
         artist_ids[artist_name] = artist_id

   # Delete all of the artists that have duplicated names
   # since we will not be able to disambiguate which artist is which
   for name in duplicate_artist_names:
      artist_ids.pop(name, None)


   # Create track id -> MSD artist name map w/ matches file
   print 'creating track map...'
   track_artists = {}
   for line in matches_fd.readlines():
      # Skip beginning comment lines
      if (line[0] == '#'):
         continue

      # Grab the track id, and artist name
      sep1_idx = line.index("<SEP>", 0)
      sep2_idx = line.index("<SEP>", sep1_idx+5)

      track_id = line[0:sep1_idx]
      artist_name = line[sep1_idx+5:sep2_idx]

      track_artists[track_id] = artist_name

   # Keep track of stats when we cant find things
   matches_not_found = 0
   artists_not_found = 0
   locations_not_found = 0

   total_tracks = 0
   track_locations_list = []

   output_data_fd = open(output_data_file_name, 'w')

   # Go through each track in the given track file
   for line in track_fd.readlines():
      # Skip the comments in the beginning of the file
      if (line[0] == '#'):
         output_data_fd.write(line)
         continue

      # Skip the line with the 5000 words
      if (line[0] == '%'):
         output_data_fd.write(line)
         continue

      track_id = line[:line.index(',')]
      total_tracks += 1

      # Check if each key exists and map all the way to the location info
      if (track_id in track_artists):
         if (track_artists[track_id] in artist_ids):
            if (artist_ids[track_artists[track_id]] in artist_locations):
               track_locations_list.append((track_id, artist_locations[artist_ids[track_artists[track_id]]]))
               output_data_fd.write(line)
               num_data_points_needed -= 1
            else:
               locations_not_found += 1
         else:
            artists_not_found += 1
      else:
         matches_not_found += 1

      # Stop if we got all the data points we need
      if (num_data_points_needed == 0):
         break

   # At the end, write the results to disk
   clean_result_file = open(output_label_file_name, 'w')
   for entry in track_locations_list:
      clean_result_file.write(entry[0] + ' ' + str(entry[1]) + '\n')

   clean_result_file.close();
   output_data_fd.close();

   # Display some statistics
   total_not_found = matches_not_found + artists_not_found + locations_not_found
   total_found = total_tracks - total_not_found
   percent_found = total_found*100/total_tracks
   print 'total tracks examined: ' + str(total_tracks)
   print 'total tracks with location data: ' + str(total_found) + ' (' + str(percent_found) + '%)'
   print '------- Breakdown of missing data --------'
   print 'matches not found: ' + str(matches_not_found)
   print 'artists not found: ' + str(artists_not_found)
   print 'locations not found: ' + str(locations_not_found)
   print '------- Other data --------'
   print 'Number of duplicated artist names: ' + str(len(duplicate_artist_names))

def main(argv):
   if (len(argv) < 5):
      print 'Incorrect usage: >> python song_location_matcher.py <[all|US]> <dataset input filename> <dataset output filename> <label output filename> <num data points (optional)'
      return

   num_data_points = sys.maxint
   # The user specified how many data points they want
   if (len(argv) == 6 and int(argv[5]) > 0):
      num_data_points = int(argv[5])

   # Supply different filter arguments based on user input
   if (argv[1] == 'all'):
      generateTrackLocationFile(argv[2], argv[3], argv[4], llz.all, llz.LatLongToZoneNum, num_data_points)
   elif (argv[1] == 'US'):
      generateTrackLocationFile(argv[2], argv[3], argv[4], llz.isWithinUnitedStates, llz.unitedStatesLatLongToZoneNum, num_data_points)
   else:
      print 'Invalid filter given: ' + str(argv[1])


if __name__ == "__main__":
   main(sys.argv)