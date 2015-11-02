# Takes the file mxm_dataset_test.txt or mxm_dataset_train.txt and creates
# another file formatted exactly the same, but the new file only contains
# tracks whose track id is found in another file that associates track ids
# with labels.
#
# Each line in the track_id -> label mapping file is assumed to be formatted:
#
# <track_id><space><label><\n>
#
# example:
# TRPYHPC128F930F9B0 5592
# TRGXSPB128F9345FBD 2348
# TRMQSZX12903CCC1BD 6092
# TRQECKI128F92FEEDF 2273
# ...
#
# 

import sys

def main(argv):
   if (len(argv) != 4):
      print 'Incorrect usage: >> python create_dataset_with_track_labels.py <input dataset filename> <input track label filename> <output dataset filename>'
      return


   trackFd = open(argv[1], 'r')
   labelFd = open(argv[2], 'r')

   # Read all track ids that have labels into a map
   trackIdsWithLabels = {}
   for line in labelFd.readlines():
      # Use space as the delimiter
      trackId = line[:line.index(' ')]
      trackIdsWithLabels[trackId] = 1 # Value doesn't matter, we only care if key is present

   outputFileFd = open(argv[3], 'w')

   iteration = 0
   # Scan each line in the track dataset, and write it to a new file if the track id has a label
   for line in trackFd.readlines():
      # Copy the comments in the beginning of the file
      if (line[0] == '#'):
         outputFileFd.write(line)
         continue;

      # Copy the line with the 5000 words
      if (line[0] == '%'):
         outputFileFd.write(line)
         continue;

      track_id = line[:line.index(',')]

      # only copy the line if the track id has a label from the label file
      if (track_id in trackIdsWithLabels):
         outputFileFd.write(line)

      # Periodic update
      if (iteration%10000 == 0):
         print 'line ' + str(iteration)
      iteration += 1

      
if __name__ == '__main__':
   main(sys.argv)