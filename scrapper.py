import urllib
import re
import sys
from HTMLParser import HTMLParser

class MyHTMLParser(HTMLParser):
	title = ""
	artist = ""
	lyrics = []
	writers = []
	watchForLyrics = 0
	
	artistPattern = re.compile("ArtistName = \"(?P<artist>[A-Z 0-9]+)\";\nSongName = \"(?P<song>[A-Za-z 0-9]*)\";.", re.VERBOSE)
	
	def handle_starttag(self, tag, attrs):
		if self.watchForLyrics == 1 and tag == 'br':
			self.watchForLyrics = 2
		elif self.watchForLyrics == 2 and tag == 'br':
			self.watchForLyrics = 3
		elif self.watchForLyrics == 3 and tag == 'div':
			self.watchForLyrics = 4
	
	def handle_endtag(self, tag):
		if self.watchForLyrics == 4 and tag == 'div':
			self.watchForLyrics = 0
#			print "Lyrics:", self.lyrics

	def handle_data(self, data):
		result = re.findall(r"([\[A-Za-z0-9()][A-Za-z0-9() ',.:]*[A-Za-z0-9()'\]])+", data)
		result2 = re.findall(r"([\[A-Za-z0-9()][A-Za-z0-9() ']*[A-Za-z0-9()'\]])+", data)
#		print "data:", result
#		print "data:", result2
		if (len(result) >= 4 and result[0] == 'ArtistName' and result[2] == 'SongName'):
#			print "Artist Title: ", result[1]
#			print "Song Name: ", result[3]
			self.artist = result[1]
			self.title = result[3]
		elif (len(result2) >= 2 and result2[0] == 'Writer(s)'):
			for x in range(1,len(result2)):
				self.writers.append(result2[x])
#			print "Writers:", self.writers
		elif (self.watchForLyrics == 0 and len(result) >= 1 and result[0] == self.title and len(self.lyrics) == 0):
			self.watchForLyrics = 1
		elif (self.watchForLyrics == 4 and len(result) >= 1):
			for x in range(0,len(result)):
				self.lyrics.append(result[x])

if len(sys.argv) < 2:
	print 'Usage: ', sys.argv[0], ' [file url]'
	exit(1)

page = urllib.urlopen(sys.argv[1])
parser = MyHTMLParser()
parser.feed(page.read())

print parser.artist
print parser.title

for x in parser.writers:
	sys.stdout.write(x + ",")

print ""

for x in parser.lyrics:
	if "[" in x: continue
	
	for y in x.split(" "):
		print y

	print "$newline$"
