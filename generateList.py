import urllib
import re
import time
import sys
import random
from HTMLParser import HTMLParser

class ArtistParser(HTMLParser):
	artists = []
	prefix = ""
	
	artistPattern = re.compile("ArtistName = \"(?P<artist>[A-Z 0-9]+)\";\nSongName = \"(?P<song>[A-Za-z 0-9]*)\";.", re.VERBOSE)
	
	def handle_starttag(self, tag, attrs):
#		print "Start: ", tag, attrs

		if len(attrs) >= 1 and ("href" in attrs[0]) and attrs[0][1].startswith(self.prefix):
			self.artists.append("http://www.azlyrics.com/" + attrs[0][1])
	
#	def handle_endtag(self, tag):
#		print "End: ", tag

#	def handle_data(self, data):
#		print "data:", data

class SongLoader(HTMLParser):
	songs = []

	def handle_starttag(self, tag, attrs):
		if len(attrs) >= 1 and ("href" in attrs[0]) and attrs[0][1].startswith("../lyrics/"):
			songs.append("http://www.azlyrics.com/" + attrs[0][1][3:])

class SongParser(HTMLParser):
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

artists = []
lists = ["http://www.azlyrics.com/19.html","http://www.azlyrics.com/a.html","http://www.azlyrics.com/b.html","http://www.azlyrics.com/c.html","http://www.azlyrics.com/d.html","http://www.azlyrics.com/e.html","http://www.azlyrics.com/f.html","http://www.azlyrics.com/g.html","http://www.azlyrics.com/h.html","http://www.azlyrics.com/i.html","http://www.azlyrics.com/j.html","http://www.azlyrics.com/k.html","http://www.azlyrics.com/l.html","http://www.azlyrics.com/m.html","http://www.azlyrics.com/n.html","http://www.azlyrics.com/o.html","http://www.azlyrics.com/p.html","http://www.azlyrics.com/q.html","http://www.azlyrics.com/r.html","http://www.azlyrics.com/s.html","http://www.azlyrics.com/t.html","http://www.azlyrics.com/u.html","http://www.azlyrics.com/v.html","http://www.azlyrics.com/w.html","http://www.azlyrics.com/x.html","http://www.azlyrics.com/y.html","http://www.azlyrics.com/z.html"]

parser = ArtistParser()

for l in lists:
	parser.prefix = l[l.rindex("/")+1:l.rindex(".html")] + "/"
	page = urllib.urlopen(l)
	parser.feed(page.read())
	
	artists.extend(parser.artists)
	page.close()
	time.sleep(5 + (random.random() % 5 - 2))
	print ".",

parser2 = SongLoader()

songs = []
print ""

for a in artists:
	page = urllib.urlopen(a)
	parser2.feed(page.read())

	songs.extend(parser2.songs)
	page.close()
	time.sleep(5 + (random.random() % 5 - 2))
	print ".",

parser3 = SongParser()
print ""

for s in songs:
	page = urllib.urlopen(s)
	parser3.feed(page.read())

	file = open(parser3.artist + " - " + parser3.title + ".txt", "w")

	print >> file, parser3.artist
	print >> file, parser3.title

	for x in parser3.writers:
		file.write(x + ",")

	print >> file, ""

	for x in parser3.lyrics:
		if "[" in x:
			continue
	
		for y in x.split(" "):
			print >> file, y

	print >> file, "$newline$"
	file.close()
	page.close()
	time.sleep(5 + (random.random() % 5 - 2))
	print ".",

print ""
print "Done"
