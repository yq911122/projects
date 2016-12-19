from mrjob.job import MRJob
from mrjob.protocol import RawValueProtocol

import csv, re

from nltk.tokenize import word_tokenize

from collections import Counter

class MRTimeDateJob(MRJob):


	def mapper(self, _, line):
		if line:
			a = csv.reader(line)
			l = list(a)
			l = [item for sublist in l for item in sublist if item]
			emo = int(l[0])
			time = l[2].split(' ')
			weekday = time[0]
			hour = time[3].split(':')[0]

			timeslot = ' '.join((weekday,hour))
			
			yield timeslot, emo
		else:
			yield 0,-1

	def reducer(self, key, values):
		l = list(values)
		ct = len(l)
		sumemo = sum(l)

		# yield key, (sum(values)/float(ct), sum(values), ct)
		yield key, (sumemo, ct, sumemo/float(ct))

class MRTweetLenJob(MRJob):

	def mapper(self, _, line):
		if line:
			a = csv.reader(line)
			l = list(a)
			l = [item for sublist in l for item in sublist if item]
			emo = int(l[0])

			tweet = l[-1].decode('utf-8')
			tweetlen = len(word_tokenize(tweet))
			
			yield tweetlen, emo
		else:
			yield 0,-1

	def reducer(self, key, values):
		l = list(values)
		counter = Counter(l)
		yield key, counter.items()

class MRUsrJob(MRJob):


	def mapper(self, _, line):
		if line:
			a = csv.reader(line)
			l = list(a)
			l = [item for sublist in l for item in sublist if item]
			emo = int(l[0])

			tweet = l[-1].decode('utf-8')
			tweetlen = len(word_tokenize(tweet))
			
			usr = l[4]
			yield usr, (tweetlen, emo)
		else:
			yield 0,-1

	def reducer(self, key, values):
		l = list(values)
		tweets = float(len(l))
		avg_tweetlen = sum([x[0] for x in l])/tweets
		avg_emo = sum([x[1] for x in l])/tweets
		# yield key, (sum(values)/float(ct), sum(values), ct)
		yield key, (avg_tweetlen, avg_emo, tweets)

class MRAtJob(MRJob):


	def mapper(self, _, line):
		if line:
			a = csv.reader(line)
			l = list(a)
			l = [item for sublist in l for item in sublist if item]

			tweet = l[-1].decode('utf-8')
			
			usr = l[4]

			at = re.findall(r'@\w+',tweet)

			yield usr, len(at)
		else:
			yield 0,-1

	def reducer(self, key, values):
		l = list(values)
		avg_at = sum(l)/float(len(l))
		# yield key, (sum(values)/float(ct), sum(values), ct)
		yield key, avg_at

if __name__ == '__main__':
	MRAtJob.run()