import csv
from nltk.tokenize import wordpunct_tokenize
from nltk.tokenize import word_tokenize
import re


pattern=re.compile("[\"\[\],]")

with open('date_totalemo_tweetnum_avgemo_pre.csv','wb') as out:
	with open('date_totalemo_tweetnum_avgemo','rU') as inputfile:
		for row in inputfile.readlines():
			row = pattern.sub('', row)
			row = row.replace(' ', ',')
			row = row.replace('\t', ',')
			out.write(row)
	inputfile.close()
out.close()

pattern=re.compile("[\[\]]")

with open('tweetlen_emodis_pre.csv','wb') as out:
	with open('tweetlen_emodis','rU') as inputfile:
		for row in inputfile.readlines():
			row = pattern.sub('', row)
			row = row.replace('\t', ',')
			l = row.split(',')
			try:
				avg_emo = float(l[3])*float(l[4])/(float(l[2])+float(l[4]))
				tweetnum = int(l[2])+int(l[4])
			except IndexError:
				avg_emo = float(l[1])
				tweetnum = int(l[2])
			out.write(','.join([l[0],str(avg_emo),str(tweetnum)])+ '\n')
	inputfile.close()
out.close()

pattern=re.compile("[\"\[\]]")

with open('usr_avgtweetlen_avgemo_tweetnum_pre.csv','wb') as out:
	with open('usr_avgtweetlen_avgemo_tweetnum','rU') as inputfile:
		for row in inputfile.readlines():
			row = pattern.sub('', row)
			row = row.replace('\t', ',')
			out.write(row)
	inputfile.close()
out.close()

pattern=re.compile("[\"]")

with open('user_avgat_pre.csv','wb') as out:
	with open('user_avgat','rU') as inputfile:
		for row in inputfile.readlines():
			row = pattern.sub('', row)
			row = row.replace('\t', ',')
			out.write(row)
	inputfile.close()
out.close()