from nltk import word_tokenize 
from nltk.stem.porter import *
import csv

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
# from sklearn import cross_validation 
# from sklearn.multiclass import OneVsRestClassifier
from sklearn.cross_validation import StratifiedShuffleSplit
from sklearn.grid_search import GridSearchCV

from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier

import numpy as np
from random import shuffle
from scipy.sparse import csr_matrix

import re


def clean(text):
	return cleanQuote(cleanRef(cleanAt(cleanUrl(text))))

def cleanUrl(text):
	return  re.sub(r'^https?:\/\/.*[\r\n]*', '', text, flags=re.MULTILINE)

def cleanAt(text):
	return  re.sub(r'@\w+', '@', text, flags=re.MULTILINE)

def cleanRef(text):
	return  re.sub(r'#\w+', '#', text, flags=re.MULTILINE)

def cleanQuote(text):
	return  re.sub(r'"', '', text, flags=re.MULTILINE)


with open ('train2.csv','rU') as IN:
	PosOut = open('pos.csv','wb')
	NegOut = open('neg.csv','wb')
	PosWriter = csv.writer(PosOut)
	NegWriter = csv.writer(NegOut)

	reader = csv.reader(IN, delimiter=',')
	for row in reader:
		pol = int(row[0])
		if pol == 4:
			PosWriter.writerow([clean(row[5])])
		elif pol == 0:
			NegWriter.writerow([clean(row[5])])
	PosOut.close()
	NegOut.close()
	IN.close()


