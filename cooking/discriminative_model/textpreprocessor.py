def tokenize(s, lower=True):
	'''
	:s : pd.Series; each element as a document
	:lower : boolean; transform words in lower case if True

	:return : pd.Series; each element as a list of words after tokenization, every word in lower case
	'''
	from nltk.tokenize import word_tokenize
	if lower: return s.map(str.lower).map(word_tokenize)
	return s.map(word_tokenize)

def get_corpus(s):
	'''
	:s : pd.Series; each element as a list of words from tokenization

	:return : list; corpus from s
	'''
	l = []
	s.map(lambda x: l.extend(x))
	return l

def get_stop_words(s, n):
	'''
	:s : pd.Series; each element as a list of words from tokenization
	:n : int; n most frequent words are judged as stop words 

	:return : list; a list of stop words
	'''
	from collections import Counter
	l = get_corpus(s)
	l = [x[0] for x in Counter(l).most_common(n)]
	return l

def remove_stop_words(s, n):
	'''
	:s : pd.Series; each element as a list of words from tokenization
	:n : int; n most frequent words are judged as stop words 

	:return : pd.Series; stop words removed
	'''
	stop_words = get_stop_words(s, n)
	return s.map(lambda x: [e for e in x if e not in stop_words])

def to_tfidf(s, cv=None, tfidf=None, stop_words=None):
	'''
	:s : pd.Series; each element as a list of words after pre-processing
	:cv : CountVectorizer; if None, s will be used to create cv
	:tfidf : TfidfTransformer; if None, s will be used to create tfidf

	:return : if cv is not None: sparse matrix; Tf-idf-weighted document-term matrix
			  if cv is None: CountVectorizer; cv created by s. TfidfTransformer; tfidf created by s. sparse matrix; Tf-idf-weighted document-term matrix
	'''
	l = []
	s.map(lambda x: l.extend([x]))
	if not cv:
		from sklearn.feature_extraction.text import CountVectorizer
		from sklearn.feature_extraction.text import TfidfTransformer
		cv = CountVectorizer(stop_words=stop_words)
		tfidf = TfidfTransformer()
		return cv, tfidf, tfidf.fit_transform(cv.fit_transform(l))
	return tfidf.fit_transform(cv.fit_transform(l))



