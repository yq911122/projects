import imp

def load(name):
	path = '../module/' + name +'.py'
	return imp.load_source(name, path)
