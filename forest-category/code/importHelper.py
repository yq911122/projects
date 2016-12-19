import imp

def load(name):
	path = 'D:/github/module/' + name +'.py'
	return imp.load_source(name, path)
