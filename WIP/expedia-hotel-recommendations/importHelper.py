import imp

def load(name):
	path = '/media/quan/Q/github/module/' + name +'.py'
	return imp.load_source(name, path)
