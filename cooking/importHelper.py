# import module in other folder

import imp

def load(name):
	path = FOLDER + name +'.py'
	return imp.load_source(name, path)
