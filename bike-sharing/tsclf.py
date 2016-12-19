import numpy as np
import pandas as pd

class GroupRegressor(object):
	"""docstring for GroupRegressor"""
	def __init__(self, grpCols=None, y_transform=None, subModel=None):
		super(GroupRegressor, self).__init__()
		if not grpCols: self.grpCols = grpCols

		if not y_transform: self.y_transform = self._identical_transform
		else: self.y_transform = y_transform

		if not subModel: subModel = np.mean

		try: 
			self.subfit = subModel.fit
			self.subpredict = lambda m, x: m.predict(x)
		except AttributeError:
			self.subfit = lambda x, y: subModel(y)
			self.subpredict = lambda m, x: m
		# self.SSR = 0.0

	def _identical_transform(self, e):
		return e

	def fit(self, X, y):
		assert (self.grpCols is not None), "no group factors are set. Use other models or set group factors before fitting the data."
		xCols = [e for e in X.columns if e not in self.grpCols]

		self.df = X.copy()
		self.df.loc[:,'y'] = y.copy()
		self.df['y'] = self.y_transform(self.df['y'])

		self.df = self.df.groupby(self.grpCols).apply(lambda e: self.subfit(e[xCols], e['y']))

		self.df = pd.DataFrame(self.df)
		# self.df.drop([e for e in self.grpCols if e in self.df.columns], axis=1, inplace=True)

		self.df.columns = ['y']
		self.df.reset_index(inplace=True)
		self.df['grp'] = self.df[self.grpCols].apply(lambda x: row_to_tuple(x), axis=1)
		self.xCols = xCols


	def predict(self, X):
		grpCols = [e for e in X.columns if e in self.grpCols]
		assert (len(grpCols) == len(self.grpCols)), "inconsist grouping features!"
		test = X.copy()
		groups = test.groupby(grpCols)[self.xCols]
		print self.df.head()

		pred = []
		for _, x in self.df.iterrows():
			try:
				g = groups.get_group(x.loc['grp'])
				print g.head()
			except KeyError:
				continue
			pred_subgroup = g.apply(lambda e: self.subpredict(x.loc['y'], e), axis=1)
			# print pred_subgroup.head()
			pred.append(pred_subgroup)
		pred = pd.concat(pred)

		return pred

	def setGroupCols(self, cols):
		self.grpCols = cols


def row_to_tuple(e):
	return tuple(e.values.tolist())