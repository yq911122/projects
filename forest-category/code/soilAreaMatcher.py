import pandas as pd
import numpy as np


URL = '../soil_area_match.csv'
def load_data(url = URL):
	df = pd.read_csv(url,index_col=0)
	soils = df.groupby(['Cover_Type','Soil_Type']).size()
	soils = soils.unstack('Cover_Type')
	soils = soils.fillna(0)

	areas = df.groupby(['Cover_Type','Wilderness_Area']).size()
	areas = areas.unstack('Cover_Type')
	areas = areas.fillna(0)
	# areas.loc[3,:] = areas.loc[4,:]
	# areas = areas.loc[[1,2,3],:]

	return soils, areas

class matcher(object):
	"""docstring for matcher"""


	def __init__(self, area, soil):
		'''
		area:
		Cover_Type		  1	 2	 3	 4	 5	 6	 7
		Wilderness_Area										  
		1				1062  1134	 0	 0   856	 0   545
		2				 181	66	 0	 0	 0	 0   252
		3				 917   940   863	 0  1304   962  1363
		4				   0	20  1297  2160	 0  1198	 0

		soil:
		Cover_Type	1	2	3	4	5	 6	7
		Soil_Type									 
		1			 0	0  121  139	0	95	0
		2			 0	3  283   94   61   182	0
		3			 0   12  133  799	0	18	0
		4			 5   20  462  133  129	87	7
		5			 0	0   55   39	0	71	0
		6			 0	7  248  244	0   151	0
		8			 0	1	0	0	0	 0	0
		9			 1	9	0	0	0	 0	0
		10			9   81  717  170   64  1101	0
		11			5   67   89   24  154	67	0
		12		   24  203	0	0	0	 0	0
		13		   17   84	4	0  305	66	0
		14			0	0	4  128	0	37	0
		16			9   14	5   40	9	37	0
		17			2	7   34  350  131	88	0
		18			0   16	0	0   44	 0	0
		19		   15   13	0	0   18	 0	0
		20		   41   55	0	0	6	37	0
		21		   12	0	0	0	0	 0	4
		22		  275   54	0	0	0	 0   16
		23		  376  149	0	0  157	 3   72
		24		  128   72	0	0   11	15   31
		25			0	1	0	0	0	 0	0
		26			7   19	0	0   28	 0	0
		27			7	5	0	0	0	 0	3
		28			0	6	0	0	3	 0	0
		29		  407  554	0	0  254	 0   76
		30		   81  144	0	0  480	 0   20
		31		  114   97	0	0   87	 7   27
		32		  230  255	5	0  100	30   70
		33		  184  184	0	0  115	66   67
		34			0   12	0	0	4	 2	4
		35			3	1	0	0	0	 0   98
		36			0	2	0	0	0	 0	8
		37			0	0	0	0	0	 0   34
		38		   80	7	0	0	0	 0  641
		39		   79	3	0	0	0	 0  575
		40		   49	3	0	0	0	 0  407
		'''
		super(matcher, self).__init__()
		self.a = area
		self.s = soil
		self.a_w = self.a.sum(1)/self.a.sum().sum()
		self.counter = 0

	def eval_classifer(self, individual):
		# self.counter += 1
		# if self.counter % 100 == 0: print self.counter / 4000
		# print(0 in individual)
		self.s['area'] = list(individual)

		area = self.s.groupby('area').sum()
		areadiff = abs(area - self.a)
		areadiff = areadiff.replace(np.inf, 0)
		areadiff = areadiff.fillna(0)

		score = areadiff.sum().sum()
		# print score
		# print sum(individual)
		# print score
		return score,
	
	def train(self, pop = 20, gen = 10):
		from deap import algorithms
		from deap import base
		from deap import creator
		from deap import tools
		from deap.tools import Statistics
		# import random
		

		from scipy.stats import rv_discrete

		# creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
		# creator.create("Individual", list, fitness=creator.FitnessMulti)

		creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
		creator.create("Individual", list, fitness=creator.FitnessMin)

		toolbox = base.Toolbox()
		# Attribute generator
		custm = rv_discrete(name='custm', values=(self.a_w.index, self.a_w.values))

		toolbox.register("attr_int", custm.rvs)
		# Structure initializers
		toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_int, n=len(self.s))
		toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=pop)

		# Operator registering
		toolbox.register("evaluate", self.eval_classifer)
		toolbox.register("mate", tools.cxUniform, indpb=0.5)
		toolbox.register("mutate", tools.mutUniformInt, low=min(self.a.index), up=max(self.a.index), indpb=0.1)
		toolbox.register("select", tools.selNSGA2)

		MU, LAMBDA = pop, pop
		population = toolbox.population(n=MU)
		hof = tools.ParetoFront()
		
		s = Statistics(key=lambda ind: ind.fitness.values)
		s.register("mean", np.mean)
		s.register("min", min)

		# pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=MU, lambda_=LAMBDA, cxpb=0.7, mutpb=0.3, ngen=gen, stats=s, halloffame=hof)
		for i in range(gen):
			offspring = algorithms.varAnd(population, toolbox, cxpb=0.95, mutpb=0.1)
			fits = toolbox.map(toolbox.evaluate, offspring)
			for fit, ind in zip(fits, offspring):
				ind.fitness.values = fit

			population = tools.selBest(offspring, int(0.05*len(offspring))) + tools.selTournament(offspring, len(offspring)-int(0.05*len(offspring)), tournsize=3)
			# population = toolbox.select(offspring, k=len(population))
			print s.compile(population)
		top10 = tools.selBest(population, k=10)
		return top10
		# print hof
		# print logbook
		# return hof[-1]

def cal_error_by_cat(part, df1, df2):
	'''
	part = [4, 3, 4, 3, 3, 4, 1, 1, 4, 3, 1, 3, 4, 3, 4, 3, 3, 3, 1, 1, 1, 3, 1, 3, 1, 3, 1, 1, 3, 3, 3, 3, 3, 1, 3, 3, 3, 1]
	df1:
	Cover_Type		  1	 2	 3	 4	 5	 6	 7
	Wilderness_Area										  
	1				1062  1134	 0	 0   856	 0   545
	2				 181	66	 0	 0	 0	 0   252
	3				 917   940   863	 0  1304   962  1363
	4				   0	20  1297  2160	 0  1198	 0

	df2:
	Cover_Type	1	2	3	4	5	 6	7
	Soil_Type									 
	1			 0	0  121  139	0	95	0
	2			 0	3  283   94   61   182	0
	3			 0   12  133  799	0	18	0
	4			 5   20  462  133  129	87	7
	5			 0	0   55   39	0	71	0
	6			 0	7  248  244	0   151	0
	8			 0	1	0	0	0	 0	0
	9			 1	9	0	0	0	 0	0
	10			9   81  717  170   64  1101	0
	11			5   67   89   24  154	67	0
	12		   24  203	0	0	0	 0	0
	13		   17   84	4	0  305	66	0
	14			0	0	4  128	0	37	0
	16			9   14	5   40	9	37	0
	17			2	7   34  350  131	88	0
	18			0   16	0	0   44	 0	0
	19		   15   13	0	0   18	 0	0
	20		   41   55	0	0	6	37	0
	21		   12	0	0	0	0	 0	4
	22		  275   54	0	0	0	 0   16
	23		  376  149	0	0  157	 3   72
	24		  128   72	0	0   11	15   31
	25			0	1	0	0	0	 0	0
	26			7   19	0	0   28	 0	0
	27			7	5	0	0	0	 0	3
	28			0	6	0	0	3	 0	0
	29		  407  554	0	0  254	 0   76
	30		   81  144	0	0  480	 0   20
	31		  114   97	0	0   87	 7   27
	32		  230  255	5	0  100	30   70
	33		  184  184	0	0  115	66   67
	34			0   12	0	0	4	 2	4
	35			3	1	0	0	0	 0   98
	36			0	2	0	0	0	 0	8
	37			0	0	0	0	0	 0   34
	38		   80	7	0	0	0	 0  641
	39		   79	3	0	0	0	 0  575
	40		   49	3	0	0	0	 0  407		
	'''
	df2['part'] = part
	diff = abs(df1-df2.groupby('part').sum()).fillna(df1)
	return diff.stack()

def main():
	s, a = load_data()
	soilAreaMatcher = matcher(a, s)
	# print soilAreaMatcher.eval_classifer([4, 3, 4, 3, 3, 4, 1, 1, 4, 3, 1, 3, 4, 3, 4, 3, 3, 3, 1, 1, 1, 3, 1, 3, 1, 3, 1, 1, 3, 3, 3, 3, 3, 1, 3, 3, 3, 1])
	print soilAreaMatcher.train(400, 50)

if __name__ == '__main__':
	main()

# 10880.2725,9051.625,7814.58,6692.6025,5887.04,5227.945,4844.8525,4365.6525,3997.5275,3651.7875,3413.2975,3157.9525,3048.09,2899.3875,2805.14,2749.4175,2672.08,2614.16,2588.5375,2570.8625,2544.3975,2489.8525,2457.51,2444.18,2418.895,2413.04,2397.4,2402.365,2396.6,2347.8,2332.82,2331.175,2331.275,2331.0,2331.0,2332.725,2332.75,2331.0,2331.0,2331.0
# 2331.8525
# 2331.0
# 2331.8
# 2331.0
# 2331.0
# 2331.0
# 2331.02
# 2331.0
# [[4, 3, 4, 3, 3, 4, 1, 1, 4, 3, 1, 3, 4, 3, 4, 3, 3, 3, 1, 1, 1, 3, 1, 3, 1, 3, 1, 1, 3, 3, 3, 3, 3, 1, 3, 3, 3, 1], [4, 3, 4, 3, 3, 4, 1, 1, 4, 3, 1, 3, 4, 3, 4, 3, 3, 3, 1, 1, 1, 3, 3, 3, 1, 3, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1], [4, 3, 4, 3, 3, 4, 1, 1, 4, 3, 1, 3, 4, 3, 4, 3, 3, 3, 4, 1, 1, 3, 3, 3, 1, 3, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1], [4, 3, 4, 3, 3, 4, 1, 1, 4, 3, 1, 3, 4, 3, 4, 3, 3, 3, 3, 1, 1, 3, 1, 3, 1, 3, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1], [4, 3, 4, 3, 3, 4, 1, 1, 4, 3, 1, 3, 4, 3, 4, 3, 3, 3, 1, 1, 1, 3, 1, 3, 1, 3, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1], [4, 3, 4, 3, 3, 4, 1, 1, 4, 3, 1, 3, 4, 3, 4, 3, 3, 3, 3, 1, 1, 3, 1, 3, 1, 3, 1, 1, 3, 3, 3, 3, 3, 1, 3, 3, 3, 1], [4, 3, 4, 3, 3, 4, 1, 3, 4, 3, 1, 3, 4, 3, 4, 3, 3, 3, 4, 1, 1, 3, 1, 3, 1, 3, 1, 1, 3, 3, 3, 3, 3, 1, 3, 3, 3, 1], [4, 3, 4, 3, 3, 4, 1, 1, 4, 3, 1, 3, 4, 3, 4, 3, 3, 3, 1, 1, 1, 3, 3, 3, 1, 3, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1], [4, 3, 4, 3, 3, 4, 1, 1, 4, 3, 1, 3, 4, 3, 4, 3, 3, 3, 3, 1, 1, 3, 3, 3, 1, 3, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1], [4, 3, 4, 3, 3, 4, 1, 1, 4, 3, 1, 3, 4, 3, 4, 3, 3, 3, 4, 1, 1, 3, 3, 3, 1, 3, 1, 1, 3, 3, 3, 3, 3, 3, 3, 3, 3, 1]]