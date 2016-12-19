class ga(object):
	"""docstring for ga"""


	def __init__(self, X, Y, clf):
		super(ga, self).__init__()
		self.X = X
		self.Y = Y
		self.clf = clf
		self.counter = 0

	def eval_classifer(self, individual):
		# self.counter += 1
		# if self.counter % 100 == 0: print self.counter / 4000

		import importHelper
		cv = importHelper.load('cvMachine')

		train = self.X
		cols_inx = [i for i in range(len(individual)) if individual[i] == 1]
		train = train[cols_inx]
		# print cols_inx
		# print train.head()

		# print len(train.columns)
		score = cv.sklearn_cross_validation(self.clf, train, self.Y).mean()
		# print score
		# print sum(individual)
		return score,

	
	def train(self, pop = 20, gen = 10):
		from deap import algorithms
		from deap import base
		from deap import creator
		from deap import tools
		import random
		import numpy as np

		from deap.tools import Statistics

		# creator.create("FitnessMulti", base.Fitness, weights=(1.0, -1.0))
		# creator.create("Individual", list, fitness=creator.FitnessMulti)

		creator.create("FitnessMax", base.Fitness, weights=(1.0,))
		creator.create("Individual", list, fitness=creator.FitnessMax)

		toolbox = base.Toolbox()
		# Attribute generator
		toolbox.register("attr_bool", random.randint, 0, 1)
		# Structure initializers
		toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attr_bool, n=len(self.X.columns))
		toolbox.register("population", tools.initRepeat, list, toolbox.individual, n=pop)

		# Operator registering
		toolbox.register("evaluate", self.eval_classifer)
		toolbox.register("mate", tools.cxUniform, indpb=0.1)
		toolbox.register("mutate", tools.mutFlipBit, indpb=0.05)
		# toolbox.register("select", tools.selNSGA2)

		MU, LAMBDA = pop, pop
		population = toolbox.population(n=MU)
		# hof = tools.ParetoFront()
		
		s = Statistics(key=lambda ind: ind.fitness.values)
		s.register("mean", np.mean)
		s.register("max", max)

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
		print top10
		return top10[0]
		# pop, logbook = algorithms.eaMuPlusLambda(pop, toolbox, mu=MU, lambda_=LAMBDA, cxpb=0.7, mutpb=0.3, ngen=gen, halloffame=hof)
		# print hof
		# return hof[-1]
