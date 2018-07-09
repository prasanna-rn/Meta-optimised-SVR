# all imports needed
from sklearn.model_selection import train_test_split
from sklearn.svm import SVR
from sklearn.multioutput import MultiOutputRegressor

from deap import creator, base
from deap import algorithms
from deap import gp
from deap import tools

import logging, sys
import numpy as np 
import operator
import random
import error_func as e 

# rerouting all output to transcript_svr_gp.txt
sys.stdout = open ('transcript_svr_gp.txt', 'w')
sys.stderr = open ('errors_svr_gp.txt', 'w')

# data is loaded and split randomly into training and test sets
# changing this path will change the data loaded. It is currently set to
# multivariate data with 5 inputs and outputs. 
x = np.loadtxt ('data_x.txt')
y = np.loadtxt ('data_y.txt')
x_train, x_test, y_train, y_test = train_test_split (x, y, train_size=45, random_state = 2) 


# declaring the primitive set, the operations the inidvidual lists of parameters can use between themselves
pset = gp.PrimitiveSet ('Main', arity=1)
pset.addPrimitive (np.add, 2)
pset.addPrimitive (np.subtract, 2)
pset.addPrimitive (np.multiply, 2)

def npSafeDiv (left, right):
    try:
        return np.divide (left, right)
    except ZeroDivisionError:
        return np.divide (left, 1+right)

pset.addPrimitive (npSafeDiv, 2)

# 7 functions to minimise + R-Squared to maximise 
# The order is: R^2, AIC, BIC, PRESS, MAPE, SRM, FPE, RMSE
weights = (+1,-1,-1,-1,-1,-1,-1,-1)

# defining the base classes with which toolbox.individual et all are declared 
creator.create ('FitnessMulti', base.Fitness, weights=weights)
creator.create ('individual', list, fitness=creator.FitnessMulti, pset=pset)

def create ():
    C       = np.random.sample ()
    epsilon = np.random.sample ()
    gamma   = np.random.sample ()  
    individual = [C, epsilon, gamma]
    return individual

# creating the toolbox of functions for eaSimple to use through
# the generational GP algorithm 
toolbox = base.Toolbox()
toolbox.register ('individual', tools.initIterate, creator.individual, create)
toolbox.register ('population', tools.initRepeat, list, toolbox.individual)

# function to evaluate each individual 
# it uses the three parameters C, epsilon and gamma to construct an SVR model 
# the predicted values of the SVR model are then used to compute the error 
# using the various functions 

# raw values of C, epsilon and gamma are unusable. 
# they must be transformed the same way as defined in `evaluate` 
def evaluate (individual): 
    C = 1 + 2 * abs (individual [0]) * 1.00e03
    epsilon = 0.1 + abs(individual [1]) * 0.1 + 0.02
    gamma   = abs(individual [2]) * 0.1 + 0.02
    multi_regr_rbf = MultiOutputRegressor (SVR(kernel='rbf', C=C, epsilon=epsilon, gamma=gamma))
    model = multi_regr_rbf.fit  (x_train, y_train)
    output= multi_regr_rbf.predict (x_test)
    r_squared = abs(multi_regr_rbf.score (x_test, y_test))
    if r_squared > 1:
        r_squared = 0
    params = (
        r_squared,
        e. AkaikeInformationCriterion_c (output), 
        e. BayesianInformationCriterion (output), 
        e. PRESS (output, y_test), 
        e. MAPE  (output, y_test), 
        e. StructuralRiskMinimisation   (output, y_test), 
        e. FinalPredictionError   (output, y_test), 
        e. RMSErrors (output, y_test)
        
    ) 
    print ("The parameters are: ", params )
    return params 

# the remaining toolbox operations: 
# genetic operators all 
toolbox.register ('evaluate', evaluate)
toolbox.register("mate",tools.cxUniform, indpb=0.25 )
toolbox.register("mutate", tools.mutGaussian, mu=2, sigma=1.34, indpb=0.1)
toolbox.register("variate", algorithms.varOr)
toolbox.register("select", tools.selTournament, tournsize=3)

def main(): 
    randnum = np.random.choice (np.arange (0, 500))
    random.seed(randnum)
    
    logging.basicConfig(level=logging.DEBUG, stream=sys.stdout)

    pop = toolbox.population(n=50)
    hof = tools.HallOfFame(5)
    
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("Avg", np.mean)
    stats.register("Std", np.std)
    stats.register("Min", np.min)
    stats.register("Max", np.max)
   
    pop, logbook = algorithms.eaSimple (pop, toolbox, cxpb=0.25, mutpb=0.15, ngen=50, stats=stats)
    #logging.info("Best individual is %s", hof[0])
    for ind in pop: 
        print (ind)
        print ("Corresponding fitnesses: ", evaluate(ind))
        print ('------------------------------')
    print (logbook[0])    
    return pop, hof

if __name__ == "__main__":
    main()
