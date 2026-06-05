import numpy as np
from genetic_selection import GeneticSelectionCV
from sklearn.neighbors import KNeighborsClassifier

"""
Genetic algorithm parameters:
    Population size
    Mating pool size
    Number of mutations
"""

def feature_selection(X,y):
    estimator = KNeighborsClassifier(n_neighbors=5, metric='euclidean')
    #number of samples
    L = len(y)
    selector = GeneticSelectionCV(estimator,
                                  cv=4,
                                  scoring="accuracy",
                                  max_features=6,
                                  n_population=L,
                                  crossover_proba=0.8,
                                  mutation_proba=0.2,
                                  n_generations=200,#200
                                  tournament_size=2,
                                  n_gen_no_change=10)
    selector = selector.fit(X, y)
    print (selector)
    return selector.support_
