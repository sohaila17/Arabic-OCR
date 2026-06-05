# sklearn-genetic - Genetic feature selection module for scikit-learn
# Copyright (C) 2016-2019  Manuel Calzolari
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU Lesser General Public License as published by
# the Free Software Foundation, version 3 of the License.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

"""Genetic algorithm for feature selection"""

import multiprocessing
import random
import numbers
import numpy as np
from sklearn.utils import check_X_y
from sklearn.utils.metaestimators import if_delegate_has_method
from sklearn.base import BaseEstimator
from sklearn.base import MetaEstimatorMixin
from sklearn.base import clone
from sklearn.base import is_classifier
from sklearn.model_selection import check_cv
from sklearn.model_selection._validation import _fit_and_score
from sklearn.metrics.scorer import check_scoring
from sklearn.feature_selection.base import SelectorMixin
from sklearn.externals.joblib import cpu_count
from deap import algorithms
from deap import base
from deap import creator
from deap import tools


creator.create("Fitness", base.Fitness, weights=(1.0, -1.0))
creator.create("Individual", list, fitness=creator.Fitness)


def _eaFunction(population, toolbox, cxpb, mutpb, ngen, ngen_no_change=None, stats=None,
                halloffame=None, verbose=0):
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats else {}
    logbook.record(gen=0, nevals=len(invalid_ind), **record)
    if verbose:
        print(logbook.stream)

    # Begin the generational process
    wait = 0
    for gen in range(1, ngen + 1):
        # Select the next generation individuals
        offspring = toolbox.select(population, len(population))

        # Vary the pool of individuals
        offspring = algorithms.varAnd(offspring, toolbox, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Get the previous best individual before updating the hall of fame
        prev_best = halloffame[0]

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Replace the current population by the offspring
        population[:] = offspring

        # Append the current generation statistics to the logbook
        record = stats.compile(population) if stats else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print(logbook.stream)

        # If the new best individual is the same as the previous best individual,
        # increment a counter, otherwise reset the counter
        if halloffame[0] == prev_best:
            wait += 1
        else:
            wait = 0

        # If the counter reached the termination criteria, stop the optimization
        if ngen_no_change is not None and wait >= ngen_no_change:
            break

    return population, logbook


def _evalFunction(individual, gaobject, estimator, X, y, cv, scorer, verbose, fit_params,
                  max_features, caching):
    #get sum for columns
    individual_sum = np.sum(individual, axis=0)
    if individual_sum == 0 or individual_sum > max_features:
        return -10000, individual_sum
    individual_tuple = tuple(individual)
    if caching and individual_tuple in gaobject.scores_cache:
        return gaobject.scores_cache[individual_tuple], individual_sum
    X_selected = X[:, np.array(individual, dtype=np.bool)]
    scores = []
    for train, test in cv.split(X, y):
        score = _fit_and_score(estimator=estimator, X=X_selected, y=y, scorer=scorer,
                               train=train, test=test, verbose=verbose, parameters=None,
                               fit_params=fit_params)
        scores.append(score)
    #get the mean for the scores
    scores_mean = np.mean(scores)
    if caching:
        gaobject.scores_cache[individual_tuple] = scores_mean
    return scores_mean, individual_sum


class GeneticSelectionCV(BaseEstimator, MetaEstimatorMixin, SelectorMixin):
    """Feature selection with genetic algorithm.

    Parameters
    ----------
    estimator : object
        A supervised learning estimator with a `fit` method.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:

        - None, to use the default 3-fold cross-validation,
        - integer, to specify the number of folds.
        - An object to be used as a cross-validation generator.
        - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validation strategies that can be used here.

    scoring : string, callable or None, optional, default: None
        A string (see model evaluation documentation) or
        a scorer callable object / function with signature
        ``scorer(estimator, X, y)``.

    fit_params : dict, optional
        Parameters to pass to the fit method.

    max_features : int or None, optional
        The maximum number of features selected.

    verbose : int, default=0
        Controls verbosity of output.

    n_jobs : int, default 1
        Number of cores to run in parallel.
        Defaults to 1 core. If `n_jobs=-1`, then number of jobs is set
        to number of cores.

    n_population : int, default=300
        Number of population for the genetic algorithm.

    crossover_proba : float, default=0.5
        Probability of crossover for the genetic algorithm.

    mutation_proba : float, default=0.2
        Probability of mutation for the genetic algorithm.

    n_generations : int, default=40
        Number of generations for the genetic algorithm.

    crossover_independent_proba : float, default=0.1
        Independent probability for each attribute to be exchanged, for the genetic algorithm.

    mutation_independent_proba : float, default=0.05
        Independent probability for each attribute to be mutated, for the genetic algorithm.

    tournament_size : int, default=3
        Tournament size for the genetic algorithm.

    n_gen_no_change : int, default None
        If set to a number, it will terminate optimization when best individual is not
        changing in all of the previous ``n_gen_no_change`` number of generations.

    caching : boolean, default=False
        If True, scores of the genetic algorithm are cached.

    Attributes
    ----------
    n_features_ : int
        The number of selected features with cross-validation.

    support_ : array of shape [n_features]
        The mask of selected features.

    generation_scores_ : array of shape [n_generations]
        The maximum cross-validation score for each generation.

    estimator_ : object
        The external estimator fit on the reduced dataset.

    Examples
    --------
    An example showing genetic feature selection.

    >>> import numpy as np
    >>> from sklearn import datasets, linear_model
    >>> from genetic_selection import GeneticSelectionCV
    >>> iris = datasets.load_iris()
    >>> E = np.random.uniform(0, 0.1, size=(len(iris.data), 20))
    >>> X = np.hstack((iris.data, E))
    >>> y = iris.target
    >>> estimator = linear_model.LogisticRegression(solver="liblinear", multi_class="ovr")
    >>> selector = GeneticSelectionCV(estimator, cv=5)
    >>> selector = selector.fit(X, y)
    >>> selector.support_ # doctest: +NORMALIZE_WHITESPACE
    array([ True  True  True  True False False False False False False False False
           False False False False False False False False False False False False], dtype=bool)
    """
    def __init__(estimator, cv=None, scoring=None, fit_params=None, max_features=None,
                 verbose=0, n_jobs=1, n_population=300, crossover_proba=0.5, mutation_proba=0.2,
                 n_generations=40, crossover_independent_proba=0.1,
                 mutation_independent_proba=0.05, tournament_size=3, n_gen_no_change=None,
                 caching=False):
         estimator = estimator
         cv = cv
         scoring = scoring
         fit_params = fit_params
         max_features = max_features
         verbose = verbose
         n_jobs = n_jobs
         n_population = n_population
         crossover_proba = crossover_proba
         mutation_proba = mutation_proba
         n_generations = n_generations
         crossover_independent_proba = crossover_independent_proba
         mutation_independent_proba = mutation_independent_proba
         tournament_size = tournament_size
         n_gen_no_change = n_gen_no_change
         caching = caching
         scores_cache = {}

    @property
    def _estimator_type(self):
        return  estimator._estimator_type

    def fit(self, X, y):
        """Fit the GeneticSelectionCV model and then the underlying estimator on the selected
           features.

        Parameters
        ----------
        X : {array-like, sparse matrix}, shape = [n_samples, n_features]
            The training input samples.

        y : array-like, shape = [n_samples]
            The target values.
        """
        return  _fit(X, y)

    def _fit(self, X, y):
        X, y = check_X_y(X, y, "csr")
        # Initialization
        cv = check_cv( cv, y, is_classifier( estimator))
        scorer = check_scoring( estimator, scoring= scoring)
        n_features = X.shape[1]

        if  max_features is not None:
            if not isinstance( max_features, numbers.Integral):
                raise TypeError("'max_features' should be an integer between 1 and {} features."
                                " Got {!r} instead."
                                .format(n_features,  max_features))
            elif  max_features < 1 or  max_features > n_features:
                raise ValueError("'max_features' should be between 1 and {} features."
                                 " Got {} instead."
                                 .format(n_features,  max_features))
            max_features =  max_features
        else:
            max_features = n_features

        if not isinstance( n_gen_no_change, (numbers.Integral, np.integer, type(None))):
            raise ValueError("'n_gen_no_change' should either be None or an integer."
                             " {} was passed."
                             .format( n_gen_no_change))

        estimator = clone( estimator)

        # Genetic Algorithm
        toolbox = base.Toolbox()

        toolbox.register("attr_bool", random.randint, 0, 1)
        toolbox.register("individual", tools.initRepeat,
                         creator.Individual, toolbox.attr_bool, n=n_features)
        toolbox.register("population", tools.initRepeat, list, toolbox.individual)
        toolbox.register("evaluate", _evalFunction, gaobject=self, estimator=estimator, X=X, y=y,
                         cv=cv, scorer=scorer, verbose= verbose, fit_params= fit_params,
                         max_features=max_features, caching= caching)
        toolbox.register("mate", tools.cxUniform, indpb= crossover_independent_proba)
        toolbox.register("mutate", tools.mutFlipBit, indpb= mutation_independent_proba)
        toolbox.register("select", tools.selTournament, tournsize= tournament_size)

        if  n_jobs == 0:
            raise ValueError("n_jobs == 0 has no meaning.")
        elif  n_jobs > 1:
            pool = multiprocessing.Pool(processes= n_jobs)
            toolbox.register("map", pool.map)
        elif  n_jobs < 0:
            pool = multiprocessing.Pool(processes=max(cpu_count() + 1 +  n_jobs, 1))
            toolbox.register("map", pool.map)

        pop = toolbox.population(n= n_population)
        #get the best indivisual
        hof = tools.HallOfFame(1, similar=np.array_equal)
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("avg", np.mean, axis=0)
        stats.register("std", np.std, axis=0)
        stats.register("min", np.min, axis=0)
        stats.register("max", np.max, axis=0)

        if  verbose > 0:
            print("Selecting features with genetic algorithm.")

        _, log = _eaFunction(pop, toolbox, cxpb= crossover_proba, mutpb= mutation_proba,
                             ngen= n_generations, ngen_no_change= n_gen_no_change,
                             stats=stats, halloffame=hof, verbose= verbose)
        if  n_jobs != 1:
            pool.close()
            pool.join()

        # Set final attributes
        support_ = np.array(hof, dtype=np.bool)[0]
         estimator_ = clone( estimator)
         estimator_.fit(X[:, support_], y)

         generation_scores_ = np.array([score for score, _ in log.select("max")])
         n_features_ = support_.sum()
         support_ = support_

        return self

    @if_delegate_has_method(delegate='estimator')
    def predict(self, X):
        """Reduce X to the selected features and then predict using the
           underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        y : array of shape [n_samples]
            The predicted target values.
        """
        return  estimator_.predict( transform(X))

    @if_delegate_has_method(delegate='estimator')
    def score(self, X, y):
        """Reduce X to the selected features and then return the score of the
           underlying estimator.

        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            The input samples.

        y : array of shape [n_samples]
            The target values.
        """
        return  estimator_.score( transform(X), y)

    def _get_support_mask(self):
        return  support_

    @if_delegate_has_method(delegate='estimator')
    def decision_function(self, X):
        return  estimator_.decision_function( transform(X))

    @if_delegate_has_method(delegate='estimator')
    def predict_proba(self, X):
        return  estimator_.predict_proba( transform(X))

    @if_delegate_has_method(delegate='estimator')
    def predict_log_proba(self, X):
        return  estimator_.predict_log_proba( transform(X))
