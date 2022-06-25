"""
MACHINE LEARNING MODEL (SVM) OPTIMIZATION WITH PARTICLE SWARM OPTIMIZATION (PSO)
DATASET: HEART
FEATURE SELECTION:
HYPERPARAMETER TUNING: 

AUTHOR: GOH JIE XIN
"""

import random
import math
import copy
import numpy as np
from pandas import read_csv
from sklearn import model_selection
from sklearn.model_selection import train_test_split
from sklearn.metrics import make_scorer, accuracy_score, recall_score, f1_score, roc_auc_score, \
                            plot_confusion_matrix, plot_roc_curve
from sklearn.feature_selection import mutual_info_classif
from sklearn.svm import SVC
from deap import base, creator, tools, algorithms
from matplotlib import pyplot as plt

import warnings
warnings.filterwarnings("ignore")

class Data:

    DATASET_FILEPATH = "Assignment\Assignment 2\src\heart.csv"
    NUM_FOLDS = 5

    def __init__(self, randomSeed=42):
        
        self.randomSeed = randomSeed
        self.data = read_csv(self.DATASET_FILEPATH)

        # Normalization to speed up the model training
        # cols_to_norm = ['age','cp','trtbps','chol','restecg','thalachh','oldpeak','slp','caa','thall']
        # self.data[cols_to_norm] = self.data[cols_to_norm].apply(lambda x: (x - x.min()) / (x.max() - x.min()))

        # Split into X, y
        self.X = self.data.iloc[:, 0:-1]
        self.y = self.data.iloc[:, -1]

        # Split into train & test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y,
                                                                                stratify=self.y, 
                                                                                test_size=0.2,
                                                                                random_state=self.randomSeed)

        # Mutual information score
        self.mutInfo = np.array(self.getMutualInfo())

        # Set CV fold
        self.kfold = model_selection.KFold(n_splits=self.NUM_FOLDS)

        # Classifier - Hyperparameters
        self.hyperparam_count = 3
        self.c_range = {"min": 0.0001, "max": 50}
        self.kernel_range = {"min": -1, "max": 1}
        # self.kernel_list = {"linear": {"min": -1, "max": -0.5},
        #                     "poly": {"min": -0.5, "max": 0},
        #                     "rbf": {"min": 0, "max": 0.5},
        #                     "sigmoid": {"min": 0.5, "max": 1}}
        self.kernel_list = {"linear": {"min": -1, "max": 0},
                            "rbf": {"min": 0, "max": 1}}
        self.gamma_range = {"min": 0.0001, "max": 10}

        # Classifier - SVM
        # self.classifier = SVC(random_state=self.randomSeed)

        # Scoring
        self.wEntSenSpec = 0.495
        self.wErrRate = 0.495
        self.wFeatRatio = 1 - self.wEntSenSpec - self.wErrRate
        self.scoring = {'accuracy': make_scorer(accuracy_score),
                        'sensitivity': make_scorer(recall_score),
                        'specificity': make_scorer(recall_score, pos_label=0)}
        
    def printInfo(self):
        """
        Print a concise summary of the dataframe.
        """
        print(self.data.info(verbose=True))

    def __len__(self):
        """
        :return: the total number of features used in this classification problem
        """
        return self.X.shape[1]

    def printClassFreq(self):
        """
        Plot class freq for entire data, train & test set.
        """
        print(self.data.output.value_counts())
        self.data.output.value_counts().sort_index(ascending=True).plot(kind='bar', title="Data")
        plt.show()
        
        print(self.y_train.value_counts())
        self.y_train.value_counts().sort_index(ascending=True).plot(kind='bar', title="Train Set")
        plt.show()

        print(self.y_test.value_counts())
        self.y_test.value_counts().sort_index(ascending=True).plot(kind='bar', title="Test Set")
        plt.show()

    def checkBoundary(self, val, min, max):
        if val < min:
            return min
        elif val > max:
            return max
        else:
            return val

    def normalization(self, val, oldmin, oldmax, newmin, newmax):
        return (val-oldmin)/(oldmax-oldmin)*(newmax-newmin)+newmin

    def decodeKernel(self, val):
        for k in self.kernel_list.keys():
            if self.kernel_list[k]["min"] <= val <= self.kernel_list[k]["max"]:
                return k

    def decodeHyperparam(self, individualHyperparam):

        c = self.normalization(individualHyperparam[0], -1, 1, self.c_range["min"], self.c_range["max"])
        c = self.checkBoundary(c, self.c_range["min"], self.c_range["max"])

        kernel = self.checkBoundary(individualHyperparam[1], self.kernel_range["min"], self.kernel_range["max"])
        kernel = self.decodeKernel(kernel)

        gamma = self.normalization(individualHyperparam[2], -1, 1, self.gamma_range["min"], self.gamma_range["max"])
        gamma = self.checkBoundary(gamma, self.gamma_range["min"], self.gamma_range["max"])

        return c, kernel, gamma

    def decodeIndividual(self, individual):
        individualFeatures = [1 if x>0 else 0 for x in individual[:len(Data())]]
        individualHyperparam = individual[len(Data()):]
        return individualFeatures, individualHyperparam

    def getMutualInfo(self):
        mi = mutual_info_classif(self.X_train, self.y_train, discrete_features=True, random_state=42)
        oldmin = mi.min()
        oldmax = mi.max()
        mi = [self.normalization(i, oldmin, oldmax, -1, 1) for i in mi]
        return mi

    def trainCVEvaluation(self, zeroOneListFeat, valueHyperparam, state="undefault"):
        """
        Train model based on given feature subset & hyperparam.
        Return the fitness score.
        Fitness = a(feature subset size/total feature subset size) + 
                  b(entropy of sensitivity & specificity) +
                  c(error rate)
        """
        random.seed(self.randomSeed)
        
        # drop the dataset columns that correspond to the unselected features:
        zeroIndices = [i for i, n in enumerate(zeroOneListFeat) if n == 0]
        currentX = self.X_train.drop(self.X_train.columns[zeroIndices], axis=1)
        
        c, kernel, gamma = self.decodeHyperparam(valueHyperparam)
        if kernel not in ["rbf", "sigmoid", "poly"]:
            gamma = "scale"
        # print("Hyperparam: ", c, kernel, gamma)
        
        if state=="default":
            self.classifier = SVC(random_state=self.randomSeed)
        else:
            self.classifier = SVC(random_state=self.randomSeed, C=round(c,4), kernel=kernel, gamma=gamma)      # round(c, 4)
        scores = model_selection.cross_validate(self.classifier, currentX, self.y_train, cv=self.kfold, scoring=self.scoring)
        
        accuracy = scores['test_accuracy'].mean()
        sensitivity = scores['test_sensitivity'].mean()
        specificity = scores['test_specificity'].mean()
        if sensitivity==0:
            sensitivity+=0.01
        elif specificity==0:
            specificity+=0.01
        p = sensitivity/(sensitivity+specificity)
        q = 1-p
        # print("Scores: ", accuracy, sensitivity, specificity, p, q)

        fitness = self.wFeatRatio*(sum(zeroOneListFeat)/self.X.shape[1]) + \
                  self.wEntSenSpec*(p*math.log(p,2) + q*math.log(q,2)) + \
                  self.wErrRate*(1-accuracy)

        return fitness

    def testEvaluation(self, zeroOneListFeat, valueHyperparam, state="undefault"):
        """
        Display the model performance on test set.
        """
        # drop the dataset columns that correspond to the unselected features:
        zeroIndices = [i for i, n in enumerate(zeroOneListFeat) if n == 0]
        currentX_train = self.X_train.drop(self.X_train.columns[zeroIndices], axis=1)
        currentX_test = self.X_test.drop(self.X_test.columns[zeroIndices], axis=1)

        c, kernel, gamma = self.decodeHyperparam(valueHyperparam)
        if kernel not in ["rbf", "sigmoid", "poly"]:
            gamma = "scale"
        
        if state=="default":
            clf = SVC(random_state=self.randomSeed, probability=True).fit(currentX_train, self.y_train)
        else:
            print("Hyperparameter: ", c, kernel, gamma)
            clf = SVC(random_state=self.randomSeed, probability=True, C=round(c,4), kernel=kernel, gamma=gamma).fit(currentX_train, self.y_train)   #
        
        self.displayScore(clf, currentX_test, self.y_test)

    def displayScore(self, classifier, X_test, y_test):
        
        # specificity_score = make_scorer(recall_score, pos_label=0)
        prediction = classifier.predict(X_test)
        prediction_prob = classifier.predict_proba(X_test)[:, 1]
        
        print("Accuracy: ", accuracy_score(y_test, prediction))
        print("Sensitivity: ", recall_score(y_test, prediction))
        print("Specificity: ", recall_score(y_test, prediction, pos_label=0))
        print("F1: ", f1_score(y_test, prediction))
        print("ROC AUC: ", roc_auc_score(y_test, prediction_prob))

        cm = plot_confusion_matrix(classifier, X_test, y_test)
        cm.ax_.set_title("Confusion Matrix for Test Set")
        plt.show()

        roc = plot_roc_curve(classifier, X_test, y_test)
        roc.ax_.set_title("ROC for Test Set")
        plt.show()


class PSO:

    """
    Particle structure:
    [f1,f2,f3,f4,f5,f6,f7,f8,f9,f10,f11,f12,f13,h1,h2,h3]
    features: f1 - f13
    hyperparameters: h1 - h3
    """
    DIMENSIONS = len(Data()) + Data().hyperparam_count
    POPULATION_SIZE = 50 #20
    MAX_GENERATIONS = 100 #10
    MIN_START_POSITION, MAX_START_POSITION = -1, 1
    MIN_SPEED, MAX_SPEED = -3, 3
    MAX_LOCAL_UPDATE_FACTOR = MAX_GLOBAL_UPDATE_FACTOR = 2.0
    P_CROSSOVER = 0.9
    P_MUTATION = 0.05 

    def __init__(self, randomSeed=42):

        self.randomSeed = randomSeed
        np.random.seed(self.randomSeed)

        self.toolbox = base.Toolbox()

        # Create Fitness (Minimize) & Particle Class
        creator.create("FitnessMin", base.Fitness, weights=(-1.0,))
        creator.create("Particle", np.ndarray, fitness=creator.FitnessMin, speed=None, best=None)

        # Register Function: Create Particle, Create Population, Evaluation, Update
        self.toolbox.register("particleCreator", self.createParticle)
        self.toolbox.register("populationCreator", tools.initRepeat, list, self.toolbox.particleCreator)
        self.toolbox.register("evaluate", self.FitnessFunction)
        self.toolbox.register("update", self.updateParticle)
        self.toolbox.register("select", tools.selTournament, tournsize=2)   # 3 individuals compete & select winner
        self.toolbox.register("mate", tools.cxTwoPoint)  
        # self.toolbox.register("mutate", tools.mutFlipBit, indpb=1/self.POPULATION_SIZE)
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1/self.POPULATION_SIZE)

    def createParticle(self):
        """
        Create & Initialize new particle.
        """
        
        # ## Mutual Info-based Initialization
        # r = random.random()
        
        # if (r > 0.7):
        #     particle = creator.Particle(np.concatenate((Data().mutInfo, np.random.uniform(self.MIN_START_POSITION,self.MAX_START_POSITION,Data().hyperparam_count))))
        #     particle.speed = np.random.uniform(self.MIN_SPEED, self.MAX_SPEED, self.DIMENSIONS)
        # else:
        #     particle = creator.Particle(np.random.uniform(self.MIN_START_POSITION,self.MAX_START_POSITION,self.DIMENSIONS))
        #     particle.speed = np.random.uniform(self.MIN_SPEED, self.MAX_SPEED, self.DIMENSIONS)
        
        # return particle
        # ##########

        ## Random Initialization
        particle = creator.Particle(np.random.uniform(self.MIN_START_POSITION,self.MAX_START_POSITION,self.DIMENSIONS))
        particle.speed = np.random.uniform(self.MIN_SPEED, self.MAX_SPEED, self.DIMENSIONS)
        return particle
        #########

    def FitnessFunction(self, individual):
        """
        Return fitness value of a particle.
        """
        # individual = [1 if x>0 else 0 for x in individual]
        # individualFeatures = [1 if x>0 else 0 for x in individual[:len(Data())]]
        # individualHyperparam = individual[len(Data()):]
        individualFeatures, individualHyperparam = Data().decodeIndividual(individual)
        numFeatureUsed = sum(individualFeatures)
        if numFeatureUsed == 0:
            return 1.0,
        else:
            # return Data().trainCVEvaluation(individual),
            return Data().trainCVEvaluation(individualFeatures, individualHyperparam),
    
    def updateParticle(self, particle, best):
        """
        Update particle position based on personal best and global best.
        """
        # create rand factors:
        localUpdateFactor = np.random.uniform(0, self.MAX_LOCAL_UPDATE_FACTOR, particle.size)
        globalUpdateFactor = np.random.uniform(0, self.MAX_GLOBAL_UPDATE_FACTOR, particle.size)

        # calc local & global speed updates:
        localSpeedUpdate = localUpdateFactor*(particle.best-particle)
        globalSpeedUpdate = globalUpdateFactor*(best-particle)

        # calc updated speed:
        particle.speed = particle.speed + (localSpeedUpdate+globalSpeedUpdate)

        # enforce limit on the updated speed:
        particle.speed = np.clip(particle.speed, self.MIN_SPEED, self.MAX_SPEED)

        # replace particle position with old position + speed:
        particle[:] = particle + particle.speed


class MLOpt:

    def __init__(self, data, pso, randomSeed=42):
        self.randomSeed = randomSeed
        self.data = data
        self.pso = pso

    def run(self):
        
        random.seed(self.randomSeed)
        
        train_model_count = 0

        # create pop
        population = self.pso.toolbox.populationCreator(n=self.pso.POPULATION_SIZE)

        # prepare stats:
        stats = tools.Statistics(lambda ind: ind.fitness.values)
        stats.register("min", np.min)
        stats.register("avg", np.mean)

        logbook = tools.Logbook()
        logbook.header = ["gen", "evals"] + stats.fields

        gbest = None
        train_log = {}

        for generation in range(self.pso.MAX_GENERATIONS):

            # evaluate all particles in pop
            for particle in population:

                # find fitness:
                # Speed up, for the model that trained before no need train again, directly obtained from logbook
                list1, list2 = self.data.decodeIndividual(particle)
                c, kernel, gamma = self.data.decodeHyperparam(list2)
                if kernel not in ["rbf", "sigmoid", "poly"]:
                    gamma = "scale"
                train_particle = str([list1] + [round(c,4), kernel, gamma])

                if train_particle not in list(train_log.keys()):
                    particle.fitness.values = self.pso.toolbox.evaluate(particle)
                    train_model_count += 1
                    train_log[train_particle] = particle.fitness.values
                else:
                    particle.fitness.values = train_log[train_particle]

                # particle.fitness.values = self.pso.toolbox.evaluate(particle)
                 
                # pbest
                if particle.best is None or particle.best.size == 0 or particle.best.fitness < particle.fitness:
                    particle.best = creator.Particle(particle)
                    particle.best.fitness.values = particle.fitness.values
                
                # gbest
                if gbest is None or gbest.fitness.values[0] > particle.fitness.values[0]:     # or best.size == 0 
                    gbest = creator.Particle(particle)
                    gbest.fitness.values = particle.fitness.values
                    gbest.speed = particle.speed
                    gbest.best = particle.best
                
            # update particle's speed & position:
            for particle in population:
                # if particle.fitness != best.fitness:    # Only update not best particle (Leave 1)
                self.pso.toolbox.update(particle, gbest)
                if particle.fitness.values == gbest.fitness.values:
                    gbest.speed = particle.speed
                
            # record stats for curr gen & print:
            logbook.record(gen=generation, evals=len(population), **stats.compile(population))
            print(logbook.stream)

            ##########
            ## Apply Elitism Mechanism to select best position; Perform Crossover & Mutation on remaining position

            # Select the next generation individuals
            offspring = self.pso.toolbox.select(population, len(population) - 1)

            # Vary the pool of individuals
            offspring = algorithms.varAnd(offspring, self.pso.toolbox, self.pso.P_CROSSOVER, self.pso.P_MUTATION)

            # add the best back to population:
            offspring.append(gbest)

            # Replace the current population by the offspring
            population = copy.deepcopy(offspring)
            ##########
        
        minFitnessValue, meanFitnessValue = logbook.select('min', 'avg')
        plt.plot(minFitnessValue, color='red', label='min')
        plt.plot(meanFitnessValue, color='green', label='mean')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness Value')
        plt.legend()
        plt.show()
        
        # Before Optimization
        print("Before Optimization")
        allOnes = [1] * (len(self.data)+self.data.hyperparam_count)
        allOnesFeatures, allOnesHyperparam = Data().decodeIndividual(allOnes)
        print("-- All features selected: ", allOnesFeatures, 
              ", fitness value = ", self.data.trainCVEvaluation(allOnesFeatures, allOnesHyperparam, state="default"))   ## two hyperparam
        self.data.testEvaluation(allOnesFeatures, allOnesHyperparam, state="default")
        
        # After Optimization
        # bestParticleList = [1 if x>0 else 0 for x in best]
        bestFeatures, bestHyperparam = Data().decodeIndividual(gbest)
        print("After Optimization")
        print("-- Features selected: ", bestFeatures, 
              ", fitness value = ", self.data.trainCVEvaluation(bestFeatures, bestHyperparam))      ## two hyperparam
        # print(self.pso.toolbox.evaluate(best))
        self.data.testEvaluation(bestFeatures, bestHyperparam)

        # Train Model Count
        print("Train Model Count: ", train_model_count)
        

if __name__ == "__main__":

    data = Data()
    # print(data.X_train.join(data.y_train).head())
    # data.printInfo()
    # data.printClassFreq()
    # print(data.trainCVEvaluation([1]*13))

    pso = PSO()
    mlOpt = MLOpt(data, pso)
    mlOpt.run()

    