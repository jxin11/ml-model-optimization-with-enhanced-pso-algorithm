import copy
import random
import xgboost as xgb
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import make_scorer, accuracy_score, recall_score, f1_score, roc_auc_score, roc_curve, \
                            plot_confusion_matrix, plot_roc_curve
from deap import base, creator, tools, algorithms

import warnings
warnings.filterwarnings("ignore")

class Data:

    def __init__(self, X_train, y_train, X_test, y_test, randomSeed=42):
        
        self.randomSeed = randomSeed
      
        self.X_train = X_train
        self.y_train = y_train
        self.X_test = X_test
        self.y_test = y_test

        # Hyperparam
        self.n_estimators_range = {"min": 50, "max": 350}
        self.gamma_range = {"min": 0, "max": 1}
        self.max_bin_range = {"min": 10, "max": 100}
        self.lr_list = [0.1, 0.01, 0.001]
        self.booster_list = ["gbtree", "gblinear", "dart"]
        
    def get_balanced_accuracy_score(self, model, x, y):
        """
        Return balanced accuracy.
        """
        return balanced_accuracy_score(y, model.predict(x))

    def get_weighted_f1_score(self, model, x, y):
        return f1_score(y, model.predict(x), average='weighted')

    def get_weighted_roc_auc_score(self, model, x, y):
        try:
            return roc_auc_score(y, model.predict_proba(x), multi_class='ovr')
        except ValueError:
            return roc_auc_score(y, np.array([[0, 0, 0, 0, 1] if np.isnan(a).all() else list(a) for a in model.predict_proba(x)]), multi_class='ovr')

    def getFitness(self, classifier, x, y):
        """
        Fitness = Balanced Accuracy --> Maximize
        """
        return self.get_balanced_accuracy_score(classifier, x, y)
    
    def checkBoundary(self, val, min, max):
        if val < min:
            return min
        elif val > max:
            return max
        else:
            return val

    def normalization(self, val, oldmin, oldmax, newmin, newmax):
        return (val-oldmin)/(oldmax-oldmin)*(newmax-newmin)+newmin

    def decodeIndividual(self, individual):
        """
        Input: individual [a,b,c,d,e]  --> [-1, 1]
        Output: Dictionary
        """
        individual = [self.checkBoundary(i, -1, 1) for i in individual]
        hyperparam = {}
        hyperparam["n_estimators"] = int(round(self.normalization(individual[0], -1, 1, self.n_estimators_range["min"], self.n_estimators_range["max"])))
        hyperparam["gamma"] = round(self.normalization(individual[1], -1, 1, self.gamma_range["min"], self.gamma_range["max"]),1)
        hyperparam["max_bin"] = int(round(self.normalization(individual[2], -1, 1, self.max_bin_range["min"], self.max_bin_range["max"])))
        hyperparam["lr"] = self.lr_list[int(round(self.normalization(individual[3], -1, 1, 0, len(self.lr_list)-1)))]
        hyperparam["booster"] = self.booster_list[int(round(self.normalization(individual[4], -1, 1, 0, len(self.booster_list)-1)))]

        return hyperparam

    def trainEvaluation(self, individual, state="undefault"):
        """
        Train model based on given feature subset.
        Return the fitness score.
        """
        random.seed(self.randomSeed)

        if state=="default":
            self.classifier = OneVsRestClassifier(xgb.XGBClassifier(objective="binary:logistic", random_state=11)).fit(self.X_train, self.y_train)
            fitness = self.getFitness(self.classifier, self.X_test, self.y_test)
        else:
            # decode hyperparam
            hyperparam = self.decodeIndividual(individual)
            print(hyperparam)
            if hyperparam["booster"] == "gblinear":
                self.classifier = OneVsRestClassifier(xgb.XGBClassifier(objective="binary:logistic", random_state=11,
                                                                        n_estimators = hyperparam["n_estimators"],
                                                                        learning_rate = hyperparam["lr"],
                                                                        booster = hyperparam["booster"])).fit(self.X_train, self.y_train)
            else:
                self.classifier = OneVsRestClassifier(xgb.XGBClassifier(objective="binary:logistic", random_state=11,
                                                                        n_estimators = hyperparam["n_estimators"],
                                                                        gamma = hyperparam["gamma"],
                                                                        max_bin = hyperparam["max_bin"],
                                                                        learning_rate = hyperparam["lr"],
                                                                        booster = hyperparam["booster"])).fit(self.X_train, self.y_train)
            fitness = self.getFitness(self.classifier, self.X_test, self.y_test)
        return fitness

    def testEvaluation(self, individual, state="undefault"):
        """
        Display the model performance on test set.
        """
        if state=="default":
            self.classifier = OneVsRestClassifier(xgb.XGBClassifier(objective="binary:logistic", random_state=11)).fit(self.X_train, self.y_train)
            fitness = self.getFitness(self.classifier, self.X_test, self.y_test)
            self.displayScore(self.classifier, self.X_test, self.y_test)
        else:
            # decode hyperparam
            hyperparam = self.decodeIndividual(individual)
            if hyperparam["booster"] == "gblinear":
                self.classifier = OneVsRestClassifier(xgb.XGBClassifier(objective="binary:logistic", random_state=11,
                                                                        n_estimators = hyperparam["n_estimators"],
                                                                        learning_rate = hyperparam["lr"],
                                                                        booster = hyperparam["booster"])).fit(self.X_train, self.y_train)
            else:
                self.classifier = OneVsRestClassifier(xgb.XGBClassifier(objective="binary:logistic", random_state=11,
                                                                        n_estimators = hyperparam["n_estimators"],
                                                                        gamma = hyperparam["gamma"],
                                                                        max_bin = hyperparam["max_bin"],
                                                                        learning_rate = hyperparam["lr"],
                                                                        booster = hyperparam["booster"])).fit(self.X_train, self.y_train)
            fitness = self.getFitness(self.classifier, self.X_test, self.y_test)
            self.displayScore(self.classifier, self.X_test, self.y_test)

    def displayScore(self, classifier, X_test, y_test):
        
        print("Balanced Accuracy: ", self.get_balanced_accuracy_score(classifier, X_test, y_test))
        print("Weighted F1: ", self.get_weighted_f1_score(classifier, X_test, y_test))
        print("Weighted AUC ROC Score: ", self.get_weighted_roc_auc_score(classifier, X_test, y_test))
        
        plot_confusion_matrix(classifier, X_test, y_test, cmap='Blues', normalize='true')
        plt.show()


class PSO:

    """
    Particle structure: list with length 5
    """
    POPULATION_SIZE = 20
    MAX_GENERATIONS = 10
    MIN_START_POSITION, MAX_START_POSITION = -1, 1
    MIN_SPEED, MAX_SPEED = -3, 3
    MAX_LOCAL_UPDATE_FACTOR = MAX_GLOBAL_UPDATE_FACTOR = 2.0
    P_CROSSOVER = 0.9
    P_MUTATION = 0.05 

    def __init__(self, data, randomSeed=42):

        self.randomSeed = randomSeed
        np.random.seed(self.randomSeed)

        self.data = data
        self.DIMENSIONS = 5  # 5 hyperparams to tune

        self.toolbox = base.Toolbox()

        # Create Fitness (Maximize) & Particle Class
        creator.create("FitnessMax", base.Fitness, weights=(1.0,))
        creator.create("Particle", np.ndarray, fitness=creator.FitnessMax, speed=None, best=None)

        # Register Function: Create Particle, Create Population, Evaluation, Update
        self.toolbox.register("particleCreator", self.createParticle)
        self.toolbox.register("populationCreator", tools.initRepeat, list, self.toolbox.particleCreator)
        self.toolbox.register("evaluate", self.FitnessFunction)
        self.toolbox.register("update", self.updateParticle)
        self.toolbox.register("select", tools.selTournament, tournsize=2)   # 3 individuals compete & select winner
        self.toolbox.register("mate", tools.cxTwoPoint)  
        self.toolbox.register("mutate", tools.mutShuffleIndexes, indpb=1/self.POPULATION_SIZE)

    def createParticle(self):
        """
        Create & Initialize new particle.
        """
        ## Random Initialization
        particle = creator.Particle(np.random.uniform(self.MIN_START_POSITION,self.MAX_START_POSITION,self.DIMENSIONS))
        particle.speed = np.random.uniform(self.MIN_SPEED, self.MAX_SPEED, self.DIMENSIONS)
        return particle

    def FitnessFunction(self, individual):
        """
        Return fitness value of a particle.
        """
        # individual = [1 if x>0 else 0 for x in individual]
        # # individualFeatures = [1 if x>0 else 0 for x in individual[:len(Data())]]
        # # individualHyperparam = individual[len(Data()):]
        # # individualFeatures, individualHyperparam = self.data.decodeIndividual(individual)
        # numFeatureUsed = sum(individual)
        # if numFeatureUsed == 0:
        #     return 0,
        # else:
        return self.data.trainEvaluation(individual),
    
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
        stats.register("max", np.max)
        stats.register("avg", np.mean)

        logbook = tools.Logbook()
        logbook.header = ["gen", "evals"] + stats.fields

        gbest = None
        train_log = {}

        for generation in range(self.pso.MAX_GENERATIONS):

            # evaluate all particles in pop
            for particle in population:

                # Speed up, for the model that trained before no need train again, directly obtained from logbook
                # list1, list2 = self.data.decodeIndividual(particle)
                # n_estimators, min_samples_split, min_samples_leaf = self.data.decodeHyperparam(list2)
                # train_particle = str([list1] + [n_estimators, min_samples_split, min_samples_leaf])
                train_particle = str(self.data.decodeIndividual(particle))

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
                if gbest is None or gbest.fitness.values[0] < particle.fitness.values[0]:     # or best.size == 0 
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
        
        maxFitnessValue, meanFitnessValue = logbook.select('max', 'avg')
        plt.plot(maxFitnessValue, color='red', label='max')
        plt.plot(meanFitnessValue, color='green', label='mean')
        plt.xlabel('Iteration')
        plt.ylabel('Fitness Value')
        plt.legend()
        plt.show()
        
        # Before Optimization
        print("Before Optimization")
        print("-- Hyperparam: Default", 
              ", fitness value = ", self.data.trainEvaluation(None, state="default"))   ## two hyperparam
        self.data.testEvaluation(None, state="default")
        
        # After Optimization
        # bestParticleList = [1 if x>0 else 0 for x in best]
        # bestFeatures, bestHyperparam = self.data.decodeIndividual(gbest)
        print("After Optimization")

        print("-- Hyperparam: ", self.data.decodeIndividual(gbest), 
              ", fitness value = ", self.data.trainEvaluation(gbest))      ## two hyperparam
        # print(self.pso.toolbox.evaluate(best))
        self.data.testEvaluation(gbest)

        # Train Model Count
        print("Train Model Count: ", train_model_count)

        return gbest
