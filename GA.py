import numpy
import sklearn.svm



def reduce_features(solution, features):
    selected_elements_indices = numpy.where(solution == 1)[0]
    reduced_features = features[:, selected_elements_indices]
    return reduced_features



def classification_accuracy(labels, predictions):
    correct = numpy.where(labels == predictions)[0]
    accuracy = correct.shape[0]/labels.shape[0]
    return accuracy


from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
def cal_pop_fitness(pop, features, labels, train_indices, test_indices):
    accuracies = numpy.zeros(pop.shape[0])
    idx = 0
    acc= numpy.zeros(pop.shape[0])
    feat= numpy.zeros(pop.shape[0])
    for curr_solution in pop:
        reduced_features = reduce_features(curr_solution, features)
        train_data = reduced_features[train_indices, :]
        test_data = reduced_features[test_indices, :]
        train_labels = labels[train_indices]
        test_labels = labels[test_indices]
        SV_classifier=sklearn.svm.SVC(gamma='scale')
        SV_classifier.fit(X=train_data, y=train_labels)
        predictions = SV_classifier.predict(test_data)
        sumZeri=(len(curr_solution)-sum(curr_solution))/len(curr_solution)
        accuracies[idx] = 0.6*classification_accuracy(test_labels, predictions)+0.4*sumZeri
        acc[idx]= classification_accuracy(test_labels, predictions)
        feat[idx]=1-sumZeri

        idx = idx + 1
    print(acc)#import pdb; pdb.set_trace()
    return accuracies,acc,feat



def select_mating_pool(pop, fitness, num_parents):
    # Selecting the best individuals in the current generation as parents for producing the offspring of the next generation.
    parents = numpy.empty((num_parents, pop.shape[1]))
    for parent_num in range(num_parents):
        max_fitness_idx = numpy.where(fitness == numpy.max(fitness))
        max_fitness_idx = max_fitness_idx[0][0]
        parents[parent_num, :] = pop[max_fitness_idx, :]
        fitness[max_fitness_idx] = -99999999999
    return parents



def crossover(parents, offspring_size):
    offspring = numpy.empty(offspring_size)
    # The point at which crossover takes place between two parents. Usually, it is at the center.
    crossover_point = numpy.uint8(offspring_size[1]/2)
    for k in range(offspring_size[0]):
        # Index of the first parent to mate.
        parent1_idx = k%parents.shape[0]
        # Index of the second parent to mate.
        parent2_idx = (k+1)%parents.shape[0]
        # The new offspring will have its first half of its genes taken from the first parent.
        offspring[k, 0:crossover_point] = parents[parent1_idx, 0:crossover_point]
        # The new offspring will have its second half of its genes taken from the second parent.
        offspring[k, crossover_point:] = parents[parent2_idx, crossover_point:]
    return offspring
    
def mutation(offspring_crossover, num_mutations=2):
    for i in numpy.arange(offspring_crossover.shape[0]):
        sel1 = offspring_crossover[i,:].nonzero()
        sel1 = numpy.random.permutation(sel1[0])
        sel0 = numpy.ones(offspring_crossover[i,:].shape[0])
        sel0[sel1] = 0
        sel0 = numpy.arange(offspring_crossover[i,:].shape[0])[sel0.astype(bool)]
        sel0 = numpy.random.permutation(sel0)
        offspring_crossover[i,sel0[0:int(num_mutations/2)]] = (1-offspring_crossover[i,sel0[0:int(num_mutations/2)]])
        offspring_crossover[i,sel1[0:int(num_mutations/2)]] = (1-offspring_crossover[i,sel1[0:int(num_mutations/2)]])

    return offspring_crossover
