import numpy
import GA
import matplotlib.pyplot
import numpy as np
from pathlib import Path


def GA_function(data_inputs,data_outputs,r_state,classes,percentage):
    path = Path('./data/GA_{}_{}.npy'.format(r_state,classes))
    if path.is_file():
        print('Open file already calculated')
        best_solution_indices = np.load(path)
    else:

        num_samples = data_inputs.shape[0]
        num_feature_elements = data_inputs.shape[1]

        train_indices = numpy.arange(1, num_samples, 2)
        test_indices = numpy.arange(0, num_samples, 2)
        print("Number of training samples: ", train_indices.shape[0])
        print("Number of test samples: ", test_indices.shape[0])

        sol_per_pop = 200# Population size.
        num_parents_mating = int(sol_per_pop/10)# Number of parents inside the mating pool.
        percentage = percentage

        num_mutations = int(num_feature_elements*percentage*0.4) # Number of elements to mutate.
        pop_shape = (sol_per_pop, num_feature_elements)

        load=False
        path=('./data'+'/GA_lastPopulation.npy')

        # Creating the initial population.
        if load==True:
            if not Path(path).is_file():
                print("FILE ERROR")
                exit(0)
            with open(path):
                new_population=np.load(path)
        else:
            new_population1 = numpy.ones((sol_per_pop, int(num_feature_elements*percentage)))
            new_population0 = numpy.zeros((sol_per_pop, num_feature_elements- int(num_feature_elements*percentage)))
            new_population = numpy.concatenate((new_population1,new_population0), axis=1)
            for i in numpy.arange(sol_per_pop): new_population[i,:]= new_population[i,numpy.random.permutation(num_feature_elements)]
        print(new_population.shape)

        best_outputs = []
        best_acc=[]
        best_feat=[]
        num_generations = 5
        for generation in range(num_generations):
            print("Generation : ", generation)
            # Measuring the fitness of each chromosome in the population.
            
            fitness,accuracy,feat = GA.cal_pop_fitness(new_population, data_inputs, data_outputs, train_indices, test_indices)
            best_outputs.append(numpy.max(fitness))
            best_acc.append(numpy.max(accuracy))
            best_feat.append(numpy.max(feat))
            # The best result in the current iteration.
            print("Best result : ", best_outputs[-1])
            # Selecting the best parents in the population for mating.
            parents = GA.select_mating_pool(new_population, fitness, num_parents_mating)
            # Generating next generation using crossover.
            offspring_crossover = GA.crossover(parents, offspring_size=(pop_shape[0]-parents.shape[0], num_feature_elements))
            # Adding some variations to the offspring using mutation.
            offspring_mutation = GA.mutation(offspring_crossover, num_mutations=num_mutations)
            # Creating the new population based on the parents and offspring.
            new_population[0:parents.shape[0], :] = parents
            new_population[parents.shape[0]:, :] = offspring_mutation

        np.save(path,new_population,allow_pickle=True)
        # Getting the best solution after iterating finishing all generations.
        # At first, the fitness is calculated for each solution in the final generation.
        fitness = GA.cal_pop_fitness(new_population, data_inputs, data_outputs, train_indices, test_indices)
        # Then return the index of that solution corresponding to the best fitness.
        best_match_idx = numpy.where(fitness == numpy.max(fitness))[0]
        best_match_idx = best_match_idx[0]

        best_solution = new_population[best_match_idx, :]
        best_solution_indices = numpy.where(best_solution == 1)[0]
        best_solution_num_elements = best_solution_indices.shape[0]
        best_solution_fitness = fitness[best_match_idx]

        print("best_match_idx : ", best_match_idx)
        print("best_solution : ", best_solution)
        print("Selected indices : ", best_solution_indices)
        print("Number of selected elements : ", best_solution_num_elements)
        print("Best solution fitness : ", best_solution_fitness)

        matplotlib.pyplot.plot(best_outputs)
        matplotlib.pyplot.xlabel("Iteration")
        matplotlib.pyplot.ylabel("Fitness")
        matplotlib.pyplot.savefig('GA_best_fitt')
        matplotlib.pyplot.close()

        matplotlib.pyplot.plot(best_acc)
        matplotlib.pyplot.xlabel("Iteration")
        matplotlib.pyplot.ylabel("Accuracy")
        matplotlib.pyplot.savefig('GA_best_acc')
        matplotlib.pyplot.close()

        matplotlib.pyplot.plot(numpy.arange(len(best_feat)),best_feat)
        matplotlib.pyplot.xlabel("Iteration")
        matplotlib.pyplot.ylabel("Nfeatures")
        matplotlib.pyplot.savefig('GA_best_features')
        matplotlib.pyplot.close()

    return best_solution_indices
