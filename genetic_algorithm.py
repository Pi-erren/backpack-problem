import time
import numpy as np
from itertools import combinations

class GeneticAlgorithm:
    """
    Genetic Algorithm

    Attributes:
        population_size (int): Number of individuals in the population.
        mutation_rate (float): Probability of mutation for each gene.
        crossover_rate (float): Probability of crossover between two parents.
        num_generations (int): Number of generations to evolve.
        budgets (list): Budget constraints for each category.
        utilities (list): Utilities (values) of the items.
        costs (list): Costs of the items across different categories.
        elitism_rate (float): Proportion of the best individuals to retain each generation.
        num_items (int): Number of items (genes) in each individual.
        population (numpy.ndarray): Current population of individuals.
        best_fitness_history (list): History of the best fitness values over generations.
        diversity_history (list): History of population diversity over generations.
    """
    def __init__(self, population_size, mutation_rate, crossover_rate, num_generations, budgets, utilities, costs, elitism_rate):
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_generations = num_generations
        self.budgets = budgets
        self.utilities = utilities
        self.costs = costs
        self.elitism_rate = elitism_rate
        self.num_items = len(utilities)
        self.population = self.initialize_population()
        self.best_fitness_history = []
        self.diversity_history = []

    def initialize_population(self):
        return np.random.randint(2, size=(self.population_size, self.num_items))

    def fitness(self, individual):
        total_utility = np.sum(individual * self.utilities)
        total_costs = np.sum(individual[:, np.newaxis] * self.costs, axis=0)
        if np.any(total_costs > self.budgets):
            return 0  # Penalty for infeasible solutions
        return total_utility

    def selection(self):
        """
        Selects individuals for the next generation using a fitness-proportionate selection.
        """
        fitnesses = np.array([self.fitness(individual) for individual in self.population])
        total_fitness = fitnesses.sum()

        if total_fitness == 0:
            probabilities = np.ones(self.population_size) / self.population_size  # Uniform distribution if all fitnesses are zero
        else:
            probabilities = fitnesses / total_fitness
        
        # Select individuals based on their fitness-proportionate probabilities.
        # Therefore, individuals with higher fitness have more chance to be selected
        selected_indices = np.random.choice(np.arange(self.population_size), size=self.population_size, p=probabilities)
        return self.population[selected_indices]

    def crossover(self, parent1, parent2):
        if np.random.rand() < self.crossover_rate:
            point = np.random.randint(1, self.num_items - 1)
            return np.concatenate((parent1[:point], parent2[point:]))
        return parent1 if np.random.rand() < 0.5 else parent2

    def mutate(self, individual):
        mutation_indices = np.random.rand(self.num_items) < self.mutation_rate
        individual[mutation_indices] = 1 - individual[mutation_indices]
        return individual

    def repair_ratio_utility_cost(self, individual):
        """
        Repairs an individual by removing items with the worst utility-to-cost ratio
        """
        total_costs = np.sum(self.costs, axis=1)
        while np.any(np.sum(individual[:, np.newaxis] * self.costs, axis=0) > self.budgets):
            # Identify the items (=1) that are included in the individual
            included_items = (individual == 1)

            # Calculating ratios
            utility_to_total_cost_ratio = self.utilities[included_items] / total_costs[included_items]

            # Find the index of the item with the worst (minimum) utility-to-total-cost ratio
            worst_item_index = np.argmin(utility_to_total_cost_ratio)

            # Get the actual index of the worst item in the original individual's array
            actual_worst_item_index = np.where(included_items)[0][worst_item_index]
            
            individual[actual_worst_item_index] = 0
        return individual
    
    def repair_algorithm_1(self, individual):
        """
        Repair method of the given subject
        Repairs an individual by ordering items by decreasing utility, 
        removing items if budgets are exceeded, and trying to add them back if possible.
        """
        # Order the objects by decreasing utility
        ordered_indices = np.argsort(-self.utilities)
        costs_sum = np.sum(individual[:, np.newaxis] * self.costs, axis=0)

        # Remove objects if budgets are exceeded
        for l in ordered_indices[::-1]:
            if individual[l] == 1 and np.any(costs_sum > self.budgets):
                individual[l] = 0
                costs_sum -= self.costs[l]

        # Try to add objects back
        for l in ordered_indices:
            if individual[l] == 0 and np.all(costs_sum + self.costs[l] <= self.budgets):
                individual[l] = 1
                costs_sum += self.costs[l]

        return individual

    def run(self, repair_method):
        """
        Runs the genetic algorithm for the specified number of generations.
        Uses the specified repair method to maintain feasibility.
        """
        num_elites = int(self.population_size * self.elitism_rate)
        self.best_fitness_history = []
        self.diversity_history = []

        for generation in range(self.num_generations):
            # Calculate current generation fitnesses
            fitnesses = np.array([self.fitness(individual) for individual in self.population])

            # Proceed to selection for crossover and selecting the elites of current generation
            elite_indices = np.argsort(fitnesses)[-num_elites:]
            elites = self.population[elite_indices]
            selected_population = self.selection()
            next_population = []

            for i in range(0, self.population_size - num_elites, 2):
                # Crossover
                parent1, parent2 = selected_population[i], selected_population[i + 1]
                offspring1, offspring2 = self.crossover(parent1, parent2), self.crossover(parent2, parent1)

                # Mutation
                offspring1, offspring2 = self.mutate(offspring1), self.mutate(offspring2)

                # Repair
                offspring1 = getattr(self, repair_method)(offspring1)
                offspring2 = getattr(self, repair_method)(offspring2)

                next_population.extend([offspring1, offspring2])

            next_population.extend(elites)
            self.population = np.array(next_population)

            # Updating best fintess and keeping track of current fitness
            best_fitness = max(fitnesses)
            self.best_fitness_history.append(best_fitness)
            self.diversity_history.append(self.calculate_diversity())

        return self.population[np.argmax([self.fitness(individual) for individual in self.population])]

    def calculate_diversity(self):
        """
        Calculates the average Hamming distance (that we use to represent diversity) 
        between all pairs of individuals in the population.
        """
        total_hamming_distance = 0
        num_comparisons = 0

        for (ind1, ind2) in combinations(self.population, 2):
            total_hamming_distance += np.sum(ind1 != ind2)
            num_comparisons += 1

        average_hamming_distance = total_hamming_distance / num_comparisons
        return average_hamming_distance

    def run_with_timing(self, repair_method):
        """
        Launch the genetic algorithm and calculate the time taken
        """
        start_time = time.time()
        self.run(repair_method)
        end_time = time.time()
        return end_time - start_time