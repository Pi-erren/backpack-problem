import numpy as np
import random

class GeneticAlgorithm:
    def __init__(self, population_size, mutation_rate, crossover_rate, num_generations, budgets, utilities, costs):
        """
        Initialize the genetic algorithm with the given parameters.
        
        Parameters:
        - population_size: Number of individuals in the population.
        - mutation_rate: Probability of mutating a gene.
        - crossover_rate: Probability of performing crossover.
        - num_generations: Number of generations to run the algorithm.
        - budgets: Array of budget constraints for each cost dimension.
        - utilities: Array of utilities for each item.
        - costs: 2D array where each row represents the costs of an item across different dimensions.
        """
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.num_generations = num_generations
        self.budgets = budgets
        self.utilities = utilities
        self.costs = costs
        self.num_items = len(utilities)
        self.population = self.initialize_population()

    def initialize_population(self):
        """
        Create an initial population of random binary solutions.
        
        Returns:
        A 2D array where each row is a binary vector representing a solution.
        """
        return np.random.randint(2, size=(self.population_size, self.num_items))

    def fitness(self, solution):
        """
        Calculate the fitness of a solution. Fitness is the total utility if the solution respects all budget constraints.
        
        Parameters:
        - solution: Binary vector representing a solution.
        
        Returns:
        Total utility if the solution is possible, otherwise 0.
        """
        total_utility = np.sum(solution * self.utilities)
        for i in range(len(self.budgets)):
            if np.sum(solution * self.costs[:, i]) > self.budgets[i]:
                return 0
        return total_utility

    def selection(self):
        """
        Select individuals from the current population based on their fitness.
        
        Returns:
        A new population array with selected individuals.
        """
        # Calculating fitnesses
        fitnesses = np.array([self.fitness(individual) for individual in self.population])

        # Compute selection probabilities for each individual based on their fitness 
        # in order to represent the weight of each individual fitness
        probabilities = fitnesses / fitnesses.sum()

        # Select individuals based on their fitness weights promoting higher ones
        selected_indices = np.random.choice(np.arange(self.population_size), size=self.population_size, p=probabilities)
        return self.population[selected_indices]

    def crossover(self, parent1, parent2):
        """
        Perform a crossover operation between two parents to produce an offspring.
        
        Parameters:
        - parent1: Binary vector representing the first parent.
        - parent2: Binary vector representing the second parent.
        
        Returns:
        A binary vector representing the offspring.
        """
        if np.random.rand() < self.crossover_rate:
            # Select a random crossover point
            # From 1 to num_items - 1 to ensure to take at least one element
            # from both parents
            point = np.random.randint(1, self.num_items - 1)

            return np.concatenate((parent1[:point], parent2[point:]))
        return random.choice([parent1, parent2])

    def mutate(self, individual):
        """
        Mutate an individual by flipping bits with a probability equal to the mutation rate.
        
        Parameters:
        - individual: Binary vector representing the individual to mutate.
        
        Returns:
        The mutated binary vector.
        """
        for i in range(self.num_items):
            if np.random.rand() < self.mutation_rate:
                individual[i] = 1 - individual[i]
        return individual

    def highest_utility_repair(self, individual):
        """
        Repair a solution to ensure it respects the budget constraints.
        
        Parameters:
        - individual: Binary vector representing the solution to repair.
        
        Returns:
        A binary vector representing the repaired solution.
        """
        for i in range(len(self.budgets)):
            while np.sum(individual * self.costs[:, i]) > self.budgets[i]:
                # Find the item with the highest utility that is included in the solution
                worst_item_indice = np.argmax(self.utilities * individual)

                # Remove that item from the solution
                individual[worst_item_indice] = 0
        return individual
    
    def ratio_utility_cost_repair(self, individual):
        """
        Repair a solution to ensure it respects the budget constraints.
        
        This version repairs the non possible solutions by evaluating the utility/cost ratios
        Then, it get rid of the les
        Parameters:
        - individual: Binary vector representing the solution to repair.
        
        Returns:
        A binary vector representing the repaired solution.
        """
        for i in range(len(self.budgets)):
            # While the total cost in dimension 'i' exceeds the budget
            while np.sum(individual * self.costs[:, i]) > self.budgets[i]:
                # Calculate utility-to-cost ratio for included items
                included_items = individual == 1
                utility_to_cost_ratio = self.utilities[included_items] / self.costs[included_items, i]
                
                # Find the index of the item with the lowest utility-to-cost ratio
                worst_item_index = np.argmin(utility_to_cost_ratio)
                
                # Map the index to the original item index
                actual_worst_item_index = np.where(included_items)[0][worst_item_index]
                
                # Remove that item from the solution
                individual[actual_worst_item_index] = 0
                
        return individual


    def run(self):
        """
        Run the genetic algorithm for the specified number of generations.
        
        Returns:
        The best solution found.
        """
        for generation in range(self.num_generations):
            # Select individuals based on fitness
            selected_population = self.selection()
            next_population = []
            for i in range(0, self.population_size, 2):
                # Choose pairs of parents
                parent1, parent2 = selected_population[i], selected_population[i + 1]
                # Perform crossover to create offspring
                offspring1, offspring2 = self.crossover(parent1, parent2), self.crossover(parent2, parent1)
                # Mutate offspring
                offspring1, offspring2 = self.mutate(offspring1), self.mutate(offspring2)
                # Repair offspring to ensure they are valid solutions
                offspring1, offspring2 = self.repair(offspring1), self.repair(offspring2)
                next_population.extend([offspring1, offspring2])
            # Replace the old population with the new one
            self.population = np.array(next_population)
            # Calculate the best fitness in the current generation
            best_fitness = max([self.fitness(individual) for individual in self.population])
            print(f'Generation {generation}: Best Fitness = {best_fitness}')
        # Find and return the best solution in the final population
        best_solution = self.population[
                np.argmax([self.fitness(individual) for individual in self.population])
            ]
        return best_solution