from genetic_algorithm import GeneticAlgorithm

import numpy as np

def run_experiments(population_size, mutation_rate, crossover_rate, num_generations, budgets, utilities, costs, elitism_rate, repair_method, num_instances=30):
    best_fitnesses = []
    diversities = []

    for _ in range(num_instances):
        ga = GeneticAlgorithm(population_size, mutation_rate, crossover_rate, num_generations, budgets, utilities, costs, elitism_rate)
        ga.run(repair_method)
        best_fitnesses.append(ga.best_fitness_history)
        diversities.append(ga.diversity_history)

    mean_fitness = np.mean(best_fitnesses, axis=0)
    std_fitness = np.std(best_fitnesses, axis=0)
    mean_diversity = np.mean(diversities, axis=0)
    std_diversity = np.std(diversities, axis=0)

    return mean_fitness, std_fitness, mean_diversity, std_diversity


def compare_population_size_computation_times(population_sizes, mutation_rate, crossover_rate, num_generations, budgets, utilities, costs, elitism_rate, num_instances=10):
    avg_times_ratio = []
    std_times_ratio = []
    avg_times_algorithm_1 = []
    std_times_algorithm_1 = []

    for size in population_sizes:
        times_ratio_repair = []
        times_algorithm_1_repair = []

        for _ in range(num_instances):
            ga = GeneticAlgorithm(size, mutation_rate, crossover_rate, num_generations, budgets, utilities, costs, elitism_rate)
            time_ratio = ga.run_with_timing('repair_ratio_utility_cost')
            times_ratio_repair.append(time_ratio)
            
            ga = GeneticAlgorithm(size, mutation_rate, crossover_rate, num_generations, budgets, utilities, costs, elitism_rate)
            time_algorithm_1 = ga.run_with_timing('repair_algorithm_1')
            times_algorithm_1_repair.append(time_algorithm_1)

        avg_time_ratio = np.mean(times_ratio_repair)
        std_time_ratio = np.std(times_ratio_repair)
        avg_time_algorithm_1 = np.mean(times_algorithm_1_repair)
        std_time_algorithm_1 = np.std(times_algorithm_1_repair)

        avg_times_ratio.append(avg_time_ratio)
        std_times_ratio.append(std_time_ratio)
        avg_times_algorithm_1.append(avg_time_algorithm_1)
        std_times_algorithm_1.append(std_time_algorithm_1)

    return avg_times_ratio, std_times_ratio, avg_times_algorithm_1, std_times_algorithm_1

