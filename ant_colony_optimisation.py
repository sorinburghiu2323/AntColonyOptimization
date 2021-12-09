from typing import List
import numpy as np
import random


class Ant:
    def __init__(self) -> None:
        self.fitness = 0
        self.path = []

    def generate_ant_path(self, graph: List[List[float]], is_bpp1: bool = True) -> None:
        """
        Generate random ant path biased based on path's pheromone level.
        Then calculate its' fitness.
        :param graph: 2D array representing current state of the graph.
        """
        bins_total = [0] * len(graph[0])
        positions = list(range(0, len(graph[0])))

        # Pack each item in a bin based on weight.
        # Then add the item weight to the bins_total.
        # In bpp1: item_weight = index. In bpp2: item_weight = index^2.
        for index, item in enumerate(graph):
            position = random.choices(positions, weights=item)[0]
            self.path.append(position)
            index = index + 1  # Weight starts at 1.
            bins_total[position] += index if is_bpp1 else index ** 2

        self.calculate_fitness(bins_total)

    def calculate_fitness(self, bins_total: List[float]) -> None:
        """
        Calculate the ant path's fitness then update it.
        :param bins_total: list with weight for each bin.
        """
        self.fitness = max(bins_total) - min(bins_total)


class AntColony:
    def __init__(
        self,
        ant_paths: int,
        evaporation_rate: float,
        item_num: int = 500,
        is_bpp1: bool = True,
        max_fitness_evaluations: int = 10_000,
    ) -> None:
        self.ant_paths = ant_paths
        self.evaporation_rate = evaporation_rate
        self.item_num = item_num
        self.max_fitness_evaluations = max_fitness_evaluations
        self.is_bpp1 = is_bpp1
        self.bin_num = 10 if is_bpp1 else 50
        self.construction_graph = self.generate_construction_graph()

        # Output properties.
        self.best_fitness = 0
        self.result_fitness = 0

    def run(self) -> float:
        """
        Main iterator method for ant colony optimization.
        Create ant path for given number, then update pheromone levels.
        Repeat for the number of max fitness evaluations.
        :return: fitness of the best ant of the final population.
        """
        best_fitness = None
        fitness_evaluations = 0
        while fitness_evaluations < self.max_fitness_evaluations:

            # Generate ant paths.
            ants = []
            for _ in range(self.ant_paths):
                ant = Ant()
                ant.generate_ant_path(self.construction_graph, self.is_bpp1)
                ants.append(ant)

                # Once an ant path is generated, a fitness evaluation is performed.
                fitness_evaluations += 1

            # Update pheromone.
            for ant in ants:
                self.update_pheromone(ant)

            # Evaporate pheromone.
            self.evaporate_pheromone()

            # Store the best fitness of the population, then compare it to the current best.
            current_best_fitness = min(ant.fitness for ant in ants)
            if not best_fitness or current_best_fitness < best_fitness:
                best_fitness = current_best_fitness

        # Once process is finish, save the best fitness and the result.
        # The result fitness is the fitness of the best ant in the population at the end.
        self.best_fitness = best_fitness
        self.result_fitness = min(ant.fitness for ant in ants)

    def generate_construction_graph(self) -> List[List[float]]:
        """
        Generate construction graph by distributing pheromone over.
        :returns: 2D list representing construction graph.
        """
        construction_graph = []
        for _ in range(self.item_num):
            construction_graph.append(self.distribute_pheromone())
        return construction_graph

    def distribute_pheromone(self) -> List[float]:
        """
        Distribute pheromone over a bin.
        :returns: list of floats between 0 and 1.
        """
        return np.random.uniform(0, 1, self.bin_num)

    def update_pheromone(self, ant: Ant):
        """
        Update the graph pheromone based on ant path's fitness.
        :param ant: Ant instance.
        """
        pheromone_value = 100 / ant.fitness
        for count, node in enumerate(ant.path):
            self.construction_graph[count][node] += pheromone_value

    def evaporate_pheromone(self):
        """
        Evaporate the pheromone by multiplying each node in the construction
        graph by the evaporation rate.
        """
        for i in range(self.item_num):
            for j in range(self.bin_num):
                self.construction_graph[i][j] * self.evaporation_rate


if __name__ == "__main__":
    # Tunning parameters - adjust accordingly.
    ant_paths = 100
    evaporation_rate = 0.9
    is_bpp1 = True

    # For collecting data purposes. Should be False for the normal usage.
    running_experiments = False

    # Start process.
    print(
        f"Running ACO algorithm using BPP{1 if is_bpp1 else 2} with: \n- {ant_paths} ant paths \n- {evaporation_rate} evaporation rate \nLoading...\n"
    )
    if not running_experiments:
        ant_colony = AntColony(ant_paths, evaporation_rate, is_bpp1=is_bpp1)
        ant_colony.run()
        print(
            f"Process finished successfully. Results: \n- Best fitness: {ant_colony.best_fitness} \n- Output: {ant_colony.result_fitness}"
        )
    else:
        # Runs 5 trials of ACO and calculates averages.
        results = []
        best = []
        for _ in range(5):
            ant_colony = AntColony(ant_paths, evaporation_rate, is_bpp1=is_bpp1)
            ant_colony.run()
            results.append(ant_colony.result_fitness)
            best.append(ant_colony.best_fitness)
            print(f"({ant_colony.result_fitness}, {ant_colony.best_fitness})")
        print(f"({sum(results) / len(results)}, {sum(best) / len(best)})")
