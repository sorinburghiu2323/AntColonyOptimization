# AntColonyOptimisation

## 📖 About

Implementing Bin-Packing with Ant Colony Optimisation. The problem allows population of ants to construct their paths based on pheromone levels which is where the BPP comes to help. The ants select the node to go to the same way items are packed in bins, the outcome fitness being the largest weight of all bins minus the lowest weight of all bins.

The fitness value is then used to evaluate different experiments run with different parameter values:
- ant_paths: number of ants in a population
- evaporation_rate: decrease rate in pheromone at the end of a population run
- bpp1/bpp2: amount of bins used and weight of the items

Algorithm summary:
1. Generate construction graph and fill it with random pheromone levels.
2. Generate ant paths for the current population.
3. Update graph pheromone levels based on the ant paths in the population.
4. Evaporate pheromone.
5. If the max fitness evaluations was reached, end with the best fitness in the current ant population, else go to step 2.

## 🛠️ Setup

Run the python file `ant_colony_optimsation.py` using:
```
python ant_colony_optimsation.py
```

Adjusting tuning parameters is possible from the main runner method at the bottom of the file.
