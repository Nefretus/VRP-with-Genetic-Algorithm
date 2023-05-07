import pygame as pg
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt

MAX_VEHICLE_DISTANCE = 1000

class Gene: 
    def __init__(self, city_nr, pos_x, pos_y):
        self.nr = int(city_nr)
        self.pos_x = int(pos_x)
        self.pos_y = int(pos_y)
        self.distances = {} 
    
    def calculate_distance(self, other_city):
        self.distances[other_city.nr] = np.sqrt((other_city.pos_x - self.pos_x) ** 2 + (other_city.pos_y - self.pos_y) ** 2)

class Individual:
    def __init__(self, cities, depot, random_keys=None):
        self.cities = cities
        self.depot = depot
        if random_keys: self.random_keys = random_keys #generate individual from existing solution
        else: self.random_keys = [random.random() for _ in range(len(cities))] #generate new solution, wtih random keys, example: [0.21, 0.51, 0.13...]
        self.routes = self.route_to_subroutes(self.decode_solution()) #example: [[1, 5], [7, 12, 24],...]
        self.calculate_solution_fitness()
    
    def route_to_subroutes(self, solution):
        global MAX_VEHICLE_DISTANCE
        curr_distance = 0
        prev_city = self.depot
        curr_route = []
        routes = []
        for city in solution:
            if curr_distance == 0:
                curr_distance += self.depot.distances[city.nr] 
            if curr_distance != 0 or len(curr_route) != 0: 
                curr_distance += prev_city.distances[city.nr]
            prev_city = city
            if curr_distance + self.depot.distances[city.nr] <= MAX_VEHICLE_DISTANCE: 
                curr_route.append(city)
            else:
                curr_distance += city.distances[self.depot.nr]
                routes.append(curr_route)
                curr_route = [city]
                curr_distance = self.depot.distances[city.nr]
        routes.append(curr_route)
        return routes
        
    def calculate_solution_fitness(self, unit_cost=1):
        self.total_distance = 0
        for route in self.routes:
            sub_distance = 0
            route_with_depot = [self.depot] + route + [self.depot]
            for idx, current_city in enumerate(route_with_depot[:-1]):
                next_city = route_with_depot[idx + 1]
                sub_distance += current_city.distances[next_city.nr] * unit_cost
            self.total_distance += sub_distance

    def decode_solution(self) -> Gene:
        return [self.cities[idx] for idx in np.argsort(self.random_keys)]

from enum import Enum
class Mutation(Enum):
    SWAP = 1 #mutacja w postaci zamiany dwoch losowych genow w chromosomie
    NEW_INDIVIDUAL  = 2 #mutacja w postaci nowego chromosomu
    
class Crossover(Enum):
    ONE_POINT = 1
    TWO_POINT = 2
    UNIFORM = 3
    
class GeneticAlgorithm:
    def __init__(self, pop_size, gen_count, crossover_type, mutation_type):
        self.pop_size = pop_size
        self.gen_count = gen_count
        self.crossover_type = crossover_type
        self.mutation_type = mutation_type
        self.population = []
        #parametry algorytmu
        if mutation_type == Mutation.NEW_INDIVIDUAL: self.mutations_per_generation = int(0.1 * self.pop_size)
        else: self.mutations_per_generation = 0
        self.best_to_next_gen_count = pop_size // 5 #ile najelepszych rozwiazan przechodzi automatycznie do next generacji - 20%
        self.tournament_size = pop_size - self.best_to_next_gen_count - self.mutations_per_generation
        self.mutation_rate = 0.1
        
    def main_loop(self, GUI):
        self.load_data()
        curr_best_soulution = []
        self.create_initial_generation()
        for nr in range(self.gen_count):
            next_generation = []
            for i in range(self.best_to_next_gen_count):
                self.add_individual(next_generation, self.population[i])
            parents = self.tournament()
            if self.crossover_type == Crossover.ONE_POINT: self.one_point_crossover(parents, next_generation)
            elif self.crossover_type == Crossover.TWO_POINT: self.two_point_crossover(parents, next_generation)
            else: self.uniform_crossover(parents, next_generation)
            if self.mutation_type == Mutation.NEW_INDIVIDUAL: self.mutate_new_individual(next_generation)
            else: self.mutation_swap(next_generation)
            self.population = next_generation    
            curr_best_soulution.append(self.best_solution.total_distance)
            GUI.update(self.best_solution.routes, self.cities, self.depot, nr + 1, self.best_solution.total_distance)
        plt.plot(range(self.gen_count), curr_best_soulution)
            
    def load_data(self):
        with open('sampleData.txt', 'r') as f:
            global MAX_VEHICLE_DISTANCE
            MAX_VEHICLE_DISTANCE= int(f.readline())
            self.depot = Gene(*f.readline().split())
            self.cities = [Gene(*line.split()) for line in f.readlines()]
        for city in self.cities:
            self.depot.calculate_distance(city)
            for other_city in self.cities:
                city.calculate_distance(other_city)
                city.calculate_distance(self.depot)
    
    def create_initial_generation(self):
        for _ in range(self.pop_size):
            soulution = Individual(self.cities, self.depot)
            self.add_individual(self.population, soulution) 
    
    def add_individual(self, population, individual):
        population.append(individual)
        population.sort(key=lambda solution : solution.total_distance)
        self.best_solution = self.population[0]
        
    def tournament(self, roulette_selection=True):
        parents = []
        while True:
            #metoda ruletki
            if roulette_selection:
                fitness_sum = np.sum([individual.total_distance for individual in self.population])
                prob_associated_with_each_entry = [individual.total_distance / fitness_sum for individual in self.population][::-1]
                parents.append(tuple(np.random.choice(self.pop_size, 2, p=prob_associated_with_each_entry)))
            #metoda turniejowa
            else:
                k_parameter = 3
                next_parents = []
                for _ in range(2):
                    parents_candidate = np.random.choice(self.pop_size - 1, k_parameter)
                    idx = np.argsort([self.population[parent].total_distance for parent in parents_candidate])[0]
                    next_parents.append(parents_candidate[idx])
                parents.append(tuple(next_parents))
            if len(parents) == self.tournament_size:
                return parents
            
    def one_point_crossover(self, parents, next_generation):
        split_point = random.randint(0, len(self.cities) - 1)
        for parent in parents:
            child = [self.population[parent[0]].random_keys[i] if i < split_point else self.population[parent[1]].random_keys[i] for i in range(len(self.cities))]
            self.add_individual(next_generation, Individual(self.cities, self.depot, random_keys=child))

    def two_point_crossover(self, parents, next_generation):
        point1 = random.randint(0, len(self.cities) - 1)
        point2 = random.randint(0, len(self.cities) - 1)
        point1, point2 = min(point1, point2), max(point1, point2)
        for parent in parents:
            child = [self.population[parent[1]].random_keys[i] if i >= point1 and i <= point2 else self.population[parent[0]].random_keys[i] for i in range(len(self.cities))]
            self.add_individual(next_generation, Individual(self.cities, self.depot, random_keys=child))
    
    def uniform_crossover(self, parents, next_generation):
        threshold = 0.5
        for parent in parents:
            child = [self.population[parent[1]].random_keys[i] if random.random() < threshold else self.population[parent[0]].random_keys[i] for i in range(len(self.cities))]
            self.add_individual(next_generation, Individual(self.cities, self.depot, random_keys=child))
        
    def mutate_new_individual(self, next_generation):
        for _ in range(self.mutations_per_generation):
            self.add_individual(next_generation, Individual(self.cities, self.depot))
    
    def mutation_swap(self, next_generation):
        for idx, individual in enumerate(next_generation):
            if random.random() < self.mutation_rate:
                pos1 = random.randint(0, len(self.cities) - 1)
                pos2 = random.randint(0, len(self.cities) - 1)
                temp = individual.random_keys[pos2]
                individual.random_keys[pos2] = individual.random_keys[pos1]
                individual.random_keys[pos1] = temp
                individual = Individual(self.cities, self.depot, random_keys=individual.random_keys) #recalculate fitness
        next_generation.sort(key=lambda solution : solution.total_distance)  
        
class GUI:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.window = pg.display.set_mode((width, height))
        pg.display.set_caption('VRP with Genetic algorithm')
        
    def update(self, routes, cities, depot, gen_nr, min_distance):
        pg.draw.rect(self.window, pg.Color('white'), pg.Rect(0, 0, self.width, self.height))
        colors_list = [pg.Color('red'), pg.Color('green'), pg.Color('purple'), pg.Color('orange'), pg.Color('blue'),  pg.Color('grey')]
        colors_list_idx = 0
        normalized_coords = []
        font = pg.font.SysFont('comicsans', 25)
        self.window.blit(font.render(f'Generation: {gen_nr}', 1, pg.Color("black")), (10, 10))
        self.window.blit(font.render(f'Minimum distance: {min_distance}', 1, pg.Color("black")), (10, 10 + font.get_height()))
        for route in routes:
            x, y = self.normalize(cities, depot.pos_x, depot.pos_y)
            normalized_coords.append((x, y))
            pg.draw.circle(self.window, pg.Color('green'), (x, y), 5)
            for city in route:
                x, y = self.normalize(cities, city.pos_x, city.pos_y)
                normalized_coords.append((x, y))
                pg.draw.circle(self.window, pg.Color('black'), (x, y), 5)
            for idx, curr_point in enumerate(normalized_coords):
                next_point = normalized_coords[(idx + 1) % len(normalized_coords)]
                pg.draw.line(self.window, colors_list[colors_list_idx], curr_point, next_point, 2)
            normalized_coords.clear()
            colors_list_idx += 1
        pg.display.update()
        
    def normalize(self, cities, x, y):
        x_coords = [city.pos_x for city in cities]
        y_coords = [city.pos_y for city in cities]
        x_norm = (x - np.min(x_coords)) / (np.max(x_coords) - np.min(x_coords)) * (self.width * 0.8) + self.width * 0.1
        y_norm = (y - np.min(y_coords)) / (np.max(y_coords) - np.min(y_coords)) * (self.height* 0.8) + self.height * 0.1
        return x_norm, y_norm
        
def main(args):
    ga = GeneticAlgorithm(args.popSize, args.genCount, Crossover.UNIFORM, Mutation.NEW_INDIVIDUAL) #rozne warianty
    ga.main_loop(GUI(900, 600))
    plt.title("Genetic algorithm progress")
    plt.xlabel("Generation")
    plt.ylabel("Min distance")
    plt.show()
    
if __name__ == '__main__':
    pg.init()
    parser = argparse.ArgumentParser()
    parser.add_argument('--popSize', type=int, default=400, required=False,
                        help="Enter the population size")
    parser.add_argument('--genCount', type=int, default=200, required=False,
                        help="Number of generations to run")
    main(parser.parse_args())
    pg.quit()
