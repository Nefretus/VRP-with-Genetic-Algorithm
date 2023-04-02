import pygame as pg
import argparse
import numpy as np
import random

#klasa ktora, reprezentuje gen czyli miasto (klienta)
class Gene: 
    def __init__(self, nr, pos_x, pos_y):
        self.city_nr = int(nr)
        self.pos_x = int(pos_x)
        self.pos_y = int(pos_y)
        self.distances = {} 
    
    def calculate_distance(self, other_city):
        self.distances[other_city.city_nr] = np.sqrt((other_city.pos_x - self.pos_x) ** 2 + (other_city.pos_y - self.pos_y) ** 2)

#klasa reprezentujaca jedno rozwiazanie - konfiguracja miast
class Individual:
    def __init__(self, route, depot):
        self.genes = [random.random() for _ in range(len(route))] #encoded
        self.route = sorted(route, key=lambda e : e.city_nr) # mozeliwe ze to niepotrzebne ale zostawiam
        self.depot = depot
        self.calculate_solution_distance()
        
    def calculate_solution_distance(self):
        self.solution = self.decode_solution() #decoded
        self.distance = 0
        for idx, current_city in enumerate(self.solution):
            next_city = self.solution[(idx + 1) % len(self.solution)]
            self.distance += current_city.distances[next_city.city_nr]

    def decode_solution(self):
        return [self.route[idx] for idx in np.argsort(self.genes)]

#zbior rozwiazan - populacja
class Population:
    def __init__(self):
        self.generation = []
    
    def add_solution(self, route, depot):
        self.generation.append(Individual(route, depot))
    
    def get_best_solution(self):
        #narazie poprostu zwracam najlepsze rozwiazanie jakie sie wygenerowalo xd
        self.generation.sort(key=lambda solution : solution.distance)
        return self.generation[0]

#klasa zarzadzajaca praca algorytmu        
class GeneticAlgorithm:
    def __init__(self, pop_size, gen_count):
        self.pop_size = pop_size
        self.gen_count = gen_count
        self.best_solutions = []
        
    def main_loop(self):
        self.load_data()
        for route in self.routes:
            self.create_initial_generation(route)
            for _ in range(self.gen_count):
                #dalej robimy turniej
                #crossover
                #mutacja
                #....
                break
            self.best_solutions.append(self.population.get_best_solution())  
        
    def load_data(self):
        with open('sampleData.txt', 'r') as f:
            (self.truck_count, self.delivery_cooldown, self.time_limit) = f.readline().split()
            self.depot = Gene(*f.readline().split())
            self.cities = [Gene(*line.split()) for line in f.readlines()]
        for city in self.cities:
            self.depot.calculate_distance(city)
            for other_city in self.cities:
                city.calculate_distance(other_city)
        self.cities.sort(key=lambda city : (city.pos_x, city.pos_y))
        self.routes = np.array_split(self.cities, int(self.truck_count)) #route to trasa dla jednego pojazdu
    
    def create_initial_generation(self, route):
        self.population = Population()
        for nr in range(self.pop_size):
            self.population.add_solution(route, self.depot)
    
class GUI:
    def __init__(self, width, height):
        self.width = width
        self.height = height
        self.window = pg.display.set_mode((width, height))
        pg.display.set_caption('VRP with Genetic algorithm')
        
    def update(self, solutions, cities, depot):
        pg.draw.rect(self.window, pg.Color('white'), pg.Rect(0, 0, self.width, self.height))
        colors_list = [pg.Color('red'), pg.Color('green'), pg.Color('purple'), pg.Color('yellow'), pg.Color('black')]
        colors_list_idx = 0
        normalized_coords = []
        for solution in solutions:
            x, y = self.normalize(cities, depot.pos_x, depot.pos_y)
            normalized_coords.append((x, y))
            pg.draw.circle(self.window, pg.Color('green'), (x, y), 5)
            for city in solution.decode_solution():
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
    ga = GeneticAlgorithm(args.popSize, args.genCount)
    ga.main_loop()
    gui = GUI(900, 600)
    running = True
    while running:
        for event in pg.event.get():
            if event.type == pg.QUIT:
                running = False
        gui.update(ga.best_solutions, ga.cities, ga.depot)
        
if __name__ == '__main__':
    pg.init()
    parser = argparse.ArgumentParser()
    parser.add_argument('--popSize', type=int, default=30, required=False,
                        help="Enter the population size")
    parser.add_argument('--genCount', type=int, default=50, required=False,
                        help="Number of generations to run")
    main(parser.parse_args())
    pg.quit()
