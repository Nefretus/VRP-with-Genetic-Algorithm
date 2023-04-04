import pygame as pg
import argparse
import numpy as np
import random
import matplotlib.pyplot as plt

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
    def __init__(self, route, depot, solution=None): #jezeli None to wygeneruj nowe rozwiazanie
        if solution:
            self.genes = solution
        else:
            self.genes = [random.random() for _ in range(len(route))] #encoded, tworzone randomowo
        self.route = sorted(route, key=lambda e : e.city_nr) # mozeliwe ze to niepotrzebne ale zostawiam
        self.depot = depot
        self.calculate_solution_fitness()
        
    def calculate_solution_fitness(self): #trzeba dodac kary za zle rozwiazania
        self.solution = self.decode_solution() #decoded
        self.distance = 0
        for idx, current_city in enumerate(self.solution):
            next_city = self.solution[(idx + 1) % len(self.solution)]
            self.distance += current_city.distances[next_city.city_nr] #tutaj blad zapomniales o depot!!!!!!!!1

    def get_city(self, index):
        return self.genes[index]

    def decode_solution(self):
        return [self.route[idx] for idx in np.argsort(self.genes)]

#zbior rozwiazan - populacja
class Population:
    def __init__(self):
        self.generation = []
    
    def add_solution(self, solution):
        self.generation.append(solution)
        self.generation.sort(key=lambda solution : solution.distance) 
    
    def get_individual(self, index):
        return self.generation[index]
    
    def get_best_solution(self):
        #narazie poprostu zwracam najlepsze rozwiazanie jakie sie wygenerowalo xd
        return self.generation[0]

#klasa zarzadzajaca praca algorytmu        
class GeneticAlgorithm:
    def __init__(self, pop_size, gen_count):
        self.pop_size = pop_size
        self.gen_count = gen_count
        self.best_solutions = []
        #parametry algorytmu
        self.mutations_per_gen = int(0.1 * self.pop_size)
        #ile najelepszych rozwiazan przechodzi automatycznie do next generacji
        self.best_to_next_gen_count = pop_size // 5
        self.tournament_size = pop_size - self.best_to_next_gen_count - self.mutations_per_gen
        
    def main_loop(self):
        self.load_data()
        for route in self.routes:
            self.current_route = route
            self.create_initial_generation()
            y_axis = []
            for gen_nr in range(self.gen_count):
                next_generation = Population()
                for i in range(self.best_to_next_gen_count):
                    next_generation.add_solution(self.population.get_individual(i))
                self.one_point_crossover(self.tournament(), next_generation)
                self.mutate(next_generation)
                self.population = next_generation
                #printowanie wynikow ale wychodzi scierwoooooooooo
                y_axis.append(self.population.get_best_solution().distance)
            plt.plot(range(self.gen_count), y_axis)
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
    
    def create_initial_generation(self):
        self.population = Population()
        for nr in range(self.pop_size):
            self.population.add_solution(Individual(self.current_route, self.depot)) #troche mnie ten depot wkurwia
    
    #mozna zaimplementowac rozne rodzaje turniejow potem
    def tournament(self):
        parents = []
        duplicates_deleted = False
        while True:
            parent_nr1 = random.randint(self.best_to_next_gen_count, self.pop_size - 1)
            parent_nr2 = random.randint(self.best_to_next_gen_count, self.pop_size - 1)
            parents.append((parent_nr1, parent_nr2))
            if len(parents) == self.tournament_size:
                parents = list(set(parents)) 
                if len(parents) == self.tournament_size: #czy po usunieciu duplikatow mamy odpowiednio duzo rodzicow
                    return parents
            
    #tak samo mozna rozne crossovery, zobaczyc ktory najelpszy
    def one_point_crossover(self, parent_nrs, next_generation):
        split_point = random.randint(0, len(self.current_route))
        for parent_nr in parent_nrs:
            child = []
            for i in range(len(self.current_route)):
                #tu bedzie troche pojebanie bo sa te random keys
                if i < split_point:
                    city = self.population.get_individual(parent_nr[0]).get_city(i)
                else:
                    city = self.population.get_individual(parent_nr[1]).get_city(i)
                child.append(city)
            next_generation.add_solution(Individual(self.current_route, self.depot, solution=child))

    def two_point_crossover(self, parents): # do implementacji
        pass
    
    def uniform_crossover(self, parents): # do implementacji
        pass
        
    def mutate(self, next_generation):
        for i in range(self.mutations_per_gen):
            next_generation.add_solution(Individual(self.current_route, self.depot)) #poprostu generuje nowe chromosomy randomowe
    
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
        plt.title("Genetic algorithm progress")
        plt.xlabel("Generation")
        plt.ylabel("Min distance")
        plt.show()
        
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
