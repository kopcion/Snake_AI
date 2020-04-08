import random
from player import Player
import numpy as np
import pygame
import time
import threading
from operator import itemgetter
from pygame.locals import *
from mutation import simulated_binary_crossover as SBX
from mutation import single_point_binary_crossover as SPBX
import sys

boardsize = int(sys.argv[1])

class Snake:
    def __init__(self, weights = None, biases = None, lifespan = int((boardsize**2) * 1.6)  + 5 * boardsize):
        self.lifespan = lifespan
        self.lifetime = 0
        self.points = 0
        self.fitness = 0
        self.boardsize = boardsize
        self.body = [(random.randint(1, boardsize-2), random.randint(1, boardsize-2)), None]
        if random.random() < 0.5:
            self.body[1] = (int(self.body[0][0]) + random.randint(0,1)*2-1, int(self.body[0][1]))
        else:
            self.body[1] = (int(self.body[0][0]), int(self.body[0][1]) + random.randint(0,1)*2-1)
        self.length = len(self.body)
        self.player = Player(weights, biases)
        self.apple_x = 0
        self.apple_y = 0
        if self.body[0][0] < self.body[1][0]:
            self.direction = -1
        elif self.body[0][0] > self.body[1][0]:
            self.direction = 1
        elif self.body[0][1] > self.body[1][1]:
            self.direction = 4
        else:
            self.direction = 2
        self.tail_direction = self.get_tail_direction()# up, down, left, right
        self.time_since_last_apple = 0
        self.generate_apple()
        
    def fitness_funtion(self):
        self.fitness = max(self.lifetime + ((2**self.points) + 500*(self.points**2.1)) - ((.25 * self.lifetime) * self.points), .1)
        return self.fitness
    
    def outside_board(self, x, y):
        return x >= self.boardsize or y >= self.boardsize or x < 0 or y < 0

    def in_direction(self, x, y, dir_x, dir_y):
        apple = 0
        body = 0
        wall = 0
        dist = 0
        while True :
            dist += 1
            x += dir_x
            y += dir_y
            if (x,y) in self.body and body == 0:
                body = 1/dist
            if self.outside_board(x,y):
                wall = 1/dist
                return apple, body, wall
            if x == self.apple_x and y == self.apple_y :
                apple = 1/dist

    def vision(self):
        vision_table = np.zeros((32, 1))
        ind = 0
        (head_x, head_y) = self.body[0]
        directions = [(1,0),(1,1),(0,1),(-1,1),(-1,0),(-1,-1),(0,-1),(1,-1)]
        for x,y in directions:
            apple, body, wall = self.in_direction(head_x, head_y, x,y)
            vision_table[ind + 0] = body
            vision_table[ind + 1] = wall
            vision_table[ind + 2] = apple
            ind += 3
        if self.direction == -1:
            vision_table[24] = 1
        if self.direction == 1:
            vision_table[25] = 1
        if self.direction == 2:
            vision_table[26] = 1
        if self.direction == 4:
            vision_table[27] = 1
        vision_table[28 + self.get_tail_direction()][0] = 1
        return vision_table

    def generate_apple(self):
        possibilities = [ divmod(i, self.boardsize) for i in range(self.boardsize**2) if divmod(i, self.boardsize) not in self.body]
        self.apple_x, self.apple_y = random.choice(possibilities)
        self.apple_x = min(self.apple_x, self.boardsize-1)
        self.apple_y = min(self.apple_y, self.boardsize-1)
        
    #direction values -1, 1 (up, down) or 2, 4 (left, right)
    def move(self, direction):
        if direction+2 == self.direction or direction-2 == self.direction:
            return False

        x, y = self.body[0]
        if direction < 2:
            x += direction
        else:
            y += (direction - 3)

        if x == self.apple_x and y == self.apple_y:
            tmp_body  = [(x,y)]
            for i in self.body:
                tmp_body.append(i)
            self.length += 1
            self.body = tmp_body
            self.points += 1
            self.time_since_last_apple = 0
            self.generate_apple()
        elif self.outside_board(x,y): # collision with a wall
            return False
        elif (x,y) in self.body[:len(self.body)-1]: # collision with self
            return False
        else: # simple move
            tmp_body = [(x, y)]
            for i in self.body[:(self.length-1)]:
                tmp_body.append(i)
            self.time_since_last_apple += 1
            if self.time_since_last_apple > self.lifespan:
              return False
            self.body = tmp_body
            
        self.lifetime += 1
        self.direction = direction

        return True

    def get_tail_direction(self):
        tail_x, tail_y = self.body[-1]
        x, y = self.body[-2]
        if tail_x > x:
            return 0
        if tail_x < x:
            return 1
        if tail_y > y:
            return 2
        if tail_y < y:
            return 3

    def ai_move(self):
        return self.move(self.player.move(self.vision()))
    def save_snake(self):
        w,b = self.player.getNetwork()        
        return w,b 
            
class Game:

    def __init__(self, weights, biases, boardsize = boardsize):
        self.windowWidth = 32 * boardsize
        self.windowHeight = 32 * boardsize
        self._running = True
        self._display_surf = None
        self._image_surf = None
        self._apple = None
        self.player = Snake(weights, biases)

    def on_init(self):
        pygame.init()
        self._display_surf  = pygame.display.set_mode((self.windowWidth,self.windowHeight), pygame.HWSURFACE)
        
        pygame.display.set_caption('SNAKE')
        self._running = True
        self._apple = pygame.image.load("apple.png").convert()
        self._image_surf = pygame.image.load("pygame.png").convert()
 
    def on_event(self, event):
        if event.type == QUIT:
            self._running = False

    def on_loop(self):
        pass
    
    def on_render(self):
        self._display_surf.fill((0,0,0))

        self._display_surf.blit(self._apple, (self.player.apple_x * 32,  self.player.apple_y * 32))
        for x,y in self.player.body:
          self._display_surf.blit(self._image_surf,(x * 32,y * 32))
        pygame.display.flip()
 
    def on_cleanup(self):
        pygame.quit()
 
    def on_execute(self):
        if self.on_init() == False:
            self._running = False
 
        #direction values -1, 1 (up, down) or 2, 4 (left, right)
        while self._running:
            self._running = self.player.ai_move()
            self.on_loop()
            self.on_render()
            time.sleep (50.0 / 1000.0)
        self.on_cleanup()


class Selection:
    def __init__(self, weights):
        self.wheel = [None] * len(weights)
        self.wheel[0] = weights[0][0]
        for i in range(1, len(weights)):
            self.wheel[i] = self.wheel[i-1] + weights[i][0]

    def select(self):
        point = np.random.uniform(0, self.wheel[-1])
        ind = 0
        while point > self.wheel[ind]:
            ind += 1
        return ind

individuals = 1500
parents = 500
offsprings = parents * 2
prob_mutation = 0.05
mutation_scale = 0.3
generations = 2500
fitnesses = [None] * individuals
scores = [None] * individuals
threads = []
networks = [None] * individuals
networks_b = [None] * individuals
apple_positions = [None] * individuals
starting_bodies = [None] * individuals

def mutate(gene):
    mutation_indexes = np.random.random(gene.shape) < prob_mutation
    mutation = np.random.normal(size=gene.shape) * mutation_scale
    gene[mutation_indexes] += mutation[mutation_indexes]
    for x in gene:
        x = np.clip(x, -1, 1)
    return gene

def make_new_generation():
  selection = Selection(fitnesses[:parents])
  new_generation = networks[:parents]
  new_generation_b = networks_b[:parents]
  
  while len(new_generation) < individuals:
      id = selection.select()
      p1_w = np.array(new_generation[id])
      p1_b = np.array(new_generation_b[id])
      id = selection.select()
      p2_w = np.array(new_generation[id])
      p2_b = np.array(new_generation_b[id])
      c1_w = np.zeros_like(p1_w)
      c1_b = np.zeros_like(p1_b)
      c2_w = np.zeros_like(p2_w)
      c2_b = np.zeros_like(p2_b)
      for i in range(len(c1_w)):
          if random.random() < 0.5:
            c1_w[i], c2_w[i] = SPBX(p1_w[i], p2_w[i])
            c1_b[i], c2_b[i] = SPBX(p1_b[i], p2_b[i])
          else:
            c1_w[i], c2_w[i] = SBX(p1_w[i], p2_w[i])
            c1_b[i], c2_b[i] = SBX(p1_b[i], p2_b[i])
      new_generation.append(mutate(c1_w))
      new_generation.append(mutate(c2_w))
      new_generation_b.append(mutate(c1_b))
      new_generation_b.append(mutate(c2_b))
  
  return new_generation, new_generation_b
  #tmp = new_generation[:parents]
  #tmp.extend(mutate(new_generation[parents:]))
  #tmp_b = new_generation_b[:parents]
  #tmp_b.extend(mutate(new_generation_b[parents:]))
  #return tmp, tmp_b
  #new_generation = new_generation[:individuals].extend(mutate(new_generation[individuals:]))
  #new_generation = new_generation[:individuals].extend(mutate(new_generation[individuals:]))
  #new_generation_b = new_generation_b[:individuals].extend(mutate(new_generation_b[individuals:]))
  #return new_generation, new_generation_b
  #new_generation = [None] * int(individuals * offspirng_ratio)
  #new_generation = [None] * int(individuals * offspirng_ratio)
  #new_generation_b = [None] * int(individuals * offspirng_ratio)
  #for i in range(len(new_generation)):
  #    id = selection.select()
  #    new_generation[i] = np.array(networks[id])
  #    new_generation_b[i] = np.array(networks_b[id])

  #new_generation = np.array(new_generation)
  #new_generation_b = np.array(new_generation_b)

  #for i in range(0, int(len(new_generation)/2), 2):
  #for i in range(0, int(len(new_generation)), 2):
  #  c1_w, c2_w = SBX(new_generation[i], new_generation[i+1])
  #  c1_b, c2_b = SBX(new_generation_b[i], new_generation_b[i+1])
  #  new_generation[i] = c1_w
  #  new_generation[i+1] = c2_w
  #  new_generation_b[i] = c1_b
  #  new_generation_b[i+1] = c2_b

  #for i in range(int(len(new_generation)/2), len(new_generation) - 1, 2):
  #  c1_w, c2_w = SPBX(new_generation[i], new_generation[i+1])
  #  c1_b, c2_b = SPBX(new_generation[i], new_generation[i+1])
  #  new_generation[i] = c1_w
  #  new_generation[i+1] = c2_w
  #  new_generation_b[i] = c1_b
  #  new_generation_b[i+1] = c2_b

  #new_generation = mutate(new_generation)
  #new_generation_b = mutate(new_generation_b)
  #tmp = [x for x in new_generation]
  #tmp_b = [x for x in new_generation_b]
  #tmp.extend(networks[:int(individuals/3)])
  #tmp_b.extend(networks_b[:int(individuals/3)])
  #random.shuffle(tmp)
  #random.shuffle(tmp_b)
  #return tmp[:individuals], tmp_b[:individuals]
  # networks.extend(new_generation)
  # networks_b.extend(new_generation_b)
  # random.shuffle(networks)
  # random.shuffle(networks_b)
  # return networks[:individuals], networks[:individuals]

def snake_background(id, generation, run):
  snake = Snake(networks[id], networks_b[id])
  while run:
    run = snake.ai_move()
  fitnesses[id] = (snake.fitness_funtion(), id)
  scores[id] = snake.points
  (networks[id], networks_b[id]) = snake.player.getNetwork()
  
def snake_forground(game):
    game.on_execute()

def run_generation(generation):
  for i in range(individuals):
    threads.append(threading.Thread(target=snake_background, args=(i, generation, True,)))
    threads[i].start()

  for thread in threads:
    thread.join()

best_results = []
  
best_w = None
best_b = None

if __name__ == '__main__':
    beginning = 0
    if len(sys.argv) == 5:
        networks = np.load(sys.argv[2], allow_pickle=True)
        networks_b = np.load(sys.argv[3], allow_pickle=True)
        beginning = int(sys.argv[4])
    for j in range (beginning, generations):
        run_generation(j)
        fitnesses = sorted(fitnesses, key=itemgetter(0), reverse=True)
        scores = sorted(scores, reverse=True)
        networks_tmp = [None] * individuals 
        networks_tmp_b = [None] * individuals
        for i in range(individuals):
            networks_tmp[i] = networks[fitnesses[i][1]]
            networks_tmp_b[i] = networks_b[fitnesses[i][1]]
        networks = networks_tmp
        networks_b = networks_tmp_b
  
        if j%50 == 0:
            np.save('all_weights%d'%(j), networks)
            np.save('all_biases%d'%(j), networks_b)
            best_tmp = np.array(best_results)
            np.save('best_results', best_tmp)
        if j%5 == 0 and j > 0:
            best_b = networks_b[0]
            best_w = networks[0]
            if j%20 == 0:      
                np.save('biases%d'%(j), best_b)
                np.save('weights%d'%(j), best_w)
            theApp = Game(best_w, best_b)
            threading.Thread(target=snake_forground, args=(theApp,)).start()
  
        networks_t, networks_t_b = make_new_generation()
        for i in range(individuals):
            networks[i] = networks_t[i]
            networks_b[i] = networks_t_b[i]
        threads = []
        best_results.append('generation %d\n\tmax points is %d with average %f, boardsize %d'%(j, max(scores), sum(scores[:parents])/parents, boardsize))
        print('generation %d\n\tmax points is %d with average %f, boardsize %d'%(j, max(scores), sum(scores[:parents])/parents, boardsize))

    best_results = np.array(best_results)
    np.save('best_results', best_results)
