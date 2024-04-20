import pygame
import neat
import time
import os
import random

pygame.font.init()

WINDOW_WIDTH = 500
WINDOW_HEIGHT = 800

GENERATION = 0

IMAGE_FOLDER_NAME = "imgs"

IDOL_CHARACTER_IMAGE = "character2.png"
UP_CHARACTER_IMAGE = "character1.png"
DOWN_CHARACTER_IMAGE = "character3.png"

OBSTACLE_IMAGE_NAME = "obstacle.png"
BACKGROUND_IMAGE_NAME = "bg.png"
GROUND_IMAGE_NAME = "base.png" 

CHARACTER_IMAGES = [pygame.transform.scale2x(pygame.image.load(os.path.join(IMAGE_FOLDER_NAME, IDOL_CHARACTER_IMAGE))), pygame.transform.scale2x(pygame.image.load(os.path.join(IMAGE_FOLDER_NAME, UP_CHARACTER_IMAGE))), pygame.transform.scale2x(pygame.image.load(os.path.join(IMAGE_FOLDER_NAME, DOWN_CHARACTER_IMAGE)))]
OBSTACLE_IMAGE = pygame.transform.scale2x(pygame.image.load(os.path.join(IMAGE_FOLDER_NAME, OBSTACLE_IMAGE_NAME)))
GROUND_IMAGE = pygame.transform.scale2x(pygame.image.load(os.path.join(IMAGE_FOLDER_NAME, GROUND_IMAGE_NAME)))
BACKGROUND_IMAGE = pygame.transform.scale2x(pygame.image.load(os.path.join(IMAGE_FOLDER_NAME, BACKGROUND_IMAGE_NAME)))

STAT_FONT = pygame.font.SysFont("comicsans", 50)

class Character:
    IMAGES = CHARACTER_IMAGES
    MAX_ROTATION_IN_DEGREES = 25
    ROTATION_VELOCITY = 20
    ANIMATION_TIME = 5
    
    def __init__(self, x, y) :
        self.x = x
        self.y = y
        self.tilt = 0
        self.tick_count = 0
        self.velocity = 0
        self.height = self.y
        self.img_count = 0
        self.img = self.IMAGES[0]
        
    def jump(self):
        self.velocity = -10.5
        self.tick_count = 0
        self.height = self.y
        
    def move(self):
        self.tick_count += 1
        distance = self.velocity*self.tick_count + 1.5*self.tick_count**2
        
        if distance >= 16:
            distance = 16
        
        if distance < 0:
            distance -= 2
        
        self.y = self.y + distance
        
        if distance < 0 or self.y < self.height + 50:
            if self.tilt < self.MAX_ROTATION_IN_DEGREES:
                self.tilt = self.MAX_ROTATION_IN_DEGREES
        else:
            if self.tilt > -90:
                self.tilt -= self.ROTATION_VELOCITY
                
    def draw(self, win):
        self.img_count += 1
        
        if self.img_count < self.ANIMATION_TIME:
            self.img = self.IMAGES[0]
        elif self.img_count < self.ANIMATION_TIME*2:
            self.img = self.IMAGES[1]
        elif self.img_count < self.ANIMATION_TIME*3:
            self.img = self.IMAGES[2]
        elif self.img_count < self.ANIMATION_TIME*4:
            self.img = self.IMAGES[1]
        elif self.img_count == self.ANIMATION_TIME*4 + 1:
            self.img = self.IMAGES[0]
            self.img_count = 0
            
        if self.tilt <= -80:
            self.img = self.IMAGES[1]
            self.img_count = self.ANIMATION_TIME*2
        
        rotated_image = pygame.transform.rotate(self.img, self.tilt)
        new_rectangle = rotated_image.get_rect(center=self.img.get_rect(topleft = (self.x, self.y)).center)
        win.blit(rotated_image, new_rectangle.topleft)
        
    def get_mask(self):
        return pygame.mask.from_surface(self.img)

class Obstacle:
    GAP = 200
    VELOCITY = 5
    
    def __init__(self, x):
        self.x = x
        self.height = 0
        
        self.top = 0
        self.bottom = 0
        self.OBSTACLE_TOP = pygame.transform.flip(OBSTACLE_IMAGE, False, True)
        self.OBSTACLE_BOTTOM = OBSTACLE_IMAGE
        
        self.passed = False
        self.set_heigth()
        
    def set_heigth(self):
        self.height = random.randrange(50,450)
        self.top = self.height - self.OBSTACLE_TOP.get_height()
        self.bottom = self.height + self.GAP
        
    def move(self):
        self.x -= self.VELOCITY
        
    def draw(self, win):
        win.blit(self.OBSTACLE_TOP, (self.x, self.top))
        win.blit(self.OBSTACLE_BOTTOM, (self.x, self.bottom))
        
    def collide(self, character):
        character_mask = character.get_mask()
        top_mask = pygame.mask.from_surface(self.OBSTACLE_TOP)
        bottom_mask = pygame.mask.from_surface(self.OBSTACLE_BOTTOM)
        
        top_offset = (self.x - character.x, self.top - round(character.y))
        bottom_offset = (self.x - character.x, self.bottom - round(character.y))
        
        bottom_collision_point = character_mask.overlap(bottom_mask, bottom_offset)
        top_collision_point = character_mask.overlap(top_mask, top_offset)
        
        if top_collision_point or bottom_collision_point:
            return True
        
        return False

class Base:
    VELOCITY = 5
    WIDTH = GROUND_IMAGE.get_width()
    IMAGE = GROUND_IMAGE
    
    def __init__(self, y):
        self.y = y
        self.x1 = 0
        self.x2 = self.WIDTH
        
    def move(self):
        self.x1 -= self.VELOCITY
        self.x2 -= self.VELOCITY
        
        if self.x1 + self.WIDTH < 0:
            self.x1 = self.x2 + self.WIDTH
        
        if self.x2 + self.WIDTH < 0:
            self.x2 = self.x1 + self.WIDTH
    def draw(self, win):
        win.blit(self.IMAGE, (self.x1, self.y))
        win.blit(self.IMAGE, (self.x2, self.y))

def draw_window(win, characters, obstacles, base, score, gen):
    win.blit(BACKGROUND_IMAGE, (0,0))
    
    for obstacle in obstacles:
        obstacle.draw(win)
        
    text = STAT_FONT.render("Score: " + str(score), 1, (255,255,255))
    win.blit(text, (WINDOW_WIDTH - 10 - text.get_width(), 10))

    text = STAT_FONT.render("Gen: " + str(gen), 1, (255,255,255))
    win.blit(text, (10, 10))
    
    base.draw(win)
    
    for character in characters:
        character.draw(win)
        
    pygame.display.update()

def main(genomes, config):
    global GENERATION
    GENERATION += 1
    nets = []
    ge = []  
    characters = []
    
    for _, genome in genomes:
        net = neat.nn.FeedForwardNetwork.create(genome, config)
        nets.append(net)
        characters.append(Character(230, 350))
        genome.fitness = 0
        ge.append(genome)
        
        
    base = Base(730)
    obstacles = [Obstacle(600)]
    win = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    clock = pygame.time.Clock()
    
    score = 0
    
    run = True
    while run:
        clock.tick(30)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                run = False
                pygame.quit()
                quit()
    
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                   character.jump()
        
        obstacle_indicator = 0
        if len(characters) > 0:
            if len(obstacles) > 1 and characters[0].x > obstacles[0].x + obstacles[0].OBSTACLE_TOP.get_width():
                obstacle_indicator = 1
        else:
            run = False
            break
        
        for character_position, character in enumerate(characters):
            character.move()
            ge[character_position].fitness += 0.1
            
            output = nets[character_position].activate((character.y, abs(character.y - obstacles[obstacle_indicator].height),
                                                        abs(character.y - obstacles[obstacle_indicator].bottom)))

            if output[0] > 0.5:
                character.jump()
        
        
        add_obstacle = False
        remove = []
        for obstacle in obstacles:
            for character_position, character in enumerate(characters):
                if obstacle.collide(character):
                    ge[character_position].fitness -= 1
                    characters.pop(character_position)
                    nets.pop(character_position)
                    ge.pop(character_position)
                
                if not obstacle.passed and obstacle.x < character.x:
                    obstacle.passed = True
                    add_obstacle = True
            
            if obstacle.x + obstacle.OBSTACLE_TOP.get_width() < 0:
                remove.append(obstacle)
                    
            obstacle.move()
            
        if add_obstacle:
            score += 1
            for genome in ge:
                genome.fitness += 5
                
            obstacles.append(Obstacle(600))
            
        for r in remove:
            obstacles.remove(r)
            
        for character_position, character in enumerate(characters):
            if character.y + character.img.get_height() >= 730 or character.y < 0:
                characters.pop(character_position)
                nets.pop(character_position)
                ge.pop(character_position)
            
        base.move()
        draw_window(win, characters, obstacles, base, score, GENERATION)
        

def run(config_path):
    config = neat.config.Config(neat.DefaultGenome, neat.DefaultReproduction,
                                neat.DefaultSpeciesSet, neat.DefaultStagnation,
                                config_path)
    
    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)
    
    winner = population.run(main, 50)

if __name__ == "__main__":
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, "config.txt")
    run(config_path)