from copy import deepcopy as dcpy
import pickle
import numpy as np
import pygame
from pygame.locals import *
from random import *
import argparse

pygame.init()

clock = pygame.time.Clock()
# fps could be changed to change speed of learning (without any repercussions)
fps = 60

screen_width = 864
screen_height = 936

screen = pygame.display.set_mode((screen_width, screen_height))
pygame.display.set_caption("Flappy Bird")

# Game Variables
ground_scroll = 0
scroll_speed = 4
pipe_gap = 150
pipe_frequency = 1500  # milliseconds
last_pipe = pygame.time.get_ticks() - pipe_frequency
game_over = False
score = 0
pass_pipe = False

# Font
FONT = pygame.font.SysFont('urwgothic', 60)

# Color
WHITE = (255, 255, 255)

bg = pygame.image.load('img/bg.png')
ground_img = pygame.image.load('img/ground.png')
button_img = pygame.image.load('img/restart.png')


def draw_text(text, font, text_col, x, y):
    img = font.render(text, True, text_col)
    screen.blit(img, (x, y))


# Defining class that will describe birds that will play game


class Bird(pygame.sprite.Sprite):
    def __init__(self, x, y):
        pygame.sprite.Sprite.__init__(self)

        self.counter = 0
        self.index = 0
        self.images = []
        for i in range(1, 4):
            img = pygame.image.load(f'img/bird{i}.png')
            self.images.append(img)
        self.image = self.images[self.index]
        self.rect = self.image.get_rect()
        self.rect.center = [x, y]

        self.vel = 0
        self.clicked = False
        self.flying = True
        self.jump = 0
        self.lifespan = 0

    def update(self):

        if self.flying == True:

            self.vel += 0.5
            if self.vel > 8:
                self.vel = 8
            if self.rect.bottom < 768:
                self.rect.y += int(self.vel)

        if game_over == False and self.flying == True:

            if self.jump == 1 and self.clicked == False:
                self.clicked = True
                self.vel = -10
            if self.jump == 0:
                self.clicked = False

            self.lifespan += 1
            self.counter += 1
            cooldown = 5

            if self.counter > 5:
                self.counter = 0
                self.index += 1
                self.index %= len(self.images)
            self.image = self.images[self.index]

            self.image = pygame.transform.rotate(
                self.images[self.index], self.vel * -2)


# Class for defining obstacles


class Pipe(pygame.sprite.Sprite):
    def __init__(self, x, y, position):
        pygame.sprite.Sprite.__init__(self)
        self.image = pygame.image.load('img/pipe.png')
        self.rect = self.image.get_rect()

        if position == 1:
            self.image = pygame.transform.flip(self.image, False, True)
            self.rect.bottomleft = [x, y - pipe_gap // 2]
        else:
            self.rect.topleft = [x, y + pipe_gap // 2]

    def update(self):
        self.rect.x -= scroll_speed
        if self.rect.right < 0:
            self.kill()


class Button():
    def __init__(self, x, y, image):
        self.image = image
        self.rect = self.image.get_rect()
        self.rect.topleft = [x, y]

    def draw(self):

        action = False

        pos = pygame.mouse.get_pos()

        if self.rect.collidepoint(pos):
            if pygame.mouse.get_pressed()[0] == 1:
                action = True

        screen.blit(self.image, (self.rect.x, self.rect.y))

        return action


# Class whose objects will be specimen of a neural network model specified to a specific bird
class Specimen:
    def _init_(self):
        self.NINPUTS = 8
        self.NOUTPUTS = 1
        self.NHIDDEN = 1
        self.HIDDENSIZE = 16

        self.ques_mutated = False

        self.inputLayer = np.zeros((self.NINPUTS, self.HIDDENSIZE))
        self.interLayers = np.zeros(
            (self.HIDDENSIZE, self.HIDDENSIZE, self.NHIDDEN))
        self.outputLayer = np.zeros((self.HIDDENSIZE, self.NOUTPUTS))

        self.inputBias = np.zeros((self.HIDDENSIZE))
        self.interBiases = np.zeros((self.HIDDENSIZE, self.NHIDDEN))
        self.outputBias = np.zeros((self.NOUTPUTS))

        self.inputValues = np.zeros((self.NINPUTS))
        self.outputValues = np.zeros((self.NOUTPUTS))

    def activation(self, value):
        return 0 if value < 0 else value

    def evaluate(self):
        terms = np.dot(self.inputValues, self.inputLayer) + self.inputBias
        for i in range(self.NHIDDEN):
            terms = np.array([self.activation(np.dot(terms, self.interLayers[j, :, i]))
                              for j in range(self.HIDDENSIZE)]) + self.interBiases[:, i]
        self.outputValues = np.dot(terms, self.outputLayer) + self.outputBias

    def mutate(self):
        RATE = 10.0
        PROB = 0.05

        self.ques_mutated = True

        for i in range(self.NINPUTS):
            for j in range(self.HIDDENSIZE):
                if(random() < PROB):
                    self.inputLayer[i, j] += gauss(0.0, RATE)
        for i in range(self.HIDDENSIZE):
            for j in range(self.HIDDENSIZE):
                for k in range(self.NHIDDEN):
                    if(random() < PROB):
                        self.interLayers[i, j, k] += gauss(0.0, RATE)
        for i in range(self.HIDDENSIZE):
            for j in range(self.NOUTPUTS):
                if(random() < PROB):
                    self.outputLayer[i, j] += gauss(0.0, RATE)

        for i in range(self.HIDDENSIZE):
            if(random() < PROB):
                self.inputBias[i] += gauss(0.0, RATE)
        for i in range(self.HIDDENSIZE):
            for j in range(self.NHIDDEN):
                if(random() < PROB):
                    self.interBiases[i, j] += gauss(0.0, RATE)
        for i in range(self.NOUTPUTS):
            if(random() < PROB):
                self.outputBias[i] += gauss(0.0, RATE)

    def calc_fitness(self, bird):
        return bird.lifespan

    def apply_input(self, bird):
        rect = bird.rect
        t, b, r = rect.top, rect.bottom, rect.right

        dist = []
        for pipe in pipe_group:
            temp = [pipe.rect.left, pipe.rect.bottom, pipe.rect.top]
            dist.append(temp)

        dist.sort()

        if len(pipe_group) < 2:

            self.inputValues[0] = 10000
            self.inputValues[1] = 1000
            self.inputValues[2] = 1000
            self.inputValues[3] = 10000
            self.inputValues[4] = bird.vel
            self.inputValues[5] = 1

        else:
            self.inputValues[0] = dist[0][0]-r
            self.inputValues[1] = dist[0][1]-t
            self.inputValues[2] = dist[1][2]-b
            self.inputValues[3] = dist[1][0]-r
            self.inputValues[4] = bird.vel
            self.inputValues[5] = 0

        self.inputValues[6] = t
        self.inputValues[7] = screen_height-b
        self.evaluate()

        # bird.jump = randint(0,1)
        bird.jump = self.outputValues[0] >= 0


# Parsing arguements
parser = argparse.ArgumentParser()
parser.add_argument('--action', help='learn, play, showtime')
args = parser.parse_args()

# Parsing arguements
parser = argparse.ArgumentParser()
parser.add_argument('--action', help='learn, play, showtime')
args = parser.parse_args()

# Block where model trains
if args.action == 'learn':
    fps=120

    def reset_game():
        pipe_group.empty()

    bird_group = pygame.sprite.Group()
    pipe_group = pygame.sprite.Group()

    button = Button(screen_width//2-50, screen_height//2-100, button_img)

    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('Flappy Bird')

    population_size = 50
    select_best = 5
    epochs = 100

    gen = []
    for _ in range(select_best):
        flappy = Bird(100, screen_height//2)
        bird_group.add(flappy)
        gen.append([Specimen(), flappy])

    for q in range(epochs):
        print(f'Epoch No.: {q}')
        if game_over:
            break

        pipe_height = randint(-100, 100)
        pipe1 = Pipe(screen_width-300, screen_height//2+pipe_height, 1)
        pipe2 = Pipe(screen_width-300, screen_height//2+pipe_height, -1)
        pipe_group.add(pipe1)
        pipe_group.add(pipe2)
        pipe_height = randint(-100, 100)
        pipe1 = Pipe(screen_width, screen_height//2+pipe_height, 1)
        pipe2 = Pipe(screen_width, screen_height//2+pipe_height, -1)
        pipe_group.add(pipe1)
        pipe_group.add(pipe2)
        time_now = pygame.time.get_ticks()
        last_pipe = time_now

        for i in range(select_best):
            parent_copy = dcpy(gen[i][0])
            parent_copy.mutate()
            flappy = Bird(100, screen_height//2)
            bird_group.add(flappy)
            gen.append([parent_copy, flappy])

        for _ in range(population_size-len(gen)):
            flappy = Bird(100, screen_height//2)
            bird_group.add(flappy)
            gen.append([Specimen(), flappy])

        # for g in gen:g[0].mutate()

        run = True
        while run:
            clock.tick(fps)

            screen.blit(bg, (0, 0))

            pipe_group.draw(screen)
            bird_group.draw(screen)
            bird_group.update()

            for g in gen:
                g[0].apply_input(g[1])
                if pygame.sprite.spritecollide(g[1], pipe_group, False, False) or g[1].rect.top < 0 or g[1].rect.bottom > 768:
                    g[1].flying = False

            if sum(g[1].flying for g in gen) == 0:
                gen.sort(key=lambda x: -x[1].lifespan)

                bird_group.empty()

                for i in range(population_size//2):
                    flappy = Bird(100, screen_height//2)
                    bird_group.add(flappy)
                    gen[i][1] = flappy

                gen = gen[:population_size//2]

                reset_game()

                break

            screen.blit(ground_img, (ground_scroll, 768))

            if game_over == False:

                pipe_height = randint(-100, 100)

                time_now = pygame.time.get_ticks()
                if time_now-last_pipe > pipe_frequency:
                    pipe1 = Pipe(screen_width, screen_height//2+pipe_height, 1)
                    pipe2 = Pipe(screen_width, screen_height //
                                 2+pipe_height, -1)
                    pipe_group.add(pipe1)
                    pipe_group.add(pipe2)
                    last_pipe = time_now

                pipe_group.update()
                ground_scroll -= scroll_speed
                if abs(ground_scroll) > 35:
                    ground_scroll = 0

            # if len(pipe_group)>0:
            #     if bird_group.sprites()[0].rect.left>pipe_group.sprites()[0].rect.left and bird_group.sprites()[0].rect.right<pipe_group.sprites()[0].rect.right:
            #         pass_pipe=True
            #     if pass_pipe==True:
            #         if bird_group.sprites()[0].rect.left>pipe_group.sprites()[0].rect.right:
            #             pass_pipe=False

            for event in pygame.event.get():
                if event.type == QUIT:
                    run = False

            key = pygame.key.get_pressed()
            if key[K_ESCAPE]:
                gen.sort(key=lambda x: -x[1].lifespan)

                game_over = True
                break

            pygame.display.update()

    gen.sort(key=lambda x: -x[1].lifespan)
    best = gen[0][0]
    with open('clappy.txt', 'wb') as file:
        print('Saved Best Model')
        pickle.dump(best, file)

    pygame.quit()
    print('Learning....')

# Block where user plays the game
elif args.action == 'play':

    def reset_game():
        global score
        pipe_group.empty()
        for bird in bird_group:
            bird.rect.x = 100
            bird.rect.y = int(screen_height / 2)
        score = 0
        return score


    bird_group = pygame.sprite.Group()
    pipe_group = pygame.sprite.Group()

    flappy = Bird(100, screen_height // 2)
    bird_group.add(flappy)

    button = Button(screen_width // 2 - 50, screen_height // 2 - 100, button_img)

    screen = pygame.display.set_mode((screen_width, screen_height))
    pygame.display.set_caption('Flappy Bird')

    run = True
    while run:
        clock.tick(fps)

        screen.blit(bg, (0, 0))

        pipe_group.draw(screen)
        bird_group.draw(screen)
        bird_group.update()

        for bird in bird_group:
            if pygame.key.get_pressed()[K_SPACE]:
                bird.jump = 1
            else:
                bird.jump = 0

        screen.blit(ground_img, (ground_scroll, 768))

        if pygame.sprite.groupcollide(bird_group, pipe_group, False, False) or flappy.rect.top < 0:
            game_over = True

        if game_over == False:

            pipe_height = randint(-100, 100)

            time_now = pygame.time.get_ticks()
            if time_now - last_pipe > pipe_frequency:
                pipe1 = Pipe(screen_width, screen_height // 2 + pipe_height, 1)
                pipe2 = Pipe(screen_width, screen_height // 2 + pipe_height, -1)
                pipe_group.add(pipe1)
                pipe_group.add(pipe2)
                last_pipe = time_now

            pipe_group.update()
            ground_scroll -= scroll_speed
            if abs(ground_scroll) > 35:
                ground_scroll = 0

        if flappy.rect.bottom > 768:
            game_over = True
            flappy.flying = False

        if len(pipe_group) > 0:
            if bird_group.sprites()[0].rect.left > pipe_group.sprites()[0].rect.left and bird_group.sprites()[
                0].rect.right < pipe_group.sprites()[0].rect.right:
                pass_pipe = True
            if pass_pipe == True:
                if bird_group.sprites()[0].rect.left > pipe_group.sprites()[0].rect.right:
                    score += 1
                    pass_pipe = False

        draw_text(str(score), FONT, WHITE, screen_width // 2, 50)

        if game_over == True:
            if button.draw() == True:
                reset_game()
                game_over = False
                flappy.flying = True

        for event in pygame.event.get():
            if event.type == QUIT:
                run = False

        key = pygame.key.get_pressed()
        if key[K_ESCAPE]:
            run = False

        pygame.display.update()

    pygame.quit()

# Block where trained model plays the game
elif args.action == 'showtime':
    print('Starting bot....')

else:
    print('Error')
    exit(0)
