import pickle
from neat.Genome import Genome
import pong
from pong import Game
import neat 
from neat.NeuralNetwork import NeuralNetwork
from neat.Population import Population
import pygame
import numpy as np

class PongGame:
    def __init__(self, window, width, height):
        self.game = Game(window, width, height)
        self.left_paddle = self.game.left_paddle
        self.right_paddle = self.game.right_paddle
        self.ball = self.game.ball

    def testAi(self, genome: Genome):
        run = True
        clock = pygame.time.Clock()
        while run:
            clock.tick(60.0)
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                self.game.move_paddle(left=True, up=True)
            if keys[pygame.K_s]:
                self.game.move_paddle(left=True, up=False)

            output = genome.nn.forward([
                        self.right_paddle.y, self.ball.y, abs(self.right_paddle.x - self.ball.x), self.ball.x_vel, self.ball.y_vel
                    ])
            decision = output.index(max(output))

            if decision == 0:
                pass
            elif decision == 1:
                self.game.move_paddle(left=False, up=True)
            else:
                self.game.move_paddle(left=False, up=False)

            game_info = self.game.loop()
            self.game.draw()
            pygame.display.update()

        pygame.quit()

    def playPong(self):
        run = True
        clock = pygame.time.Clock()
        while run:
            clock.tick(60.0)
            game_info = self.game.loop()
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    run = False
                    break

            keys = pygame.key.get_pressed()
            if keys[pygame.K_w]:
                self.game.move_paddle(left=True, up=True)
            if keys[pygame.K_s]:
                self.game.move_paddle(left=True, up=False)

            if keys[pygame.K_UP]:
                self.game.move_paddle(left=False, up=True)
            if keys[pygame.K_DOWN]:
                self.game.move_paddle(left=False, up=False)

            self.game.draw(True, True)
            pygame.display.update()

            if game_info.left_score >= 1 or game_info.right_score >= 1 or game_info.left_hits > 50:
                        break
        
        pygame.quit()

    def trainAi(self, population: Population, generations, trained_genome: Genome = 0):
        for generation in range(generations):
            print(generation)

            if trained_genome == 0:
                trainer = population.genomes[0]
            else:
                trainer = trained_genome
                trainer.id = "pre trained ai"

            for genome in population.genomes[1:]:

                run = True
                while run:
                    for event in pygame.event.get():
                        if event.type == pygame.QUIT:
                            quit()

                    output1 = trainer.nn.forward([
                        self.left_paddle.y, self.ball.y, abs(self.left_paddle.x - self.ball.x), self.ball.x_vel, self.ball.y_vel
                    ])
                    decision1 = output1.index(max(output1))

                    if decision1 == 0:
                        pass
                    elif decision1 == 1:
                        self.game.move_paddle(left=True, up=True)
                    else:
                        self.game.move_paddle(left=True, up=False)

                    output2 = genome.nn.forward([
                        self.right_paddle.y, self.ball.y, abs(self.right_paddle.x - self.ball.x), self.ball.x_vel, self.ball.y_vel
                    ])
                    decision2 = output2.index(max(output2))

                    if decision2 == 0:
                        pass
                    elif decision2 == 1:
                        self.game.move_paddle(left=False, up=True)
                    else:
                        self.game.move_paddle(left=False, up=False)

                    game_info = self.game.loop()

                    self.game.draw(draw_score=True, draw_hits=True)
                    pygame.display.update()

                    if game_info.left_score >= 1 or game_info.right_score >= 1 or game_info.left_hits > 30:
                        trainer.fitness += 45 * game_info.left_score + game_info.left_hits
                        genome.fitness += 45 * game_info.right_score + game_info.right_hits
                        run = False
                        self.game.reset()

            trainer.fitness = trainer.fitness / (len(population.genomes) - 1)
            population.genomes[0] = trainer
            population.evolve()
            best_genome = population.genomes[0]
            with open("best.pickle", "wb") as f:
                pickle.dump(best_genome, f)

#Opens a Player vs Player game of Pong
def playPong():
    width, height = 700, 500
    window = pygame.display.set_mode((width, height))

    game = PongGame(window, width, height)
    game.playPong()

#Opens a Player vs AI game of Pong and loads a model from the "best.pickle" file
def testAi():
    width, height = 700, 500
    window = pygame.display.set_mode((width, height))

    with open("best.pickle", "rb") as f:
        winner = pickle.load(f)

    game = PongGame(window, width, height)
    game.testAi(winner)

#Starts the training process and saves the model in the "best.pickle" file
def trainAi(trained_genome: any = 0):
    width, height = 700, 500
    window = pygame.display.set_mode((width, height))
    game = PongGame(window, width, height)

    if trained_genome != 0:
        with open(trained_genome, "rb") as f:
            genome = pickle.load(f)
    else:
        genome = 0

    population = Population(10, 5, 3)
    game.trainAi(population, 150, genome)

#Displays all connections, nodes and biases of the model from the "best.pickle" file
def readAi():
    with open("best.pickle", "rb") as f:
        genome: Genome = pickle.load(f)

    print("connections")
    for c in genome.nn.connections:
        print(c, genome.nn.connections[c])

    print("")

    print("nodes")
    for n in genome.nn.nodes:
        print(n)

    print("")

    print("biases")
    for b in genome.nn.biases:
        print(b)
    
    print(genome.id)
 
"""
def editAi():
    with open("best.pickle", "rb") as f:
        genome = pickle.load(f)
    
    with open("best.pickle", "wb") as f:
        pickle.dump(genome, f)       
"""

#uncomment the function you want to run here
if __name__ == "__main__":
    testAi()
    #trainAi()
    #readAi()
    #playPong()