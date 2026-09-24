import random
import numpy as np

from neat.Genome import Genome
from neat.NeuralNetwork import NeuralNetwork



class Population:
    def __init__(self, size, input_size, output_size):
        self.genomes = [Genome(input_size, output_size) for _ in range(size)]

    def evolve(self):
        self.genomes.sort(key=lambda genome: genome.fitness, reverse=True)
        
        survivors = self.genomes[:int(len(self.genomes)//2.5)]

        children: list[Genome] = []
        for _ in range(len(self.genomes) - len(survivors)):
            parent1, parent2 = np.random.choice(survivors, 2)
            if parent2.fitness > parent1.fitness:
                temp = parent1
                parent1 = parent2
                parent2 = temp
            print(parent1.id, parent2.id)
            child = self.crossover(parent1, parent2)
            children.append(child)

        self.genomes = survivors + children

        for genome in self.genomes[1:]:
            genome.mutate()

        for genome in self.genomes:
            genome.fitness = 0

    def evaluate(self):
        pass

    def crossover(self, parent1: Genome, parent2: Genome):
        child = Genome(parent1.nn.input_size, parent1.nn.output_size)
        
        for connection in parent1.nn.connections:
            if connection in parent2.nn.connections and np.random.rand() > 0.5:
                child.nn.connections[connection] = parent2.nn.connections[connection]
            else:
                child.nn.connections[connection] = parent1.nn.connections[connection]
        
        for node in parent1.nn.nodes:
            if node not in child.nn.nodes:
                if node in parent2.nn.nodes and np.random.rand() > 0.5:
                    child.addNode() 
                else:
                    child.addNode()

        return child


'''
I will potentially use this code in the crossover function, because in this one connections and nodes get carried from both parents,
unlike in the one I currently using. This could  be useful if I wanted to increase the amount of noodes and connections.
 
    if connection in parent2.nn.connections and np.random.rand() > 0.5:
                child.nn.connections[connection] = parent2.nn.connections[connection]
            else:
                child.nn.connections[connection] = parent1.nn.connections[connection]
        
        for node in parent1.nn.nodes:
            if node in parent2.nn.nodes and np.random.rand() > 0.5:
                child.nn.addNode() 
            else:
                child.nn.addNode()
        return child

for connection in set(parent1.nn.connections.keys()) | set(parent2.nn.connections.keys()):
            if connection in parent1.nn.connections and connection in parent2.nn.connections:
                # If both parents have the connection, randomly choose which parent to inherit it from
                child.nn.connections[connection] = random.choice([parent1.nn.connections[connection], parent2.nn.connections[connection]])
            elif connection in parent1.nn.connections:
                child.nn.connections[connection] = parent1.nn.connections[connection]
            elif connection in parent2.nn.connections:
                child.nn.connections[connection] = parent2.nn.connections[connection]

        for node in parent1.nn.nodes | parent2.nn.nodes:
            if node in parent1.nn.nodes and parent2.nn.nodes:    #later I am going to add biases and  for each node this is going to decide which parents biase the child is going to inherit like with the connections
                child.addNode()
            elif node in parent1.nn.nodes:
                child.addNode()
            elif node in parent2.nn.nodes:
                child.addNode()

'''