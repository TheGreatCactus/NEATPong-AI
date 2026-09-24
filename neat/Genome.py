import itertools
import random
import numpy as np

from neat.NeuralNetwork import NeuralNetwork


class Genome():
    def __init__(self, input_size, output_size, id = "genome"):
        self.nn = NeuralNetwork(input_size, output_size)
        self.fitness = 0
        self.addConnection()
        self.id = id

    def mutate(self):
        self.mutateWeights()
        if np.random.randn() < 0.05:
            self.addNode()
        if np.random.randn() < 0.1:
            self.addConnection()
    
    def mutateWeights(self):
        for key in self.nn.connections:
            if np.random.rand() < 0.9:
                self.nn.connections[key] += np.random.rand() * 0.5
            else:
                self.nn.connections[key] = np.random.rand()

    def addNode(self):
        connection = random.choice(list(self.nn.connections.keys()))
        self.nn.addNode(connection[0], connection[1])

    def addConnection(self):
        niggers = list(itertools.combinations(self.nn.nodes, 2))
        possible_connections = [n for n in niggers if n not in self.nn.connections 
                                and (n[0] not in range(self.nn.input_size, self.nn.input_size + self.nn.output_size) #those two lines are making sure that there are no connections made between two output nodes
                                 or n[1] not in range(self.nn.input_size, self.nn.input_size + self.nn.output_size))
                                 and (n[0] not in range(self.nn.input_size) #same thing but for inputs
                                 or n[1] not in range(self.nn.input_size))]
        '''
        possible_connections = [(i, j) for i in self.nn.nodes for j in self.nn.nodes 
                                if i != j and (i, j)not in self.nn.connections and (j, i) not in possible_connections and (j, i) not in self.nn.connections 
                                and (i not in range(self.nn.input_size, self.nn.input_size + self.nn.output_size) #those two last lines are making sure that there are no connections made between two output nodes
                                 or j not in range(self.nn.input_size, self.nn.input_size + self.nn.output_size))] 
        print(possible_connections)
        '''
        if possible_connections:
            input_node, output_node = possible_connections[np.random.randint(len(possible_connections))]
            weight = np.random.randn()  # Random weight for the new connection
            self.nn.addConnection(input_node, output_node, weight)
            

    def crossover(self, other_genome):
        pass
