import numpy as np

class NeuralNetwork():
    def __init__(self, input_size, output_size):
        self.input_size = input_size
        self.output_size = output_size
        self.connections = {}
        self.nodes = set(range(input_size + output_size))
        self.next_node_id = input_size + output_size
        self.biases = np.random.randn(output_size)

    def addNode(self, input_node, output_node):
        self.nodes.add(self.next_node_id)

        original_weight = self.connections.pop((input_node, output_node))
        self.addConnection(input_node, self.next_node_id, original_weight)
        self.addConnection(self.next_node_id, output_node, 1.0)  # New connection weight

        self.next_node_id += 1

    def addConnection(self, input_node, output_node, weight):
        if (input_node, output_node) not in self.connections:
            self.connections[(input_node, output_node)] = weight
        else:
             print(f"Connection between {input_node} and {output_node} already exists.")
    
    def forward(self, inputs):
        assert len(inputs) == self.input_size
        node_values = {i: inputs[i] for i in range(self.input_size)}
        #print("Initial node values:", node_values)

        for node in self.nodes:
            if node not in node_values:
                node_values[node] = 1

        for node in sorted(self.nodes):
            if node >= self.input_size:
                inputs_sum = sum(
                    node_values[in_node] * self.connections[(in_node, node)]
                    for in_node in self.nodes
                    if (in_node, node) in self.connections and in_node in node_values)
                
                #print(f"Node {node} receives inputs_sum = {inputs_sum}")

                node_values[node] = np.tanh(inputs_sum)
                #print(f"Node {node} value after activation: {node_values[node]}")

        #print("Final node values:", node_values)
 

        print([node_values[node] for node in range(len(node_values) - self.output_size, len(node_values))])     
        return [node_values[node] for node in range(len(node_values) - self.output_size, len(node_values))]
