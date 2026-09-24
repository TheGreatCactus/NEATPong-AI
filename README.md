Pong AI with NEAT
A simple Python project that uses the NEAT (NeuroEvolution of Augmenting Topologies) algorithm to train an artificial intelligence to play the classic game of Pong.

Features
NEAT Training: The program trains neural networks over multiple generations to learn optimal Pong strategies.

Model Saving: The highest-performing neural network (the best genome) is automatically saved to a best.pickle file after training is complete.

Player vs. AI Mode: You can load the pre-trained best.pickle model and play a real-time game of Pong against the AI.

Requirements
To run this project, you will need Python 3 installed along with the Pygame library

How to Run
Go into the "main.py" file uncomment the function that you want to run and run the "main.py" file

## Acknowledgements & Credits
* The base Pong game implementation and window setup were adapted from the tutorial by [Tech With Tim](https://github.com/techwithtim/NEAT-Pong-Python).
* The NEAT integration, model training logic, serialization (`best.pickle`), and Player vs AI implementation were developed as part of this portfolio project.

The AI implementation and training logic written by me are free to use and modify for educational purposes.
