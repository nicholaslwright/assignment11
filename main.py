print("Hello World")

#Task 1
from neural import NeuralNet

# each row is an (input, output) tuple
xor_data = [
 # input output corresponding example
 ([0.0, 0.0], [0.0]), #[0, 0] => 0
 ([0.0, 1.0], [1.0]), #[0, 1] => 1
 ([1.0, 0.0], [1.0]), #[1, 1] => 1
 ([1.0, 1.0], [0.0]) #[1, 0] => 0
]
nn = NeuralNet(2, 5, 1)
nn.train(xor_data)