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

print(nn.evaluate([0.0, 1.0]))

for triple in nn.test_with_expected(xor_data):
 print(triple)

#Task 2
print("Task 2:")
nn = NeuralNet(2, 1, 1)
nn.train(xor_data)

for triple in nn.test_with_expected(xor_data):
 print(triple)

# Task 3

import pandas as pd

df = pd.read_csv('wine.data')

normalized_df = (df-df.min())/(df.max()-df.min())

print(normalized_df)

new_list = []

for row in normalized_df.iterrows():
 attributes = list(row[1:])
 answer = [row[0]]
 new_row = (attributes, answer)
 new_list.append(new_row)

print(new_list)

# Put new list inside the neural net
nn = NeuralNet(13, 5, 1)
nn.train(new_list)

#print(nn.evaluate([0.0, 1.0]))

for triple in nn.test_with_expected(new_list):
 print(triple)




