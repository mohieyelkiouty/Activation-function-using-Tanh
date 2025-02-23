import math
import random

def tanh(x):
    exp_x = math.exp(x)
    exp_neg_x = math.exp(-x)
    return (exp_x - exp_neg_x) / (exp_x + exp_neg_x)

def dot_product(a, b):
    return sum(x * y for x, y in zip(a, b))
x = [0.05, 0.1]  
w1 = [[random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)],
      [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]] 
b1 = [0.5, 0.5] 
w2 = [[random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)], 
      [random.uniform(-0.5, 0.5), random.uniform(-0.5, 0.5)]]
b2 = [0.7, 0.7]

hidden_input = [dot_product(x, col) + b for col, b in zip(zip(*w1), b1)]
hidden_output = [tanh(h) for h in hidden_input]
output_input = [dot_product(hidden_output, col) + b for col, b in zip(zip(*w2), b2)]
output = [tanh(o) for o in output_input]

print("Network Output:", output)
