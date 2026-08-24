import numpy as np

q_table= np.load("q_tableV0.1.npy")
print(q_table[15, 1, 0])
print(q_table[10, 1, 0])
print(q_table[5, 1, 0])
print(q_table[1, 1, 0])