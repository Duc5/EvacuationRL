import numpy as np

q_table= np.load("q_tableV0.4.3.npy")
print(q_table[(5,0,0)])
print(q_table[(4,1,0)])
print(q_table[(4,0,1)])
print(q_table[(3,1,1)])
print(q_table[(2,2,1)])
print(q_table[(2,1,2)])