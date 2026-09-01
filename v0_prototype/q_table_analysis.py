import numpy as np

q_table= np.load("q_tableV0.4.3.npy")
print(q_table[(5,0,0)])
print(q_table[(4,1,0)])
print(q_table[(4,0,1)])
print(q_table[(3,1,1)])
print(q_table[(2,2,1)])
print(q_table[(2,1,2)])
guidance_zone = {
           (4,1), (4,2), (4,3), (4,4), (4,5),
           (4,6), (4,7), (4,8), (4,9)
        }
print([(*cell,0) for cell in guidance_zone])
