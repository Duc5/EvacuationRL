from grid_evac_env import GridEvacEnv
import numpy as np
env = GridEvacEnv()



obs,info = env.reset()
terminated= False
truncated = False
totalReward = 0
episodeRewards= []
episodeSteps = []
for i in range(100000):
    obs,info = env.reset()
    terminated= False
    truncated = False
    totalReward = 0
    while (terminated == False and truncated == False):

        obs,reward,terminated,truncated,info = env.step(0)
        totalReward += reward
        print(
            "step:", env.current_step,
            "positions:", env.pedestrian_positions
        )
    episodeRewards.append(totalReward)
    episodeSteps.append(env.current_step)
    print(i)
        
print(f"BFS Average reward: {np.mean(episodeRewards)}, Average steps: {np.mean(episodeSteps)}, No of 8,9,10 steps episodes {episodeSteps.count(8)} {episodeSteps.count(9)} {episodeSteps.count(10)}")