from grid_evac_env import GridEvacEnv
import numpy as np
env = GridEvacEnv()



obs,info = env.reset()
terminated= False
truncated = False
totalReward = 0
episodeRewards= []
episodeSteps = []
for i in range(10000000):
    while (terminated == False and truncated == False):

        rand = np.random.randint(0,4)
        action = rand

        obs,reward,terminated,truncated,info = env.step(action)
        totalReward += reward
    episodeRewards.append(totalReward)
    episodeSteps.append(env.current_step)
    print(i)        
print(f"Random Average reward: {np.mean(episodeRewards)}, Average steps: {np.mean(episodeSteps)}")