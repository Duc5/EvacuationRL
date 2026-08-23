from evac_env import TwoExitEvacEnv
import numpy as np
env = TwoExitEvacEnv()


episodeRewards= []
episodeSteps = []
for i in range(100000):
    obs,info = env.reset()
    terminated= False
    truncated = False
    totalReward = 0
    while (terminated == False and truncated == False):
        action = env.current_step %2
        
        obs,reward,terminated,truncated,info = env.step(action)
        totalReward += reward

    episodeRewards.append(totalReward)
    episodeSteps.append(env.current_step)
    print(i)

print(f"alternate Average reward: {np.mean(episodeRewards)}, Average steps: {np.mean(episodeSteps)}")

