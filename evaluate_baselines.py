from evac_env import TwoExitEvacEnv
import numpy as np
env = TwoExitEvacEnv(pLeft=0.7,pRight=0.3)


episodeRewards= []
episodeSteps = []
for i in range(100000):
    obs,info = env.reset()
    terminated= False
    truncated = False
    totalReward = 0
    while (terminated == False and truncated == False):
        if (obs[1]+1)/env.pLeft <= (obs[2]+1)/env.pRight:
            action = 0
        else:
            action = 1
        
        obs,reward,terminated,truncated,info = env.step(action)
        totalReward += reward

    episodeRewards.append(totalReward)
    episodeSteps.append(env.current_step)
    print(i)

print(f"alternate Average reward: {np.mean(episodeRewards)}, Average steps: {np.mean(episodeSteps)}")

