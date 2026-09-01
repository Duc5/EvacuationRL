from evac_env import TwoExitEvacEnv
import numpy as np
env = TwoExitEvacEnv()


episodeRewards= []
episodeSteps = []
def expected_cleartime(population):
    total = 0
    for q in range(1,population+1):
        total += 1/env.escapeProbability(q)
    return total

for i in range(100000):
    obs,info = env.reset()
    terminated= False
    truncated = False
    totalReward = 0
    while (terminated == False and truncated == False):
        # Static Etime
        # if (obs[1]+1)/env.escapeProbability(obs[1]+1) <= (obs[2]+1/env.escapeProbability(obs[2]+1)):
        #     action = 0
        # else:
        #     action = 1


        # # Dynamic Etime
        left_time = expected_cleartime(obs[1]+1)
        right_time = expected_cleartime(obs[2]+1)
        if left_time <= right_time:
            action = 0
        else:
            action = 1


        obs,reward,terminated,truncated,info = env.step(action)
        totalReward += reward

    episodeRewards.append(totalReward)
    episodeSteps.append(env.current_step)
    print(i)

print(f"alternate Average reward: {np.mean(episodeRewards)}, Average steps: {np.mean(episodeSteps)}")


