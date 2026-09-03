from v0_prototype.grid_evac_env import GridEvacEnv
import numpy as np

starts_3 = [
    (1,1), (1,2), (1,3)
]

starts_4 = [
    (1,1), (1,2), (1,3), (1,4)
]

starts_9 = [
    (1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7), (1,8), (1,9)
]
starts_18 = [
    (1,1), (1,2), (1,3), (1,4), (1,5), (1,6), (1,7), (1,8), (1,9), 
    (2,1), (2,2), (2,3), (2,4), (2,5), (2,6), (2,7), (2,8), (2,9)
]

env = GridEvacEnv(start_positions=starts_18)



obs,info = env.reset()
terminated= False
truncated = False
totalReward = 0
episodeRewards= []
episodeSteps = []
escapedLeft = 0
escapedRight = 0
truncateCount = 0
for i in range(10000):
    obs,info = env.reset()
    terminated= False
    truncated = False
    totalReward = 0
    while (terminated == False and truncated == False):

        action = env.current_step % 2
        obs,reward,terminated,truncated,info = env.step(action)
        totalReward += reward
        print(
            "step:", env.current_step,
            "positions:", env.pedestrian_positions,
            "guidance:" , env.guidance_states
        )
        escapedLeft += info["escaped_left"]
        escapedRight += info["escaped_right"]
        print(obs, sum(obs), len(env.pedestrian_positions))
    if truncated:
        truncateCount +=1
    else:
        episodeRewards.append(totalReward)
        episodeSteps.append(env.current_step)

    

stepNumbers= set(episodeSteps)
stepDict ={}
for i in stepNumbers:
    stepDict[f"{i}"] = f"{episodeSteps.count(i)/len(episodeSteps)}"
print(f"""BFS Crowd 18, spatial roudting, Average steps: {np.mean(episodeSteps)}, 
        {stepDict}
        Escaped left: {escapedLeft}, Escaped right: {escapedRight}
        Truncated: {truncateCount}
        """) 
