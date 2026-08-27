from grid_evac_env import GridEvacEnv
import numpy as np

starts_3 = [
    (1,1), (1,2), (1,3)
]

starts_4 = [
    (1,1), (1,2), (1,3), (1,4)
]

starts_5 = [
    (1,1), (1,2), (1,3), (1,4), (1,5)
]


env = GridEvacEnv(start_positions=starts_5)



obs,info = env.reset()
terminated= False
truncated = False
totalReward = 0
episodeRewards= []
episodeSteps = []
escapedLeft = 0
escapedRight = 0
for i in range(10000):
    obs,info = env.reset()
    terminated= False
    truncated = False
    totalReward = 0
    while (terminated == False and truncated == False):

        obs,reward,terminated,truncated,info = env.step(0)
        totalReward += reward
        print(
            "step:", env.current_step,
            "positions:", env.pedestrian_positions,
            "guidance:" , env.guidance_states
        )
        escapedLeft += info["escaped_left"]
        escapedRight += info["escaped_right"]
    episodeRewards.append(totalReward)
    episodeSteps.append(env.current_step)

    

stepNumbers= set(episodeSteps)
stepDict ={}
for i in stepNumbers:
    stepDict[f"{i}"] = f"{episodeSteps.count(i)/len(episodeSteps)}"
print(f"""BFS Crowd 5, Average steps: {np.mean(episodeSteps)}, 
        {stepDict}
        Escaped left: {escapedLeft}, Escaped right: {escapedRight}
        """)