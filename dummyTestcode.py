import torch
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import CheckpointCallback
from stable_baselines3.common.monitor import Monitor

from scenarios.building_v2 import BuildingV2Scenario
from envs.evacuation_env import EvacuationEnv
from pathlib import Path
MODEL_PATH = Path("models/ppo_threeway_incident_v3b.zip")

SEED = 0

INITIAL_COUNTS = {"A": 40,"B": 20,"C": 20,"D": 20,}

SURGE_ROOM = "C"
SURGE_TIME = 10.0
SURGE_COUNT = 40

INCIDENTS = ["C_exit","B_route_08","bottom_loop_left"]

CONTROL_INTERVAL = 10.0
MAX_TIME = 180.0

TOTAL_TIMESTEPS = 8192

def make_env():
    scenario = BuildingV2Scenario(
        room_counts=dict(INITIAL_COUNTS)
    )

    env = EvacuationEnv(
        scenario=scenario,
        control_interval=CONTROL_INTERVAL,  
        max_time=MAX_TIME,
        dynamic_surge_rooms=[SURGE_ROOM],
        dynamic_surge_times=[SURGE_TIME],
        dynamic_surge_count=SURGE_COUNT,
        incidents=INCIDENTS,
    )

    return Monitor(env)

env = make_env()

for episode in range(50):
    obs, info = env.reset()

    assert env.observation_space.contains(obs)
    assert np.all(obs[-3:] == 0)

    action = env.action_space.sample()

    obs, reward, terminated, truncated, info = env.step(action)

    print(
        episode,
        action,
        info["active_incident"],
        obs[-3:],
        env.observation_space.contains(obs),
    )

env.close()