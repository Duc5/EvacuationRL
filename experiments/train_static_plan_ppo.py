import torch

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

from envs.static_plan_env import StaticPlanEnv
from scenarios.building_v2 import BuildingV2Scenario


torch.set_num_threads(1)
torch.set_num_interop_threads(1)


POPULATION_SCENARIOS = {
    "balanced": {"A": 25, "B": 25, "C": 25, "D": 25},
    "A_extreme": {"A": 49, "B": 17, "C": 17, "D": 17},
    "B_extreme": {"A": 19, "B": 42, "C": 20, "D": 19},
    "C_extreme": {"A": 19, "B": 20, "C": 42, "D": 19},
    "D_extreme": {"A": 22, "B": 21, "C": 21, "D": 36},
}


scenario = BuildingV2Scenario(
    room_counts=POPULATION_SCENARIOS["balanced"]
)

env = StaticPlanEnv(
    scenario=scenario,
    population_scenarios=POPULATION_SCENARIOS,
    results_path="results/static_plan_seed0.json",
)

env = Monitor(env)

model = PPO(
    "MlpPolicy",
    env,
    seed=0,
    verbose=1,
    device="cpu",
    n_steps=256,
    batch_size=64,
)

model.learn(total_timesteps=20_000)

model.save("models/ppo_static_plan")