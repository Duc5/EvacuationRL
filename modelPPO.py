from stable_baselines3 import PPO
from scenarios.two_exit_v1 import TwoExitV1Scenario
from envs.evacuation_env import EvacuationEnv


scenario = TwoExitV1Scenario()

env = EvacuationEnv(
    scenario=scenario,
    record=False,
)

model = PPO(
    "MlpPolicy",
    env,
    verbose=1,
    seed=0
)

model.learn(
    total_timesteps=10_000
)

model.save("models/ppo_v1")