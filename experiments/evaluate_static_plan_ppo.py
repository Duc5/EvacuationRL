from stable_baselines3 import PPO

from envs.static_plan_env import StaticPlanEnv
from scenarios.building_v2 import BuildingV2Scenario


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

model = PPO.load(
    "models/ppo_static_plan",
    device="cpu",
)

for population_name in POPULATION_SCENARIOS:
    obs, _ = env.reset(
        options={"population_name": population_name}
    )

    action, _ = model.predict(
        obs,
        deterministic=True,
    )

    _, reward, _, _, info = env.step(action)

    print(f"\n{population_name}")
    print("Policy:", info["policy"])
    print("Time:", info["elapsed_time"])
    print("Reward:", reward)