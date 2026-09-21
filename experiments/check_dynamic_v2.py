from scenarios.building_v2 import BuildingV2Scenario
from envs.evacuation_env import EvacuationEnv


scenario = BuildingV2Scenario(
    room_counts={"A": 15, "B": 15, "C": 15, "D": 15},
    dynamic_events=[
        {"time": 20.0, "room": "A", "count": 40}
    ],
)

env = EvacuationEnv(
    scenario=scenario,
    control_interval=10.0,
    max_time=180.0,
    record=True,
    trajectory_path="check_crowd_surge.sqlite"
)

obs, info = env.reset(seed=0)

cca = {"J1": "C", "J2": "C", "J3": "A"}
cca_action = next(
    i for i, plan in enumerate(env.guidance_plans)
    if plan == cca
)

print(
    f"t={env.backend.elapsed_time:.2f} | "
    f"active={env.backend.remaining_agents} | "
    f"spawned={env.backend.initial_population}"
)

terminated = False
truncated = False

while not terminated and not truncated:
    obs, reward, terminated, truncated, info = env.step(cca_action)

    print(
        f"t={info['elapsed_time']:.2f} | "
        f"active={info['remaining_agents']} | "
        f"spawned={env.backend.initial_population} | "
        f"reward={reward:.4f}"
    )

env.close()