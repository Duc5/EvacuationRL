from pathlib import Path

from scenarios.building_v2 import BuildingV2Scenario
from envs.evacuation_env import EvacuationEnv


INITIAL_COUNTS = {"A": 40, "B": 20, "C": 20, "D": 20}

INCIDENT = "C_exit"   # or "C_exit"
SURGE_TIME = 10.0
SURGE_ROOM = "C"
SURGE_COUNT = 40

PRE_POLICY = "CCA"
POST_POLICY = "CJ3A"      # good response for B_connector

CONTROL_INTERVAL = 10.0
MAX_TIME = 180.0
SEED = 0

TRAJECTORY_PATH = Path(
    f"trajectories/debug_{INCIDENT}.sqlite"
)


def short_policy(policy):
    return "".join(policy[j] for j in ("J1", "J2", "J3"))


def main():
    scenario = BuildingV2Scenario(
        room_counts=dict(INITIAL_COUNTS),
        dynamic_events=[{
            "time": SURGE_TIME,
            "room": SURGE_ROOM,
            "count": SURGE_COUNT,
            "incident": INCIDENT,
        }],
    )

    env = EvacuationEnv(
        scenario=scenario,
        control_interval=CONTROL_INTERVAL,
        max_time=MAX_TIME,
        record=True,
        trajectory_path=TRAJECTORY_PATH,
    )

    labels = [
        short_policy(policy)
        for policy in env.guidance_plans
    ]

    pre_action = labels.index(PRE_POLICY)
    post_action = labels.index(POST_POLICY)

    obs, info = env.reset(seed=SEED)

    print(f"Recording to: {TRAJECTORY_PATH}")
    print(f"Incident: {INCIDENT}")
    print(f"Pre-policy:  {PRE_POLICY}")
    print(f"Post-policy: {POST_POLICY}")
    print()

    terminated = False
    truncated = False
    step = 0

    while not terminated and not truncated:
        action = pre_action if step == 0 else post_action

        obs, reward, terminated, truncated, info = env.step(action)

        print(
            f"t={info['elapsed_time']:6.2f} | "
            f"remaining={info['remaining_agents']:3d} | "
            f"incident={env.backend.active_incident} | "
            f"policy={PRE_POLICY if step == 0 else POST_POLICY}"
        )

        step += 1

    print()
    print(f"Finished at t={info['elapsed_time']:.2f}s")
    print(f"Remaining: {info['remaining_agents']}")

    env.close()


if __name__ == "__main__":
    main()