from statistics import mean

from scenarios.building_v2 import BuildingV2Scenario
from envs.evacuation_env import EvacuationEnv


INITIAL_COUNTS = {
    "A": 40,
    "B": 20,
    "C": 20,
    "D": 20,
}

INCIDENTS = [
    "C_exit",
    "B_route_08",
    "bottom_loop_left",
]

SEEDS = range(5, 15)

SURGE_ROOM = "C"
SURGE_TIME = 10.0
SURGE_COUNT = 40

CONTROL_INTERVAL = 10.0
MAX_TIME = 180.0

POLICY = "ACA"


def short_policy(policy):
    return "".join(policy[j] for j in ["J1", "J2", "J3"])


def make_env(incident):
    scenario = BuildingV2Scenario(
        room_counts=dict(INITIAL_COUNTS)
    )

    return EvacuationEnv(
        scenario=scenario,
        control_interval=CONTROL_INTERVAL,
        max_time=MAX_TIME,
        dynamic_surge_rooms=[SURGE_ROOM],
        dynamic_surge_times=[SURGE_TIME],
        dynamic_surge_count=SURGE_COUNT,
        incidents=INCIDENTS,
        forced_incident=incident,
    )


def get_action():
    env = make_env("C_exit")

    labels = [
        short_policy(policy)
        for policy in env.guidance_plans
    ]

    env.close()

    return labels.index(POLICY)


def run_episode(incident, seed, action):
    env = make_env(incident)

    try:
        obs, info = env.reset(seed=seed)

        terminated = False
        truncated = False
        episode_reward = 0.0

        while not terminated and not truncated:
            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            episode_reward += reward

        return {
            "incident": incident,
            "seed": seed,
            "time": info["elapsed_time"],
            "reward": episode_reward,
            "remaining": info["remaining_agents"],
        }

    finally:
        env.close()


def main():
    action = get_action()

    print(f"Testing fixed policy {POLICY}")
    print(f"Action index: {action}\n")

    rows = []

    for incident in INCIDENTS:
        for seed in SEEDS:
            row = run_episode(
                incident=incident,
                seed=seed,
                action=action,
            )

            rows.append(row)

            print(
                f"{incident:<18} | "
                f"seed={seed:2} | "
                f"time={row['time']:6.2f} | "
                f"reward={row['reward']:7.4f} | "
                f"remaining={row['remaining']}"
            )

    print("\n===== ACA FIXED VALIDATION =====")

    for incident in INCIDENTS:
        incident_rows = [
            row for row in rows
            if row["incident"] == incident
        ]

        print(
            f"{incident:<18} | "
            f"mean={mean(row['time'] for row in incident_rows):6.2f}s | "
            f"min={min(row['time'] for row in incident_rows):6.2f}s | "
            f"max={max(row['time'] for row in incident_rows):6.2f}s"
        )

    print(
        f"\nOverall ACA mean: "
        f"{mean(row['time'] for row in rows):.2f}s"
    )


if __name__ == "__main__":
    main()