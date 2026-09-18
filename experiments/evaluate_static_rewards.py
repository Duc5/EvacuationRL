import csv
from pathlib import Path

from scenarios.building_v2 import BuildingV2Scenario
from envs.evacuation_env import EvacuationEnv


POPULATION_SCENARIOS = {
    "balanced": {"A": 25, "B": 25, "C": 25, "D": 25},
    "A_extreme": {"A": 49, "B": 17, "C": 17, "D": 17},
    "B_extreme": {"A": 19, "B": 42, "C": 20, "D": 19},
    "C_extreme": {"A": 19, "B": 20, "C": 42, "D": 19},
    "D_extreme": {"A": 22, "B": 21, "C": 21, "D": 36},
}

SEED = 0
GAMMA = 0.99
CONTROL_INTERVAL = 5.0
MAX_TIME = 180.0

OUTPUT_PATH = Path("results/static_policy_rewards_seed0.csv")


def short_policy(policy):
    return "".join(policy[junction] for junction in ["J1", "J2", "J3"])


def evaluate_policy(room_counts, action):
    scenario = BuildingV2Scenario(room_counts=dict(room_counts))

    env = EvacuationEnv(
        scenario=scenario,
        control_interval=CONTROL_INTERVAL,
        max_time=MAX_TIME,
    )

    obs, info = env.reset(seed=SEED)

    terminated = False
    truncated = False

    episode_reward = 0.0
    discounted_return = 0.0
    discount = 1.0
    steps = 0

    while not terminated and not truncated:
        obs, reward, terminated, truncated, info = env.step(action)

        episode_reward += reward
        discounted_return += discount * reward

        discount *= GAMMA
        steps += 1

    result = {
        "elapsed_time": info["elapsed_time"],
        "remaining_agents": info["remaining_agents"],
        "episode_reward": episode_reward,
        "discounted_return": discounted_return,
        "steps": steps,
    }

    env.close()
    return result


def main():
    template = BuildingV2Scenario(
        room_counts=POPULATION_SCENARIOS["balanced"]
    )

    template_env = EvacuationEnv(
        scenario=template,
        control_interval=CONTROL_INTERVAL,
        max_time=MAX_TIME,
    )

    policies = template_env.guidance_plans
    template_env.close()

    print(f"Policies: {len(policies)}")
    print(f"Simulations: {len(policies) * len(POPULATION_SCENARIOS)}")

    rows = []

    for action, policy in enumerate(policies):
        policy_name = short_policy(policy)

        print(f"\n===== {action:02d}: {policy_name} =====")

        for population_name, room_counts in POPULATION_SCENARIOS.items():
            result = evaluate_policy(room_counts, action)

            row = {
                "action": action,
                "policy": policy_name,
                "population": population_name,
                **result,
            }

            rows.append(row)

            print(
                f"{population_name:<10} | "
                f"time={result['elapsed_time']:6.2f} | "
                f"reward={result['episode_reward']:7.4f} | "
                f"discounted={result['discounted_return']:7.4f}"
            )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with open(OUTPUT_PATH, "w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved to {OUTPUT_PATH}")

    print("\n===== MEAN PERFORMANCE ACROSS POPULATIONS =====")

    summaries = []

    for action, policy in enumerate(policies):
        policy_name = short_policy(policy)
        policy_rows = [row for row in rows if row["action"] == action]

        mean_time = sum(row["elapsed_time"] for row in policy_rows) / len(policy_rows)
        mean_reward = sum(row["episode_reward"] for row in policy_rows) / len(policy_rows)
        mean_discounted = sum(row["discounted_return"] for row in policy_rows) / len(policy_rows)

        summaries.append({
            "action": action,
            "policy": policy_name,
            "mean_time": mean_time,
            "mean_reward": mean_reward,
            "mean_discounted": mean_discounted,
        })

    print("\nBest by evacuation time:")
    for result in sorted(summaries, key=lambda x: x["mean_time"])[:10]:
        print(
            f"{result['policy']:<8} "
            f"time={result['mean_time']:6.2f} | "
            f"reward={result['mean_reward']:7.4f} | "
            f"discounted={result['mean_discounted']:7.4f}"
        )

    print("\nBest by undiscounted RL reward:")
    for result in sorted(summaries, key=lambda x: x["mean_reward"], reverse=True)[:10]:
        print(
            f"{result['policy']:<8} "
            f"time={result['mean_time']:6.2f} | "
            f"reward={result['mean_reward']:7.4f} | "
            f"discounted={result['mean_discounted']:7.4f}"
        )

    print("\nBest by PPO discounted return:")
    for result in sorted(summaries, key=lambda x: x["mean_discounted"], reverse=True)[:10]:
        print(
            f"{result['policy']:<8} "
            f"time={result['mean_time']:6.2f} | "
            f"reward={result['mean_reward']:7.4f} | "
            f"discounted={result['mean_discounted']:7.4f}"
        )


if __name__ == "__main__":
    main()