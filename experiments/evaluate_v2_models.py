import json
import numpy as np
import torch

from stable_baselines3 import PPO

from envs.evacuation_env import EvacuationEnv
from envs.static_plan_env import policy_key
from scenarios.building_v2 import BuildingV2Scenario


POPULATION_SCENARIOS = {
    "balanced": {"A": 25, "B": 25, "C": 25, "D": 25},
    "A_extreme": {"A": 49, "B": 17, "C": 17, "D": 17},
    "B_extreme": {"A": 19, "B": 42, "C": 20, "D": 19},
    "C_extreme": {"A": 19, "B": 20, "C": 42, "D": 19},
    "D_extreme": {"A": 22, "B": 21, "C": 21, "D": 36},
}

MODELS = {
    "PPO_5k": "models/ppo_v2_5k_success",
    "PPO_10k": "models/ppo_v2_5k10s",
}

CCA = {"J1": "C", "J2": "C", "J3": "A"}

SEED = 0
CONTROL_INTERVAL = 5.0
MAX_TIME = 180.0


def short_policy(policy):
    return "".join(policy[junction] for junction in ["J1", "J2", "J3"])


def get_action_probability(model, observation, action):
    obs_tensor, _ = model.policy.obs_to_tensor(observation)

    with torch.no_grad():
        distribution = model.policy.get_distribution(obs_tensor)
        probabilities = distribution.distribution.probs.cpu().numpy()[0]

    return float(probabilities[int(action)])


def evaluate_model(model, room_counts, seed=0, record_sequence=False):
    scenario = BuildingV2Scenario(room_counts=dict(room_counts))

    env = EvacuationEnv(
        scenario=scenario,
        control_interval=CONTROL_INTERVAL,
        max_time=MAX_TIME,
    )

    observation, info = env.reset(seed=seed)

    terminated = False
    truncated = False
    sequence = []
    previous_policy = None

    while not terminated and not truncated:
        action, _ = model.predict(observation, deterministic=True)
        action = int(action)

        guidance = env._decode_action(action)
        probability = get_action_probability(model, observation, action)

        if record_sequence and guidance != previous_policy:
            sequence.append({
                "time": env.backend.elapsed_time,
                "action": action,
                "policy": dict(guidance),
                "probability": probability,
            })
            previous_policy = dict(guidance)

        observation, reward, terminated, truncated, info = env.step(action)

    result = {
        "time": info["elapsed_time"],
        "remaining": info["remaining_agents"],
        "reward": reward,
        "sequence": sequence,
    }

    env.close()
    return result


def main():
    with open("results/static_plan_seed0.json") as file:
        static_results = json.load(file)

    models = {
        name: PPO.load(path, device="cpu")
        for name, path in MODELS.items()
    }

    print("\n===== SEED 0 EVALUATION =====\n")

    header = (
        f"{'Population':<12}"
        f"{'Best static':>14}"
        f"{'Policy':>10}"
        f"{'CCA':>10}"
        f"{'PPO 5k':>10}"
        f"{'PPO 10k':>10}"
    )

    print(header)
    print("-" * len(header))

    for population_name, room_counts in POPULATION_SCENARIOS.items():
        population_results = static_results[population_name]

        successful = {
            key: result
            for key, result in population_results.items()
            if result["remaining_agents"] == 0
        }

        best_key, best_result = min(
            successful.items(),
            key=lambda item: item[1]["elapsed_time"],
        )

        cca_key = policy_key(CCA, ["J1", "J2", "J3"])
        cca_time = population_results[cca_key]["elapsed_time"]

        ppo_results = {
            name: evaluate_model(model, room_counts, SEED)
            for name, model in models.items()
        }

        best_policy = {
            part.split("=")[0]: part.split("=")[1]
            for part in best_key.split("|")
        }

        print(
            f"{population_name:<12}"
            f"{best_result['elapsed_time']:>14.2f}"
            f"{short_policy(best_policy):>10}"
            f"{cca_time:>10.2f}"
            f"{ppo_results['PPO_5k']['time']:>10.2f}"
            f"{ppo_results['PPO_10k']['time']:>10.2f}"
        )

    print("\n===== PPO 10k DYNAMIC GUIDANCE =====")

    model = models["PPO_10k"]

    for population_name, room_counts in POPULATION_SCENARIOS.items():
        result = evaluate_model(
            model,
            room_counts,
            seed=SEED,
            record_sequence=True,
        )

        print(f"\n{population_name} | final time = {result['time']:.2f}s")

        for entry in result["sequence"]:
            print(
                f"  t={entry['time']:6.2f} | "
                f"action={entry['action']:2d} | "
                f"{short_policy(entry['policy']):<8} | "
                f"p={entry['probability']:.3f}"
            )


if __name__ == "__main__":
    main()