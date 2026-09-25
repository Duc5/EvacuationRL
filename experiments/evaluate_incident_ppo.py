"""Evaluate trained PPO against fixed and one-switch incident baselines."""

import csv
from collections import Counter
from pathlib import Path

import numpy as np
import torch
from stable_baselines3 import PPO

from scenarios.building_v2 import BuildingV2Scenario
from envs.evacuation_env import EvacuationEnv


MODEL_PATH = Path(
    "models/incident_ppo_checkpoints/ppo_incident_24576_steps.zip"
    # "models/ppo_incident_v1.zip"
    # "models/incident_ppo_checkpoints/ppo_incident_26624_steps.zip"
)

OUTPUT_PATH = Path(
    "results/incident_ppo_eval_24576_steps_seeds5-14.csv"
)

INITIAL_COUNTS = {
    "A": 40,
    "B": 20,
    "C": 20,
    "D": 20,
}

SURGE_ROOM = "C"
SURGE_TIME = 10.0
SURGE_COUNT = 40

INCIDENTS = [
    "B_connector",
    "C_exit",
]

SEEDS = range(5, 15)

CONTROL_INTERVAL = 10.0
MAX_TIME = 180.0

ROBUST_FIXED_POLICY = "CJ3A"
EVENT_BLIND_POST_POLICY = "AJ1B"

ORACLE_POST_POLICIES = {
    "B_connector": "CJ3A",
    "C_exit": "AJ3B",
}


def short_policy(policy):
    return "".join(
        policy[j]
        for j in ["J1", "J2", "J3"]
    )


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
        incidents=[incident],
    )


def get_policy_mapping():
    env = make_env("B_connector")

    labels = [
        short_policy(policy)
        for policy in env.guidance_plans
    ]

    env.close()

    return {
        label: action
        for action, label in enumerate(labels)
    }, labels


def run_episode(
    controller,
    incident,
    seed,
    model,
    action_map,
    labels,
):
    env = make_env(incident)

    try:
        obs, info = env.reset(seed=seed)

        terminated = False
        truncated = False

        episode_reward = 0.0
        step_number = 0
        action_trace = []

        first_action = None
        first_post_action = None

        while not terminated and not truncated:
            if controller == "ppo":
                action, _ = model.predict(
                    obs,
                    deterministic=True,
                )
                action = int(action)

            elif controller == "fixed":
                action = action_map[ROBUST_FIXED_POLICY]

            elif controller == "event_blind":
                if step_number == 0:
                    action = action_map["CCA"]
                else:
                    action = action_map[
                        EVENT_BLIND_POST_POLICY
                    ]

            elif controller == "oracle":
                if step_number == 0:
                    action = action_map["CCA"]
                else:
                    action = action_map[
                        ORACLE_POST_POLICIES[incident]
                    ]

            else:
                raise ValueError(
                    f"Unknown controller: {controller}"
                )

            policy_name = labels[action]

            if step_number == 0:
                first_action = policy_name

            elif step_number == 1:
                first_post_action = policy_name

            action_trace.append(
                f"{info['elapsed_time']:.2f}:{policy_name}"
            )

            obs, reward, terminated, truncated, info = (
                env.step(action)
            )

            episode_reward += reward

            # The incident occurs during the first 10-second step.
            # Check that PPO can actually observe it afterward.
            if step_number == 0 and not terminated:
                expected_flags = (
                    np.array([1.0, 0.0])
                    if incident == "B_connector"
                    else np.array([0.0, 1.0])
                )

                if not np.allclose(
                    obs[-2:],
                    expected_flags,
                ):
                    raise RuntimeError(
                        f"{incident}: expected flags "
                        f"{expected_flags}, got {obs[-2:]}"
                    )

            step_number += 1

        return {
            "controller": controller,
            "seed": seed,
            "incident": incident,
            "elapsed_time": info["elapsed_time"],
            "episode_reward": episode_reward,
            "remaining_agents": info["remaining_agents"],
            "first_action": first_action,
            "first_post_action": first_post_action,
            "action_trace": " | ".join(action_trace),
            "error": "",
        }

    except Exception as error:
        return {
            "controller": controller,
            "seed": seed,
            "incident": incident,
            "elapsed_time": MAX_TIME,
            "episode_reward": 0.0,
            "remaining_agents": -1,
            "first_action": "",
            "first_post_action": "",
            "action_trace": "",
            "error": str(error),
        }

    finally:
        env.close()


def successful(rows, controller):
    return [
        row for row in rows
        if row["controller"] == controller
        and row["remaining_agents"] == 0
        and not row["error"]
    ]


def main():
    torch.set_num_threads(1)
    torch.set_num_interop_threads(1)

    if not MODEL_PATH.exists():
        raise FileNotFoundError(
            f"Model not found: {MODEL_PATH}"
        )

    print(f"Loading model: {MODEL_PATH}")

    model = PPO.load(
        MODEL_PATH,
        device="cpu",
    )

    print(f"Model timesteps: {model.num_timesteps}")
    print(f"n_steps: {model.n_steps}")
    print(f"batch_size: {model.batch_size}")

    action_map, labels = get_policy_mapping()

    controllers = [
        "ppo",
        "fixed",
        "event_blind",
        "oracle",
    ]

    rows = []

    total_runs = (
        len(controllers)
        * len(SEEDS)
        * len(INCIDENTS)
    )

    completed = 0

    for seed in SEEDS:
        for incident in INCIDENTS:
            for controller in controllers:
                row = run_episode(
                    controller=controller,
                    incident=incident,
                    seed=seed,
                    model=model,
                    action_map=action_map,
                    labels=labels,
                )

                rows.append(row)
                completed += 1

                print(
                    f"[{completed:03}/{total_runs}] "
                    f"seed={seed:2} | "
                    f"{incident:<11} | "
                    f"{controller:<11} | "
                    f"time={row['elapsed_time']:6.2f} | "
                    f"reward={row['episode_reward']:7.4f} | "
                    f"{row['first_action']} -> "
                    f"{row['first_post_action']}"
                )

                if row["error"]:
                    print(
                        f"    ERROR: {row['error']}"
                    )

    # ------------------------------------------------------
    # Save raw results
    # ------------------------------------------------------

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(
        OUTPUT_PATH,
        "w",
        newline="",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nSaved results to {OUTPUT_PATH}")

    # ------------------------------------------------------
    # Completeness
    # ------------------------------------------------------

    print("\n===== COMPLETENESS =====")

    expected = len(SEEDS) * len(INCIDENTS)

    for controller in controllers:
        controller_rows = [
            row for row in rows
            if row["controller"] == controller
        ]

        errors = sum(
            bool(row["error"])
            for row in controller_rows
        )

        timeouts = sum(
            row["remaining_agents"] > 0
            and not row["error"]
            for row in controller_rows
        )

        print(
            f"{controller:<11} | "
            f"rows={len(controller_rows):2}/{expected} | "
            f"errors={errors:2} | "
            f"timeouts={timeouts:2}"
        )

    # ------------------------------------------------------
    # Controller means
    # ------------------------------------------------------

    print("\n===== OVERALL PERFORMANCE =====")

    summaries = {}

    for controller in controllers:
        valid = successful(rows, controller)

        if len(valid) != expected:
            raise RuntimeError(
                f"{controller}: only "
                f"{len(valid)}/{expected} successful runs."
            )

        mean_time = sum(
            row["elapsed_time"]
            for row in valid
        ) / len(valid)

        mean_reward = sum(
            row["episode_reward"]
            for row in valid
        ) / len(valid)

        summaries[controller] = {
            "mean_time": mean_time,
            "mean_reward": mean_reward,
        }

        print(
            f"{controller:<11} | "
            f"time={mean_time:6.2f} | "
            f"reward={mean_reward:7.4f}"
        )

    # ------------------------------------------------------
    # Performance by incident
    # ------------------------------------------------------

    print("\n===== PERFORMANCE BY INCIDENT =====")

    for incident in INCIDENTS:
        print(f"\n{incident}")

        for controller in controllers:
            valid = [
                row for row in successful(rows, controller)
                if row["incident"] == incident
            ]

            mean_time = sum(
                row["elapsed_time"]
                for row in valid
            ) / len(valid)

            mean_reward = sum(
                row["episode_reward"]
                for row in valid
            ) / len(valid)

            print(
                f"  {controller:<11} | "
                f"time={mean_time:6.2f} | "
                f"reward={mean_reward:7.4f}"
            )

    # ------------------------------------------------------
    # PPO action behaviour
    # ------------------------------------------------------

    print("\n===== PPO ACTION BEHAVIOUR =====")

    ppo_rows = successful(rows, "ppo")

    pre_actions = Counter(
        row["first_action"]
        for row in ppo_rows
    )

    print("\nPre-incident actions:")
    for action, count in pre_actions.most_common():
        print(f"  {action:<6}: {count}")

    for incident in INCIDENTS:
        post_actions = Counter(
            row["first_post_action"]
            for row in ppo_rows
            if row["incident"] == incident
        )

        print(
            f"\nFirst PPO action after {incident}:"
        )

        for action, count in post_actions.most_common():
            print(f"  {action:<6}: {count}")

    print("\nPPO action traces:")

    for row in ppo_rows:
        print(
            f"seed={row['seed']:2} | "
            f"{row['incident']:<11} | "
            f"{row['action_trace']}"
        )

    # ------------------------------------------------------
    # Headroom captured
    # ------------------------------------------------------

    fixed_time = summaries["fixed"]["mean_time"]
    ppo_time = summaries["ppo"]["mean_time"]
    oracle_time = summaries["oracle"]["mean_time"]
    blind_time = summaries["event_blind"]["mean_time"]

    ppo_gain = fixed_time - ppo_time
    ppo_gain_pct = 100 * ppo_gain / fixed_time

    available_headroom = fixed_time - oracle_time

    if available_headroom > 0:
        captured = (
            100 * ppo_gain / available_headroom
        )
    else:
        captured = float("nan")

    print("\n===== PPO ADAPTIVE PERFORMANCE =====")

    print(
        f"Robust fixed CJ3A: "
        f"{fixed_time:.2f}s"
    )

    print(
        f"Event-blind CCA->{EVENT_BLIND_POST_POLICY}: "
        f"{blind_time:.2f}s"
    )

    print(
        f"PPO: "
        f"{ppo_time:.2f}s"
    )

    print(
        f"Incident-aware oracle: "
        f"{oracle_time:.2f}s"
    )

    print(
        f"\nPPO gain vs robust fixed: "
        f"{ppo_gain:.2f}s "
        f"({ppo_gain_pct:.1f}%)"
    )

    print(
        f"Available adaptive headroom: "
        f"{available_headroom:.2f}s"
    )

    print(
        f"Headroom captured by PPO: "
        f"{captured:.1f}%"
    )


if __name__ == "__main__":
    main()