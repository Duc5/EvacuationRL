"""Cheap multi-seed validation of the V3-B three-way A/B/C response."""

import csv
import multiprocessing as mp
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from scenarios.building_v2 import BuildingV2Scenario
from envs.evacuation_env import EvacuationEnv


INITIAL_COUNTS = {"A": 40, "B": 20, "C": 20, "D": 20}

SURGE_ROOM = "C"
SURGE_TIME = 10.0
SURGE_COUNT = 40

PRE_POLICY = "CCA"

SEEDS = [1, 2, 3, 4]

CONTROL_INTERVAL = 10.0
MAX_TIME = 180.0
MAX_WORKERS = 11

# Four policies are shared across every incident so we can also
# calculate an event-blind benchmark from this validation.
POLICIES_BY_INCIDENT = {
    "B_route_08": [
        "CJ3A",   # expected A-routing winner
        "AJ3B",   # B-routing competitor
        "ACB",    # C-routing competitor
        "CCA",    # C-routing competitor
    ],
    "C_exit": [
        "AJ3B",   # expected B-routing winner
        "CJ3A",   # A-routing competitor
        "ACB",    # C-routing competitor
        "CCA",    # C-routing competitor
    ],
    "bottom_loop_left": [
        "ACB",    # seed-0 reward winner
        "ACA",    # same A,C,* routing class
        "ACJ1",
        "CCA",
        "AJ3B",   # B-routing competitor
        "CJ3A",   # A-routing competitor
    ],
}

ROUTE_CLASS = {
    "CJ3A": "A-route",
    "AJ3B": "B-route",
    "ACB": "C-route",
    "ACA": "C-route",
    "ACJ1": "C-route",
    "CCA": "C-route",
}

EXPECTED_CLASS = {
    "B_route_08": "A-route",
    "C_exit": "B-route",
    "bottom_loop_left": "C-route",
}

OUTPUT_PATH = Path(
    "results/v3b_threeway_multiseed_seeds1-4.csv"
)


def short_policy(policy):
    return "".join(policy[j] for j in ["J1", "J2", "J3"])


def run_case(incident, pre_action, post_action, post_policy, seed):
    scenario = BuildingV2Scenario(
        room_counts=dict(INITIAL_COUNTS),
        dynamic_events=[{
            "time": SURGE_TIME,
            "room": SURGE_ROOM,
            "count": SURGE_COUNT,
            "incident": incident,
        }],
    )

    env = EvacuationEnv(
        scenario=scenario,
        control_interval=CONTROL_INTERVAL,
        max_time=MAX_TIME,
    )

    try:
        obs, info = env.reset(seed=seed)

        terminated = False
        truncated = False
        episode_reward = 0.0
        step_number = 0

        while not terminated and not truncated:
            action = pre_action if step_number == 0 else post_action
            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            step_number += 1

        return {
            "incident": incident,
            "pre_policy": PRE_POLICY,
            "post_policy": post_policy,
            "route_class": ROUTE_CLASS.get(post_policy, "other"),
            "seed": seed,
            "elapsed_time": info["elapsed_time"],
            "remaining_agents": info["remaining_agents"],
            "episode_reward": episode_reward,
            "error": "",
        }

    except Exception as error:
        return {
            "incident": incident,
            "pre_policy": PRE_POLICY,
            "post_policy": post_policy,
            "route_class": ROUTE_CLASS.get(post_policy, "other"),
            "seed": seed,
            "elapsed_time": MAX_TIME,
            "remaining_agents": -1,
            "episode_reward": 0.0,
            "error": str(error),
        }

    finally:
        env.close()


def mean(values):
    return sum(values) / len(values)


def main():
    template = BuildingV2Scenario(
        room_counts=dict(INITIAL_COUNTS)
    )

    env = EvacuationEnv(
        scenario=template,
        control_interval=CONTROL_INTERVAL,
        max_time=MAX_TIME,
    )

    policies = env.guidance_plans
    labels = [short_policy(policy) for policy in policies]
    env.close()

    if PRE_POLICY not in labels:
        raise RuntimeError(
            f"Pre-event policy {PRE_POLICY} not found."
        )

    requested = {
        policy
        for incident_policies in POLICIES_BY_INCIDENT.values()
        for policy in incident_policies
    }

    missing = sorted(requested - set(labels))
    if missing:
        raise RuntimeError(
            f"Policies not found in action space: {missing}"
        )

    pre_action = labels.index(PRE_POLICY)

    jobs = []

    for incident, tested_policies in POLICIES_BY_INCIDENT.items():
        for post_policy in tested_policies:
            post_action = labels.index(post_policy)

            for seed in SEEDS:
                jobs.append((
                    incident,
                    pre_action,
                    post_action,
                    post_policy,
                    seed,
                ))

    print(f"Full action space: {len(labels)}")
    print(f"Pre-event policy: {PRE_POLICY}")
    print(f"Seeds: {SEEDS}")
    print(f"C-room surge: {SURGE_COUNT} at t={SURGE_TIME:.0f}s")
    print(f"Total simulations: {len(jobs)}")

    rows = []

    ctx = mp.get_context("spawn")
    executor = ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        mp_context=ctx,
    )

    try:
        futures = {
            executor.submit(run_case, *job): job
            for job in jobs
        }

        for completed, future in enumerate(
            as_completed(futures),
            start=1,
        ):
            result = future.result()
            rows.append(result)

            print(
                f"[{completed:02}/{len(jobs)}] "
                f"{result['incident']:<16} | "
                f"seed={result['seed']} | "
                f"{PRE_POLICY}->{result['post_policy']:<5} | "
                f"time={result['elapsed_time']:6.2f} | "
                f"remaining={result['remaining_agents']}",
                flush=True,
            )

            if result["error"]:
                print(
                    f"  ERROR: {result['error']}",
                    flush=True,
                )

    finally:
        executor.shutdown(
            wait=True,
            cancel_futures=True,
        )

    rows.sort(key=lambda row: (
        row["incident"],
        row["post_policy"],
        row["seed"],
    ))

    OUTPUT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    with open(OUTPUT_PATH, "w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=rows[0].keys(),
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nResults saved to {OUTPUT_PATH}")

    # ---------------------------------------------------------
    # Aggregate each incident-policy pair over seeds
    # ---------------------------------------------------------

    grouped = defaultdict(list)

    for row in rows:
        grouped[
            (row["incident"], row["post_policy"])
        ].append(row)

    summaries = []

    for (incident, policy), policy_rows in grouped.items():
        successes = [
            row for row in policy_rows
            if row["remaining_agents"] == 0
            and not row["error"]
        ]

        summary = {
            "incident": incident,
            "policy": policy,
            "route_class": ROUTE_CLASS.get(policy, "other"),
            "successful": len(successes),
            "total": len(policy_rows),
            "errors": sum(bool(row["error"]) for row in policy_rows),
        }

        if successes:
            summary["mean_time"] = mean(
                row["elapsed_time"] for row in successes
            )
            summary["mean_reward"] = mean(
                row["episode_reward"] for row in successes
            )
            summary["worst_time"] = max(
                row["elapsed_time"] for row in successes
            )
        else:
            summary["mean_time"] = float("inf")
            summary["mean_reward"] = float("-inf")
            summary["worst_time"] = float("inf")

        summaries.append(summary)

    print("\n===== MULTI-SEED POLICY RANKING =====")

    best_time_by_incident = {}
    best_reward_by_incident = {}

    for incident in POLICIES_BY_INCIDENT:
        incident_rows = [
            row for row in summaries
            if row["incident"] == incident
        ]

        # Require 4/4 success to be eligible as a robust winner.
        eligible = [
            row for row in incident_rows
            if row["successful"] == len(SEEDS)
            and row["errors"] == 0
        ]

        print(f"\n{incident}")

        for result in sorted(
            incident_rows,
            key=lambda row: row["mean_time"],
        ):
            print(
                f"{result['policy']:<5} | "
                f"{result['route_class']:<7} | "
                f"mean={result['mean_time']:6.2f} | "
                f"worst={result['worst_time']:6.2f} | "
                f"reward={result['mean_reward']:7.4f} | "
                f"success={result['successful']}/{result['total']}"
            )

        if not eligible:
            raise RuntimeError(
                f"No policy completed all seeds for {incident}."
            )

        best_time = min(
            eligible,
            key=lambda row: row["mean_time"],
        )

        best_reward = max(
            eligible,
            key=lambda row: row["mean_reward"],
        )

        best_time_by_incident[incident] = best_time
        best_reward_by_incident[incident] = best_reward

    # ---------------------------------------------------------
    # Main A/B/C validation
    # ---------------------------------------------------------

    print("\n===== BEST BY MEAN EVACUATION TIME =====")

    for incident, result in best_time_by_incident.items():
        expected = EXPECTED_CLASS[incident]
        passed = result["route_class"] == expected

        print(
            f"{incident:<16} | "
            f"{result['policy']:<5} | "
            f"{result['route_class']:<7} | "
            f"{result['mean_time']:6.2f}s | "
            f"expected={expected:<7} | "
            f"{'PASS' if passed else 'CHECK'}"
        )

    print("\n===== BEST BY MEAN REWARD =====")

    for incident, result in best_reward_by_incident.items():
        expected = EXPECTED_CLASS[incident]
        passed = result["route_class"] == expected

        print(
            f"{incident:<16} | "
            f"{result['policy']:<5} | "
            f"{result['route_class']:<7} | "
            f"reward={result['mean_reward']:7.4f} | "
            f"expected={expected:<7} | "
            f"{'PASS' if passed else 'CHECK'}"
        )

    # ---------------------------------------------------------
    # Event-blind comparison
    # ---------------------------------------------------------

    shared_policies = set.intersection(
        *(
            set(policies)
            for policies in POLICIES_BY_INCIDENT.values()
        )
    )

    blind_summaries = []

    for policy in sorted(shared_policies):
        policy_rows = [
            row for row in rows
            if row["post_policy"] == policy
        ]

        expected_count = (
            len(POLICIES_BY_INCIDENT) * len(SEEDS)
        )

        if len(policy_rows) != expected_count:
            continue

        if any(
            row["remaining_agents"] != 0
            or row["error"]
            for row in policy_rows
        ):
            continue

        blind_summaries.append({
            "policy": policy,
            "mean_time": mean(
                row["elapsed_time"] for row in policy_rows
            ),
            "mean_reward": mean(
                row["episode_reward"] for row in policy_rows
            ),
            "worst_time": max(
                row["elapsed_time"] for row in policy_rows
            ),
        })

    print("\n===== EVENT-BLIND SHARED POLICIES =====")

    for result in sorted(
        blind_summaries,
        key=lambda row: row["mean_time"],
    ):
        print(
            f"{PRE_POLICY}->{result['policy']:<5} | "
            f"mean={result['mean_time']:6.2f} | "
            f"worst={result['worst_time']:6.2f} | "
            f"reward={result['mean_reward']:7.4f}"
        )

    if blind_summaries:
        best_blind_time = min(
            blind_summaries,
            key=lambda row: row["mean_time"],
        )

        best_blind_reward = max(
            blind_summaries,
            key=lambda row: row["mean_reward"],
        )

        aware_time = mean(
            result["mean_time"]
            for result in best_time_by_incident.values()
        )

        aware_reward = mean(
            result["mean_reward"]
            for result in best_reward_by_incident.values()
        )

        gain = best_blind_time["mean_time"] - aware_time
        gain_pct = 100 * gain / best_blind_time["mean_time"]

        reward_gain = (
            aware_reward
            - best_blind_reward["mean_reward"]
        )

        print("\n===== MULTI-SEED INCIDENT INFORMATION =====")

        print(
            f"Best event-blind by time: "
            f"{PRE_POLICY}->{best_blind_time['policy']} "
            f"({best_blind_time['mean_time']:.2f}s)"
        )

        print(
            f"Incident-aware mean time: "
            f"{aware_time:.2f}s"
        )

        print(
            f"Time gain: "
            f"{gain:.2f}s ({gain_pct:.1f}%)"
        )

        print(
            f"\nBest event-blind by reward: "
            f"{PRE_POLICY}->{best_blind_reward['policy']} "
            f"({best_blind_reward['mean_reward']:.4f})"
        )

        print(
            f"Incident-aware mean reward: "
            f"{aware_reward:.4f}"
        )

        print(
            f"Reward improvement: "
            f"{reward_gain:+.4f}"
        )


if __name__ == "__main__":
    mp.freeze_support()
    main()