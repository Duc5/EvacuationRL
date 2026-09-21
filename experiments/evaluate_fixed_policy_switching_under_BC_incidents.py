"""Evaluate CCA-first timed switching under B/C capacity incidents."""

import csv
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from scenarios.building_v2 import BuildingV2Scenario
from envs.evacuation_env import EvacuationEnv


INITIAL_COUNTS = {"A": 40, "B": 20, "C": 20, "D": 20}

SURGE_ROOM = "C"
SURGE_TIME = 10.0
SURGE_COUNT = 40

INCIDENTS = ["B_connector", "C_exit"]

PRE_POLICY = "CCA"

SEED = 1
CONTROL_INTERVAL = 10.0
MAX_TIME = 180.0
MAX_WORKERS = 11

# Result from the fixed-policy sweep you just ran.
STATIC_BASELINE_POLICY = "CJ3A"
STATIC_BASELINE_MEAN = 71.47

OUTPUT_PATH = Path(
    "results/incident_one_switch_"
    "A40_B20_C20_D20_Csurge40_t10_seed1.csv"
)


def short_policy(policy):
    return "".join(policy[j] for j in ["J1", "J2", "J3"])


def run_case(incident, pre_action, post_action, post_policy):
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
        obs, info = env.reset(seed=SEED)

        terminated = False
        truncated = False
        episode_reward = 0.0
        step_number = 0

        while not terminated and not truncated:
            # First 10-second interval:
            # use CCA while the building is still in its normal state.
            #
            # The incident + surge trigger during this advance at t=10.
            #
            # Every later interval:
            # use the candidate post-event policy.
            action = pre_action if step_number == 0 else post_action

            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            step_number += 1

        return {
            "incident": incident,
            "pre_policy": PRE_POLICY,
            "post_policy": post_policy,
            "seed": SEED,
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
            "seed": SEED,
            "elapsed_time": MAX_TIME,
            "remaining_agents": -1,
            "episode_reward": 0.0,
            "error": str(error),
        }

    finally:
        env.close()


def main():
    # Build once so action numbers exactly match EvacuationEnv's
    # Discrete(35) action ordering.
    template = BuildingV2Scenario(
        room_counts=dict(INITIAL_COUNTS)
    )

    env = EvacuationEnv(
        scenario=template,
        control_interval=CONTROL_INTERVAL,
        max_time=MAX_TIME,
    )

    policies = env.guidance_plans
    env.close()

    labels = [
        short_policy(policy)
        for policy in policies
    ]

    if PRE_POLICY not in labels:
        raise RuntimeError(
            f"Pre-event policy {PRE_POLICY} not found."
        )

    pre_action = labels.index(PRE_POLICY)

    jobs = [
        (
            incident,
            pre_action,
            post_action,
            post_policy,
        )
        for incident in INCIDENTS
        for post_action, post_policy in enumerate(labels)
    ]

    print(f"Policies: {len(policies)}")
    print(f"Pre-event policy: {PRE_POLICY}")
    print(f"Incident conditions: {len(INCIDENTS)}")
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
                f"[{completed:03}/{len(jobs)}] "
                f"{result['incident']:<11} | "
                f"{result['pre_policy']}->{result['post_policy']:<6} | "
                f"time={result['elapsed_time']:6.2f} | "
                f"remaining={result['remaining_agents']}",
                flush=True,
            )

            if result["error"]:
                print(
                    f"  ERROR: {result['error']}",
                    flush=True,
                )

        print(
            "\nAll simulation jobs returned.",
            flush=True,
        )

        rows.sort(key=lambda row: (
            row["incident"],
            row["elapsed_time"],
        ))

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

        print(
            f"Results saved to {OUTPUT_PATH}",
            flush=True,
        )

    finally:
        print(
            "Shutting down worker processes...",
            flush=True,
        )

        executor.shutdown(
            wait=True,
            cancel_futures=True,
        )

        print(
            "Worker processes shut down.",
            flush=True,
        )

    # ---------------------------------------------------------
    # Completeness
    # ---------------------------------------------------------

    print("\n===== SWEEP COMPLETENESS =====")

    for incident in INCIDENTS:
        incident_rows = [
            row for row in rows
            if row["incident"] == incident
        ]

        errors = sum(
            bool(row["error"])
            for row in incident_rows
        )

        timeouts = sum(
            row["remaining_agents"] > 0
            and not row["error"]
            for row in incident_rows
        )

        print(
            f"{incident:<11} | "
            f"rows={len(incident_rows):2}/{len(policies)} | "
            f"errors={errors:2} | "
            f"timeouts={timeouts:2}"
        )

    # ---------------------------------------------------------
    # Best incident-aware switch
    # ---------------------------------------------------------

    print(
        "\n===== BEST CCA-FIRST POLICY PER INCIDENT ====="
    )

    best_by_incident = {}

    for incident in INCIDENTS:
        valid = [
            row for row in rows
            if row["incident"] == incident
            and row["remaining_agents"] == 0
            and not row["error"]
        ]

        if not valid:
            raise RuntimeError(
                f"No successful runs for {incident}."
            )

        best = min(
            valid,
            key=lambda row: row["elapsed_time"],
        )

        best_by_incident[incident] = best

        print(
            f"{incident:<11} | "
            f"{PRE_POLICY}->{best['post_policy']:<6} | "
            f"time={best['elapsed_time']:6.2f} | "
            f"reward={best['episode_reward']:7.4f}"
        )

    incident_aware_mean = (
        sum(
            row["elapsed_time"]
            for row in best_by_incident.values()
        )
        / len(best_by_incident)
    )

    incident_aware_reward = (
        sum(
            row["episode_reward"]
            for row in best_by_incident.values()
        )
        / len(best_by_incident)
    )

    print(
        f"\nIncident-aware mean time: "
        f"{incident_aware_mean:.2f}s"
    )

    print(
        f"Incident-aware mean reward: "
        f"{incident_aware_reward:.4f}"
    )

    # ---------------------------------------------------------
    # Best event-blind timed switch
    #
    # Same post-event policy must be used regardless of which
    # incident happened.
    # ---------------------------------------------------------

    print(
        "\n===== BEST EVENT-BLIND CCA-FIRST POLICIES ====="
    )

    summaries = []

    for post_policy in labels:
        policy_rows = [
            row for row in rows
            if row["post_policy"] == post_policy
        ]

        if len(policy_rows) != len(INCIDENTS):
            continue

        if any(
            row["remaining_agents"] != 0
            or row["error"]
            for row in policy_rows
        ):
            continue

        mean_time = (
            sum(
                row["elapsed_time"]
                for row in policy_rows
            )
            / len(policy_rows)
        )

        mean_reward = (
            sum(
                row["episode_reward"]
                for row in policy_rows
            )
            / len(policy_rows)
        )

        worst_time = max(
            row["elapsed_time"]
            for row in policy_rows
        )

        summaries.append({
            "post_policy": post_policy,
            "mean_time": mean_time,
            "mean_reward": mean_reward,
            "worst_time": worst_time,
        })

    summaries.sort(
        key=lambda row: row["mean_time"]
    )

    if not summaries:
        raise RuntimeError(
            "No event-blind timed policy completed "
            "successfully under every incident."
        )

    for rank, result in enumerate(
        summaries[:10],
        start=1,
    ):
        print(
            f"{rank:2}. "
            f"{PRE_POLICY}->{result['post_policy']:<6} | "
            f"mean={result['mean_time']:6.2f} | "
            f"worst={result['worst_time']:6.2f} | "
            f"reward={result['mean_reward']:7.4f}"
        )

    event_blind = summaries[0]

    # ---------------------------------------------------------
    # Headroom comparison
    # ---------------------------------------------------------

    print("\n===== RESPONSIVE GUIDANCE HEADROOM =====")

    gain_vs_static = (
        STATIC_BASELINE_MEAN
        - incident_aware_mean
    )

    gain_vs_static_pct = (
        100
        * gain_vs_static
        / STATIC_BASELINE_MEAN
    )

    gain_vs_blind = (
        event_blind["mean_time"]
        - incident_aware_mean
    )

    gain_vs_blind_pct = (
        100
        * gain_vs_blind
        / event_blind["mean_time"]
    )

    print(
        f"Best robust fixed: "
        f"{STATIC_BASELINE_POLICY} "
        f"({STATIC_BASELINE_MEAN:.2f}s)"
    )

    print(
        f"Best event-blind timed switch: "
        f"{PRE_POLICY}->{event_blind['post_policy']} "
        f"({event_blind['mean_time']:.2f}s)"
    )

    print(
        f"Incident-aware switch mean: "
        f"{incident_aware_mean:.2f}s"
    )

    print(
        f"\nGain vs robust fixed: "
        f"{gain_vs_static:.2f}s "
        f"({gain_vs_static_pct:.1f}%)"
    )

    print(
        f"Gain vs event-blind timed switch: "
        f"{gain_vs_blind:.2f}s "
        f"({gain_vs_blind_pct:.1f}%)"
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()