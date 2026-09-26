"""Cheap V3-B diagnostic: CCA-first switching with a smaller C-room surge."""

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

INCIDENTS = [None,"C_exit","B_route_08","bottom_loop_left"]

PRE_POLICY = "CCA"

TEST_POST_POLICIES = [
    "AJ3B",    # J2 -> J3 -> B: should be directly hurt
    "ACA",   # another strong B-bound response
    "CCA",     # J2 -> C
    "ACB",     # J2 -> C
    "CJ3A",    # J2 -> J3 -> A
    "ACJ1",    # J2 -> J3 -> A
]

SEED = 0
CONTROL_INTERVAL = 10.0
MAX_TIME = 180.0
MAX_WORKERS = 11

OUTPUT_PATH = Path(
    "results/v3b_diagnostic_Csurge20_CCAfirst_seed0.csv"
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
            # CCA controls the first 0-10 s interval.
            # Incident + surge happen during that interval at t=10.
            # Candidate response is used afterwards.
            action = pre_action if step_number == 0 else post_action

            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            step_number += 1

        return {
            "incident": incident,
            "pre_policy": PRE_POLICY,
            "post_policy": post_policy,
            "seed": SEED,
            "surge_count": SURGE_COUNT,
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
            "surge_count": SURGE_COUNT,
            "elapsed_time": MAX_TIME,
            "remaining_agents": -1,
            "episode_reward": 0.0,
            "error": str(error),
        }

    finally:
        env.close()


def main():
    # Build once so policy labels map to the correct Discrete(35) action IDs.
    template = BuildingV2Scenario(room_counts=dict(INITIAL_COUNTS))

    env = EvacuationEnv(
        scenario=template,
        control_interval=CONTROL_INTERVAL,
        max_time=MAX_TIME,
    )

    policies = env.guidance_plans
    env.close()

    labels = [short_policy(policy) for policy in policies]

    if PRE_POLICY not in labels:
        raise RuntimeError(f"Pre-event policy {PRE_POLICY} not found.")

    missing = [
        policy for policy in TEST_POST_POLICIES
        if policy not in labels
    ]

    if missing:
        raise RuntimeError(f"Post-event policies not found: {missing}")

    pre_action = labels.index(PRE_POLICY)

    post_candidates = [
        (labels.index(policy), policy)
        for policy in TEST_POST_POLICIES
    ]

    jobs = [
        (incident, pre_action, post_action, post_policy)
        for incident in INCIDENTS
        for post_action, post_policy in post_candidates
    ]

    print(f"Full action space: {len(policies)}")
    print(f"Tested post-event policies: {len(TEST_POST_POLICIES)}")
    print(f"Pre-event policy: {PRE_POLICY}")
    print(f"C-room surge: {SURGE_COUNT} at t={SURGE_TIME:.0f}s")
    print(f"Incident conditions: {len(INCIDENTS)}")
    print(f"Total simulations: {len(jobs)}")

    rows = []

    ctx = mp.get_context("spawn")
    executor = ProcessPoolExecutor(
        max_workers=MAX_WORKERS,
        mp_context=ctx,
    )
    def incident_label(incident):
        return "None" if incident is None else incident
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
                f"{incident_label(result['incident']):<11} | "
                f"{result['pre_policy']}->{result['post_policy']:<6} | "
                f"time={result['elapsed_time']:6.2f} | "
                f"remaining={result['remaining_agents']}",
                flush=True,
            )

            if result["error"]:
                print(f"  ERROR: {result['error']}", flush=True)

        print("\nAll simulation jobs returned.", flush=True)

        rows.sort(key=lambda row: (
            incident_label(row["incident"]),
            row["elapsed_time"],
        ))

        OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

        with open(OUTPUT_PATH, "w", newline="") as file:
            writer = csv.DictWriter(file, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)

        print(f"Results saved to {OUTPUT_PATH}", flush=True)

    finally:
        print("Shutting down worker processes...", flush=True)
        executor.shutdown(wait=True, cancel_futures=True)
        print("Worker processes shut down.", flush=True)

    # ---------------------------------------------------------
    # Completeness
    # ---------------------------------------------------------

    print("\n===== SWEEP COMPLETENESS =====")

    for incident in INCIDENTS:
        incident_rows = [
            row for row in rows
            if row["incident"] == incident
        ]

        errors = sum(bool(row["error"]) for row in incident_rows)

        timeouts = sum(
            row["remaining_agents"] > 0 and not row["error"]
            for row in incident_rows
        )

        print(
            f"{incident_label(incident):<11} | "
            f"rows={len(incident_rows):2}/{len(TEST_POST_POLICIES)} | "
            f"errors={errors:2} | "
            f"timeouts={timeouts:2}"
        )

    # ---------------------------------------------------------
    # Best tested response per incident
    # ---------------------------------------------------------

    print(
        f"\n===== BEST {PRE_POLICY}-FIRST POLICY PER INCIDENT "
        f"(AMONG {len(TEST_POST_POLICIES)} TESTED) ====="
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
            print(f"{incident:<11} | NO SUCCESSFUL RUNS")
            continue

        best = min(valid, key=lambda row: row["elapsed_time"])
        best_by_incident[incident] = best

        print(
            f"{incident_label(incident):<11} | "
            f"{PRE_POLICY}->{best['post_policy']:<6} | "
            f"time={best['elapsed_time']:6.2f} | "
            f"reward={best['episode_reward']:7.4f}"
        )

    if len(best_by_incident) == len(INCIDENTS):
        incident_aware_mean = (
            sum(row["elapsed_time"] for row in best_by_incident.values())
            / len(best_by_incident)
        )

        incident_aware_reward = (
            sum(row["episode_reward"] for row in best_by_incident.values())
            / len(best_by_incident)
        )

        print(
            f"\nBest tested incident-aware mean time: "
            f"{incident_aware_mean:.2f}s"
        )
        print(
            f"Best tested incident-aware mean reward: "
            f"{incident_aware_reward:.4f}"
        )

    # ---------------------------------------------------------
    # Full ranking for each incident
    # ---------------------------------------------------------

    print("\n===== POLICY RANKING BY INCIDENT =====")

    for incident in INCIDENTS:
        print(f"\n{incident_label(incident)}")

        incident_rows = [
            row for row in rows
            if row["incident"] == incident
        ]

        incident_rows.sort(key=lambda row: (
            row["remaining_agents"] != 0 or bool(row["error"]),
            row["elapsed_time"],
        ))

        for rank, row in enumerate(incident_rows, start=1):
            status = "OK"

            if row["error"]:
                status = "ERROR"
            elif row["remaining_agents"] > 0:
                status = f"TIMEOUT({row['remaining_agents']})"

            print(
                f"{rank:2}. "
                f"{PRE_POLICY}->{row['post_policy']:<6} | "
                f"time={row['elapsed_time']:6.2f} | "
                f"reward={row['episode_reward']:7.4f} | "
                f"{status}"
            )

    # ---------------------------------------------------------
    # Best event-blind response among tested policies
    # ---------------------------------------------------------

    print(
        f"\n===== BEST EVENT-BLIND {PRE_POLICY}-FIRST POLICY "
        f"(AMONG TESTED) ====="
    )

    summaries = []

    for post_policy in TEST_POST_POLICIES:
        policy_rows = [
            row for row in rows
            if row["post_policy"] == post_policy
        ]

        if len(policy_rows) != len(INCIDENTS):
            continue

        if any(
            row["remaining_agents"] != 0 or row["error"]
            for row in policy_rows
        ):
            continue

        mean_time = (
            sum(row["elapsed_time"] for row in policy_rows)
            / len(policy_rows)
        )

        mean_reward = (
            sum(row["episode_reward"] for row in policy_rows)
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

    summaries.sort(key=lambda row: row["mean_time"])

    if not summaries:
        print("No tested post-event policy completed all four incidents.")
        return

    for rank, result in enumerate(summaries, start=1):
        print(
            f"{rank:2}. "
            f"{PRE_POLICY}->{result['post_policy']:<6} | "
            f"mean={result['mean_time']:6.2f} | "
            f"worst={result['worst_time']:6.2f} | "
            f"reward={result['mean_reward']:7.4f}"
        )

    # ---------------------------------------------------------
    # Value of incident information among these six policies
    # ---------------------------------------------------------

    if len(best_by_incident) == len(INCIDENTS):
        event_blind = summaries[0]

        gain_vs_blind = (
            event_blind["mean_time"]
            - incident_aware_mean
        )

        gain_vs_blind_pct = (
            100 * gain_vs_blind / event_blind["mean_time"]
        )

        print("\n===== INCIDENT-INFORMATION VALUE =====")

        print(
            f"Best tested event-blind switch: "
            f"{PRE_POLICY}->{event_blind['post_policy']} "
            f"({event_blind['mean_time']:.2f}s)"
        )

        print(
            f"Best tested incident-aware mean: "
            f"{incident_aware_mean:.2f}s"
        )

        print(
            f"Gain from knowing incident identity: "
            f"{gain_vs_blind:.2f}s "
            f"({gain_vs_blind_pct:.1f}%)"
        )


if __name__ == "__main__":
    mp.freeze_support()
    main()