import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from scenarios.building_v2 import BuildingV2Scenario
from envs.evacuation_env import EvacuationEnv
import multiprocessing as mp

INITIAL_COUNTS = {"A": 40, "B": 20, "C": 20, "D": 20}
SURGE_ROOM = "C"
SURGE_TIME = 10.0
SURGE_COUNT = 40
INCIDENTS = ["A_exit","C_exit","top_loop","bottom_loop"]

SEED = 0
CONTROL_INTERVAL = 10.0
MAX_TIME = 180.0
MAX_WORKERS = 11

OUTPUT_PATH = Path("results/v3b_static_A40_B20_C20_D20_Csurge40_t10_seed0.csv")

def short_policy(policy):
    return "".join(policy[j] for j in ["J1", "J2", "J3"])


def run_case(action, policy,incident):
    scenario = BuildingV2Scenario(
        room_counts=dict(INITIAL_COUNTS),
        dynamic_events=[{
            "time": SURGE_TIME,
            "room": SURGE_ROOM,
            "count": SURGE_COUNT,
            "incident": incident
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

        while not terminated and not truncated:
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward

        return {
            "action": action,
            "policy": short_policy(policy),
            "incident": incident,
            "surge_room": SURGE_ROOM,
            "surge_time": SURGE_TIME,
            "elapsed_time": info["elapsed_time"],
            "remaining_agents": info["remaining_agents"],
            "episode_reward": episode_reward,
            "error": "",
        }

    except Exception as error:
        return {
            "action": action,
            "policy": short_policy(policy),
            "incident": incident,
            "surge_room": SURGE_ROOM,
            "surge_time": SURGE_TIME,
            "elapsed_time": MAX_TIME,
            "remaining_agents": -1,
            "episode_reward": 0.0,
            "error": str(error),
        }

    finally:
        env.close()


def main():
    template = BuildingV2Scenario(room_counts=dict(INITIAL_COUNTS))

    env = EvacuationEnv(
        scenario=template,
        control_interval=CONTROL_INTERVAL,
        max_time=MAX_TIME,
    )

    policies = env.guidance_plans
    env.close()

    jobs = [
        (action, policy, incident)
        for action, policy in enumerate(policies)
        for incident in INCIDENTS
    ]


    print(f"Policies: {len(policies)}")
    print(f"Incident conditions: {len(INCIDENTS)}")
    print(f"Total simulations: {len(jobs)}")

    rows = []

    ctx = mp.get_context("spawn")
    executor = ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=ctx)

    try:
        futures = {executor.submit(run_case, *job): job for job in jobs}

        for completed, future in enumerate(as_completed(futures), start=1):
            result = future.result()
            rows.append(result)

            print(
                f"[{completed:03}/{len(jobs)}] "
                f"{result['policy']:<6} | "
                f"{result['incident']:<11} | "
                f"time={result['elapsed_time']:6.2f} | "
                f"remaining={result['remaining_agents']}",
                flush=True,
            )

            if result["error"]:
                print(f"  ERROR: {result['error']}", flush=True)

        print("\nAll simulation jobs returned.", flush=True)

        rows.sort(key=lambda row: (
            row["incident"],
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



    print("\n===== BEST STATIC POLICY PER INCIDENT =====")

    oracle_times = []

    for incident in INCIDENTS:
        condition_rows = [
            row for row in rows
            if row["incident"] == incident
            and row["remaining_agents"] == 0
            and not row["error"]
        ]

        if not condition_rows:
            print(f"{incident}: NO SUCCESSFUL RUNS")
            continue

        best = min(condition_rows, key=lambda row: row["elapsed_time"])
        oracle_times.append(best["elapsed_time"])

        print(
            f"{incident:<11} | "
            f"{best['policy']:<6} | "
            f"time={best['elapsed_time']:6.2f} | "
            f"reward={best['episode_reward']:7.4f}"
        )

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
            f"{incident:<11} | "
            f"rows={len(incident_rows):2}/{len(policies)} | "
            f"errors={errors:2} | "
            f"timeouts={timeouts:2}"
        )


    print("\n===== BEST ROBUST STATIC POLICIES =====")

    summaries = []

    for action in range(len(policies)):
        policy_rows = [row for row in rows if row["action"] == action]

        if len(policy_rows) != len(INCIDENTS):
            continue

        if any(row["remaining_agents"] != 0 or row["error"] for row in policy_rows):
            continue

        mean_time = sum(row["elapsed_time"] for row in policy_rows) / len(policy_rows)
        mean_reward = sum(row["episode_reward"] for row in policy_rows) / len(policy_rows)
        worst_time = max(row["elapsed_time"] for row in policy_rows)

        summaries.append({
            "policy": policy_rows[0]["policy"],
            "mean_time": mean_time,
            "mean_reward": mean_reward,
            "worst_time": worst_time,
        })

    summaries.sort(key=lambda row: row["mean_time"])

    for rank, result in enumerate(summaries[:10], start=1):
        print(
            f"{rank:2}. {result['policy']:<6} | "
            f"mean={result['mean_time']:6.2f} | "
            f"worst={result['worst_time']:6.2f} | "
            f"reward={result['mean_reward']:7.4f}"
        )

    if not summaries:
        raise RuntimeError(
            "No static policy completed successfully under all incidents."
        )

    robust = summaries[0]

    if len(oracle_times) == len(INCIDENTS):
        oracle_mean = sum(oracle_times) / len(oracle_times)
        print(f"\nOracle per-incident mean: {oracle_mean:.2f}s")
    else:
        raise RuntimeError(
            f"Only {len(oracle_times)}/{len(INCIDENTS)} incidents had successful runs."
        )

    improvement = robust["mean_time"] - oracle_mean
    improvement_pct = 100 * improvement / robust["mean_time"]

    print("\n===== ADAPTIVE HEADROOM =====")
    print(
        f"Best robust fixed policy: {robust['policy']} "
        f"({robust['mean_time']:.2f}s mean)"
    )
    print(f"Best condition-specific fixed mean: {oracle_mean:.2f}s")
    print(
        f"Condition specific fixed headroom: "
        f"{improvement:.2f}s ({improvement_pct:.1f}%)"
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()