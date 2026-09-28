import csv
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
import multiprocessing as mp

from scenarios.building_v2 import BuildingV2Scenario
from envs.evacuation_env import EvacuationEnv


INITIAL_COUNTS = {"A": 40, "B": 20, "C": 20, "D": 20}

SURGE_ROOMS = ["A", "B", "C", "D"]
SURGE_COUNTS = [20, 30, 40]
SURGE_TIME = 10.0

INCIDENTS = [
    "C_exit",
    "B_route_08",
    "bottom_loop_left",
]

SEED = 0
CONTROL_INTERVAL = 10.0
MAX_TIME = 180.0
MAX_WORKERS = 11

OUTPUT_PATH = Path(
    "results/v3c2_static_all_surges_seed0.csv"
)


def short_policy(policy):
    return "".join(
        policy[j]
        for j in ["J1", "J2", "J3"]
    )


def run_case(
    action,
    policy,
    incident,
    surge_room,
    surge_count,
):
    scenario = BuildingV2Scenario(
        room_counts=dict(INITIAL_COUNTS),
        dynamic_events=[{
            "time": SURGE_TIME,
            "room": surge_room,
            "count": surge_count,
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

        while not terminated and not truncated:
            obs, reward, terminated, truncated, info = (
                env.step(action)
            )
            episode_reward += reward

        return {
            "action": action,
            "policy": short_policy(policy),
            "incident": incident,
            "surge_room": surge_room,
            "surge_count": surge_count,
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
            "surge_room": surge_room,
            "surge_count": surge_count,
            "surge_time": SURGE_TIME,
            "elapsed_time": MAX_TIME,
            "remaining_agents": -1,
            "episode_reward": 0.0,
            "error": str(error),
        }

    finally:
        env.close()


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
    env.close()

    jobs = [
        (
            action,
            policy,
            incident,
            surge_room,
            surge_count,
        )
        for action, policy in enumerate(policies)
        for incident in INCIDENTS
        for surge_room in SURGE_ROOMS
        for surge_count in SURGE_COUNTS
    ]

    num_conditions = (
        len(INCIDENTS)
        * len(SURGE_ROOMS)
        * len(SURGE_COUNTS)
    )

    print(f"Policies: {len(policies)}")
    print(f"Incident types: {len(INCIDENTS)}")
    print(
        f"Surge conditions: "
        f"{len(SURGE_ROOMS)} rooms × "
        f"{len(SURGE_COUNTS)} counts"
    )
    print(f"Full conditions: {num_conditions}")
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
                f"[{completed:04}/{len(jobs)}] "
                f"{result['policy']:<6} | "
                f"{result['incident']:<18} | "
                f"{result['surge_room']}"
                f"+{result['surge_count']:<2} | "
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

        rows.sort(
            key=lambda row: (
                row["incident"],
                row["surge_room"],
                row["surge_count"],
                row["elapsed_time"],
            )
        )

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

    # ==============================================================
    # Best policy for each exact condition
    # ==============================================================

    print(
        "\n===== BEST STATIC POLICY PER FULL CONDITION ====="
    )

    condition_best_times = []

    for incident in INCIDENTS:
        for surge_room in SURGE_ROOMS:
            for surge_count in SURGE_COUNTS:
                condition_rows = [
                    row for row in rows
                    if row["incident"] == incident
                    and row["surge_room"] == surge_room
                    and row["surge_count"] == surge_count
                    and row["remaining_agents"] == 0
                    and not row["error"]
                ]

                if not condition_rows:
                    print(
                        f"{incident:<18} | "
                        f"{surge_room}+{surge_count:<2} | "
                        f"NO SUCCESSFUL RUNS"
                    )
                    continue

                best = min(
                    condition_rows,
                    key=lambda row: row["elapsed_time"],
                )

                condition_best_times.append(
                    best["elapsed_time"]
                )

                print(
                    f"{incident:<18} | "
                    f"{surge_room}+{surge_count:<2} | "
                    f"{best['policy']:<6} | "
                    f"time={best['elapsed_time']:6.2f} | "
                    f"reward={best['episode_reward']:7.4f}"
                )

    # ==============================================================
    # Best policy per incident, averaged over all surge variations
    # ==============================================================

    print(
        "\n===== BEST INCIDENT-ONLY STATIC RESPONSE ====="
    )

    incident_best_means = []

    for incident in INCIDENTS:
        candidates = []

        expected = (
            len(SURGE_ROOMS)
            * len(SURGE_COUNTS)
        )

        for action in range(len(policies)):
            policy_rows = [
                row for row in rows
                if row["action"] == action
                and row["incident"] == incident
            ]

            if len(policy_rows) != expected:
                continue

            if any(
                row["remaining_agents"] != 0
                or row["error"]
                for row in policy_rows
            ):
                continue

            mean_time = sum(
                row["elapsed_time"]
                for row in policy_rows
            ) / len(policy_rows)

            mean_reward = sum(
                row["episode_reward"]
                for row in policy_rows
            ) / len(policy_rows)

            worst_time = max(
                row["elapsed_time"]
                for row in policy_rows
            )

            candidates.append({
                "policy": policy_rows[0]["policy"],
                "mean_time": mean_time,
                "mean_reward": mean_reward,
                "worst_time": worst_time,
            })

        candidates.sort(
            key=lambda row: row["mean_time"]
        )

        if not candidates:
            print(
                f"{incident:<18} | "
                f"NO POLICY COMPLETED ALL SURGE CONDITIONS"
            )
            continue

        best = candidates[0]
        incident_best_means.append(
            best["mean_time"]
        )

        print(
            f"{incident:<18} | "
            f"{best['policy']:<6} | "
            f"mean={best['mean_time']:6.2f} | "
            f"worst={best['worst_time']:6.2f} | "
            f"reward={best['mean_reward']:7.4f}"
        )

    # ==============================================================
    # Sweep completeness
    # ==============================================================

    print("\n===== SWEEP COMPLETENESS =====")

    for incident in INCIDENTS:
        for surge_room in SURGE_ROOMS:
            for surge_count in SURGE_COUNTS:
                condition_rows = [
                    row for row in rows
                    if row["incident"] == incident
                    and row["surge_room"] == surge_room
                    and row["surge_count"] == surge_count
                ]

                errors = sum(
                    bool(row["error"])
                    for row in condition_rows
                )

                timeouts = sum(
                    row["remaining_agents"] > 0
                    and not row["error"]
                    for row in condition_rows
                )

                print(
                    f"{incident:<18} | "
                    f"{surge_room}+{surge_count:<2} | "
                    f"rows={len(condition_rows):2}/"
                    f"{len(policies)} | "
                    f"errors={errors:2} | "
                    f"timeouts={timeouts:2}"
                )

    # ==============================================================
    # Robust static policy across all 36 conditions
    # ==============================================================

    print(
        "\n===== BEST ROBUST STATIC POLICIES ====="
    )

    summaries = []

    expected_rows = (
        len(INCIDENTS)
        * len(SURGE_ROOMS)
        * len(SURGE_COUNTS)
    )

    for action in range(len(policies)):
        policy_rows = [
            row for row in rows
            if row["action"] == action
        ]

        if len(policy_rows) != expected_rows:
            continue

        if any(
            row["remaining_agents"] != 0
            or row["error"]
            for row in policy_rows
        ):
            continue

        mean_time = sum(
            row["elapsed_time"]
            for row in policy_rows
        ) / len(policy_rows)

        mean_reward = sum(
            row["episode_reward"]
            for row in policy_rows
        ) / len(policy_rows)

        worst_time = max(
            row["elapsed_time"]
            for row in policy_rows
        )

        summaries.append({
            "policy": policy_rows[0]["policy"],
            "mean_time": mean_time,
            "mean_reward": mean_reward,
            "worst_time": worst_time,
        })

    summaries.sort(
        key=lambda row: row["mean_time"]
    )

    if not summaries:
        print(
            "No static policy completed all 36 conditions."
        )
        return

    for rank, result in enumerate(
        summaries[:10],
        start=1,
    ):
        print(
            f"{rank:2}. "
            f"{result['policy']:<6} | "
            f"mean={result['mean_time']:6.2f} | "
            f"worst={result['worst_time']:6.2f} | "
            f"reward={result['mean_reward']:7.4f}"
        )

    robust = summaries[0]

    # ==============================================================
    # Overall summary
    # ==============================================================

    print("\n===== STATIC BASELINE SUMMARY =====")

    print(
        f"Best robust fixed policy: "
        f"{robust['policy']} | "
        f"mean={robust['mean_time']:.2f}s | "
        f"worst={robust['worst_time']:.2f}s"
    )

    if len(incident_best_means) == len(INCIDENTS):
        incident_only_mean = (
            sum(incident_best_means)
            / len(incident_best_means)
        )

        print(
            f"Incident-only best response mean: "
            f"{incident_only_mean:.2f}s"
        )

        improvement = (
            robust["mean_time"]
            - incident_only_mean
        )

        improvement_pct = (
            100
            * improvement
            / robust["mean_time"]
        )

        print(
            f"Gain from incident-specific response: "
            f"{improvement:.2f}s "
            f"({improvement_pct:.1f}%)"
        )

    if len(condition_best_times) == num_conditions:
        condition_specific_mean = (
            sum(condition_best_times)
            / len(condition_best_times)
        )

        print(
            f"Full-condition best response mean: "
            f"{condition_specific_mean:.2f}s"
        )

        improvement = (
            robust["mean_time"]
            - condition_specific_mean
        )

        improvement_pct = (
            100
            * improvement
            / robust["mean_time"]
        )

        print(
            f"Gain from full condition knowledge: "
            f"{improvement:.2f}s "
            f"({improvement_pct:.1f}%)"
        )

    else:
        print(
            f"Only "
            f"{len(condition_best_times)}/"
            f"{num_conditions} "
            f"conditions had a successful policy."
        )


if __name__ == "__main__":
    mp.freeze_support()
    main()