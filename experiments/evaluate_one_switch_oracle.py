import csv
import multiprocessing as mp
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from scenarios.building_v2 import BuildingV2Scenario
from envs.evacuation_env import EvacuationEnv


# 80 initially present
INITIAL_COUNTS = {
    "A": 20,
    "B": 20,
    "C": 20,
    "D": 20,
}

SURGE_ROOMS = ["A", "B", "C", "D"]
SURGE_TIMES = [20.0, 30.0]
SURGE_COUNT = 40

PRE_POLICY = "CCA"

SEED = 0
CONTROL_INTERVAL = 10.0
MAX_TIME = 180.0
MAX_WORKERS = 11

OUTPUT_PATH = Path(
    "results/one_switch_oracle_80initial_40surge_seed0.csv"
)


def short_policy(policy):
    return "".join(policy[j] for j in ["J1", "J2", "J3"])


def run_case(
    pre_action,
    post_action,
    pre_policy,
    post_policy,
    surge_room,
    surge_time,
):
    scenario = BuildingV2Scenario(
        room_counts=dict(INITIAL_COUNTS),
        dynamic_events=[{
            "time": surge_time,
            "room": surge_room,
            "count": SURGE_COUNT,
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
        steps = 0

        while not terminated and not truncated:
            # At t=0,10,... before the event → CCA.
            # Once the event boundary has been reached → candidate policy.
            if info["elapsed_time"] < surge_time:
                action = pre_action
            else:
                action = post_action

            obs, reward, terminated, truncated, info = env.step(action)

            episode_reward += reward
            steps += 1

        return {
            "pre_action": pre_action,
            "pre_policy": pre_policy,
            "post_action": post_action,
            "post_policy": post_policy,
            "surge_room": surge_room,
            "surge_time": surge_time,
            "elapsed_time": info["elapsed_time"],
            "remaining_agents": info["remaining_agents"],
            "episode_reward": episode_reward,
            "steps": steps,
            "error": "",
        }

    except Exception as error:
        return {
            "pre_action": pre_action,
            "pre_policy": pre_policy,
            "post_action": post_action,
            "post_policy": post_policy,
            "surge_room": surge_room,
            "surge_time": surge_time,
            "elapsed_time": MAX_TIME,
            "remaining_agents": -1,
            "episode_reward": 0.0,
            "steps": 0,
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

    pre_action = next(
        i for i, policy in enumerate(policies)
        if short_policy(policy) == PRE_POLICY
    )

    env.close()

    jobs = [
        (
            pre_action,
            post_action,
            PRE_POLICY,
            short_policy(post_policy),
            room,
            time,
        )
        for post_action, post_policy in enumerate(policies)
        for room in SURGE_ROOMS
        for time in SURGE_TIMES
    ]

    print(f"Pre-surge policy: {PRE_POLICY}")
    print(f"Post-surge policies: {len(policies)}")
    print(f"Dynamic conditions: {len(SURGE_ROOMS) * len(SURGE_TIMES)}")
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
                f"{result['surge_room']}@{result['surge_time']:.0f} | "
                f"{PRE_POLICY} -> {result['post_policy']:<6} | "
                f"time={result['elapsed_time']:6.2f} | "
                f"remaining={result['remaining_agents']}",
                flush=True,
            )

            if result["error"]:
                print(
                    f"  ERROR: {result['error']}",
                    flush=True,
                )

        print("\nAll simulation jobs returned.", flush=True)

        rows.sort(key=lambda row: (
            row["surge_room"],
            row["surge_time"],
            row["post_action"],
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

        print(
            f"Results saved to {OUTPUT_PATH}",
            flush=True,
        )

    except KeyboardInterrupt:
        print(
            "\nInterrupted. Cancelling pending jobs...",
            flush=True,
        )

        for future in futures:
            future.cancel()

        executor.shutdown(
            wait=False,
            cancel_futures=True,
        )
        return

    print("Shutting down worker processes...", flush=True)
    executor.shutdown(
        wait=True,
        cancel_futures=True,
    )
    print("Worker processes shut down.", flush=True)

    print_summary(rows)


def print_summary(rows):
    print("\n===== ONE-SWITCH ORACLE: EVACUATION TIME =====")

    baseline_times = []
    oracle_times = []

    for room in SURGE_ROOMS:
        for time in SURGE_TIMES:
            condition_rows = [
                row for row in rows
                if row["surge_room"] == room
                and row["surge_time"] == time
                and row["remaining_agents"] == 0
                and not row["error"]
            ]

            if not condition_rows:
                print(f"{room}@{time:>4.0f} | NO SUCCESSFUL RUNS")
                continue

            baseline = next(
                row for row in condition_rows
                if row["post_policy"] == PRE_POLICY
            )

            best = min(
                condition_rows,
                key=lambda row: row["elapsed_time"],
            )

            gain = (
                baseline["elapsed_time"]
                - best["elapsed_time"]
            )

            gain_pct = (
                100 * gain / baseline["elapsed_time"]
            )

            baseline_times.append(
                baseline["elapsed_time"]
            )
            oracle_times.append(
                best["elapsed_time"]
            )

            print(
                f"{room}@{time:>4.0f} | "
                f"{PRE_POLICY}->{best['post_policy']:<6} | "
                f"{baseline['elapsed_time']:6.2f} -> "
                f"{best['elapsed_time']:6.2f} | "
                f"gain={gain:5.2f}s ({gain_pct:4.1f}%)"
            )

    baseline_mean = sum(baseline_times) / len(baseline_times)
    oracle_mean = sum(oracle_times) / len(oracle_times)

    gain = baseline_mean - oracle_mean
    gain_pct = 100 * gain / baseline_mean

    print("\n===== TIME HEADROOM =====")
    print(
        f"CCA fixed mean:             {baseline_mean:.2f}s"
    )
    print(
        f"CCA -> oracle switch mean: {oracle_mean:.2f}s"
    )
    print(
        f"One-switch headroom:       "
        f"{gain:.2f}s ({gain_pct:.1f}%)"
    )

    print("\n===== ONE-SWITCH ORACLE: PPO REWARD =====")

    baseline_rewards = []
    oracle_rewards = []

    for room in SURGE_ROOMS:
        for time in SURGE_TIMES:
            condition_rows = [
                row for row in rows
                if row["surge_room"] == room
                and row["surge_time"] == time
                and row["remaining_agents"] == 0
                and not row["error"]
            ]

            baseline = next(
                row for row in condition_rows
                if row["post_policy"] == PRE_POLICY
            )

            best = max(
                condition_rows,
                key=lambda row: row["episode_reward"],
            )

            baseline_rewards.append(
                baseline["episode_reward"]
            )
            oracle_rewards.append(
                best["episode_reward"]
            )

            print(
                f"{room}@{time:>4.0f} | "
                f"{PRE_POLICY}->{best['post_policy']:<6} | "
                f"{baseline['episode_reward']:7.4f} -> "
                f"{best['episode_reward']:7.4f}"
            )

    baseline_reward = (
        sum(baseline_rewards) / len(baseline_rewards)
    )
    oracle_reward = (
        sum(oracle_rewards) / len(oracle_rewards)
    )

    baseline_cost = -baseline_reward
    oracle_cost = -oracle_reward

    reduction = baseline_cost - oracle_cost
    reduction_pct = (
        100 * reduction / baseline_cost
    )

    print("\n===== REWARD HEADROOM =====")
    print(
        f"CCA fixed mean reward:      {baseline_reward:.4f}"
    )
    print(
        f"Oracle-switch mean reward: {oracle_reward:.4f}"
    )
    print(
        f"Reward-cost reduction:     "
        f"{reduction:.4f} ({reduction_pct:.1f}%)"
    )


if __name__ == "__main__":
    mp.freeze_support()
    main()