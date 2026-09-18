import csv
import itertools
import os
import time

from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from scenarios.building_v2 import BuildingV2Scenario
from simulation.jupedsim_backend import JuPedSimBackend


# ============================================================
# Experiment configuration
# ============================================================
POPULATION_SCENARIOS = {
    "balanced": {
        "A": 25, "B": 25, "C": 25, "D": 25
    },

    "A_extreme": {
        "A": 49, "B": 17, "C": 17, "D": 17
    },

    "B_extreme": {
        "A": 19, "B": 42, "C": 20, "D": 19
    },

    "C_extreme": {
        "A": 19, "B": 20, "C": 42, "D": 19
    },

    "D_extreme": {
        "A": 22, "B": 21, "C": 21, "D": 36
    },
}
SEED = 1
MAX_TIME = 180.0
CONTROL_INTERVAL = 0.5

try:
    AVAILABLE_CPUS = len(os.sched_getaffinity(0))
except AttributeError:
    AVAILABLE_CPUS = os.cpu_count() or 1

MAX_WORKERS = min(11, AVAILABLE_CPUS)


# ============================================================
# Policy generation
# ============================================================

def generate_all_policies(scenario):
    junction_names = ["J1", "J2", "J3"]

    choices = [
        scenario.guidance_choices[junction]
        for junction in junction_names
    ]

    for combination in itertools.product(*choices):
        yield dict(zip(junction_names, combination))


# ============================================================
# Cycle detection
# ============================================================

def trace_from_junction(start_junction, policy):
    visited_order = []
    visited_set = set()

    current = start_junction

    while current in policy:
        if current in visited_set:
            cycle_start = visited_order.index(current)
            cycle = tuple(visited_order[cycle_start:])

            return {
                "reaches_exit": False,
                "cycle": cycle,
            }

        visited_order.append(current)
        visited_set.add(current)

        current = policy[current]

    return {
        "reaches_exit": True,
        "cycle": None,
    }


def analyse_cycles(policy, room_counts, initial_targets):
    trapped_rooms = []
    trapped_agents = 0
    cycles = set()

    for room_name, count in room_counts.items():
        if count == 0:
            continue

        initial_target = initial_targets[room_name]

        # Example: Room D currently goes directly to Exit B.
        if initial_target not in policy:
            continue

        trace = trace_from_junction(initial_target, policy)

        if not trace["reaches_exit"]:
            trapped_rooms.append(room_name)
            trapped_agents += count
            cycles.add(tuple(sorted(trace["cycle"])))

    return {
        "cyclic": trapped_agents > 0,
        "trapped_rooms": tuple(trapped_rooms),
        "trapped_agents": trapped_agents,
        "cycles": tuple(sorted(cycles)),
    }


# ============================================================
# One JuPedSim experiment
# ============================================================

def run_policy(
    scenario_name,
    policy,
    room_counts,
    seed,
    max_time,
    control_interval,
):
    """
    Run one population scenario + one fixed routing policy.

    This function is executed inside a worker process.
    """

    scenario = BuildingV2Scenario(room_counts=room_counts)

    sim = JuPedSimBackend(
        scenario=scenario,
        record=False,
    )

    try:
        sim.reset(seed=seed)
        sim.apply_guidance(policy)

        while not sim.is_evacuated and sim.elapsed_time < max_time:
            sim.advance(control_interval)

        return {
            "scenario": scenario_name,

            "A": room_counts["A"],
            "B": room_counts["B"],
            "C": room_counts["C"],
            "D": room_counts["D"],

            "J1": policy["J1"],
            "J2": policy["J2"],
            "J3": policy["J3"],

            "status": "PASS" if sim.is_evacuated else "TIMEOUT",
            "evacuation_time": sim.elapsed_time,
            "remaining_agents": sim.remaining_agents,

            "cyclic": False,
            "trapped_agents": 0,
            "trapped_rooms": "",
            "cycles": "",
        }

    finally:
        sim.close()


# ============================================================
# Formatting / sorting
# ============================================================

def policy_string(result):
    return (
        f"J1={result['J1']:<2} "
        f"J2={result['J2']:<2} "
        f"J3={result['J3']:<2}"
    )


def result_sort_key(result):
    if result["status"] == "PASS":
        return (0, result["evacuation_time"], 0)

    if result["status"] == "TIMEOUT":
        return (1, result["remaining_agents"], result["evacuation_time"])

    return (2, result["remaining_agents"], 0)


# ============================================================
# Main experiment
# ============================================================

def main():
    start_wall_time = time.perf_counter()

    reference_scenario = BuildingV2Scenario(
        room_counts=POPULATION_SCENARIOS["balanced"]
    )

    policies = list(generate_all_policies(reference_scenario))

    print()
    print("===== V2 POPULATION SWEEP =====")
    print()

    print(f"Population scenarios: {len(POPULATION_SCENARIOS)}")
    print(f"Policies per scenario: {len(policies)}")
    print(
        f"Total combinations: "
        f"{len(POPULATION_SCENARIOS) * len(policies)}"
    )
    print(f"Seed: {SEED}")
    print(f"Max simulation time: {MAX_TIME:.0f} s")
    print(f"Worker processes: {MAX_WORKERS}")
    print()

    # ========================================================
    # Pre-screen cycles and prepare jobs
    # ========================================================

    results = []
    jobs = []

    for scenario_name, room_counts in POPULATION_SCENARIOS.items():
        print(f"Preparing {scenario_name}: {room_counts}")

        scenario = BuildingV2Scenario(room_counts=room_counts)

        for policy in policies:
            cycle_info = analyse_cycles(
                policy,
                room_counts,
                scenario.initial_targets,
            )

            if cycle_info["cyclic"]:
                result = {
                    "scenario": scenario_name,

                    "A": room_counts["A"],
                    "B": room_counts["B"],
                    "C": room_counts["C"],
                    "D": room_counts["D"],

                    "J1": policy["J1"],
                    "J2": policy["J2"],
                    "J3": policy["J3"],

                    "status": "CYCLIC",
                    "evacuation_time": None,
                    "remaining_agents": cycle_info["trapped_agents"],

                    "cyclic": True,
                    "trapped_agents": cycle_info["trapped_agents"],
                    "trapped_rooms": ",".join(
                        cycle_info["trapped_rooms"]
                    ),
                    "cycles": str(cycle_info["cycles"]),
                }

                results.append(result)

            else:
                jobs.append(
                    (scenario_name, policy, room_counts)
                )

    print()
    print(f"Cyclic combinations skipped: {len(results)}")
    print(f"JuPedSim runs required: {len(jobs)}")
    print()

    # ========================================================
    # Parallel simulation
    # ========================================================

    completed = 0

    with ProcessPoolExecutor(max_workers=MAX_WORKERS) as executor:
        future_to_job = {}

        for scenario_name, policy, room_counts in jobs:
            future = executor.submit(
                run_policy,
                scenario_name,
                policy,
                room_counts,
                SEED,
                MAX_TIME,
                CONTROL_INTERVAL,
            )

            future_to_job[future] = (scenario_name, policy)

        for future in as_completed(future_to_job):
            result = future.result()
            results.append(result)

            completed += 1

            print(
                f"[{completed:03d}/{len(jobs):03d}] "
                f"{result['scenario']:<10} "
                f"| {policy_string(result)} "
                f"| {result['evacuation_time']:6.2f} s "
                f"| remaining={result['remaining_agents']:3d} "
                f"| {result['status']}"
            )

    # ========================================================
    # Per-scenario summaries
    # ========================================================

    print()
    print("========================================")
    print("      BEST POLICY BY CROWD STATE")
    print("========================================")
    print()

    best_results = []

    for scenario_name, room_counts in POPULATION_SCENARIOS.items():
        scenario_results = [
            result
            for result in results
            if result["scenario"] == scenario_name
        ]

        scenario_results.sort(key=result_sort_key)

        successful = [
            result
            for result in scenario_results
            if result["status"] == "PASS"
        ]

        timeouts = [
            result
            for result in scenario_results
            if result["status"] == "TIMEOUT"
        ]

        cyclic = [
            result
            for result in scenario_results
            if result["status"] == "CYCLIC"
        ]

        print(scenario_name)
        print(f"  Population: {room_counts}")
        print(
            f"  PASS={len(successful)} "
            f"TIMEOUT={len(timeouts)} "
            f"CYCLIC={len(cyclic)}"
        )

        if successful:
            best = successful[0]
            best_results.append(best)

            print(
                f"  BEST: {policy_string(best)} "
                f"| {best['evacuation_time']:.2f} s"
            )

            print("  Top 5:")

            for rank, result in enumerate(successful[:5], start=1):
                print(
                    f"    {rank}. {policy_string(result)} "
                    f"| {result['evacuation_time']:.2f} s"
                )

        else:
            print("  No policy completed evacuation.")

        print()

    # ========================================================
    # Save complete CSV
    # ========================================================

    output_dir = Path("results")
    output_dir.mkdir(exist_ok=True)

    scenario_order = {
        name: index
        for index, name in enumerate(POPULATION_SCENARIOS)
    }

    results.sort(
        key=lambda result: (
            scenario_order[result["scenario"]],
            result_sort_key(result),
        )
    )

    output_path = (
        output_dir
        / "v2_population_sweep_seed1.csv"
    )

    fieldnames = [
        "scenario",
        "A",
        "B",
        "C",
        "D",
        "J1",
        "J2",
        "J3",
        "status",
        "evacuation_time",
        "remaining_agents",
        "cyclic",
        "trapped_agents",
        "trapped_rooms",
        "cycles",
    ]

    with output_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    # ========================================================
    # Save best-policy summary
    # ========================================================

    summary_path = (
        output_dir
        / "v2_population_best_seed1.csv"
    )

    summary_fields = [
        "scenario",
        "A",
        "B",
        "C",
        "D",
        "J1",
        "J2",
        "J3",
        "evacuation_time",
    ]

    with summary_path.open("w", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=summary_fields,
            extrasaction="ignore",
        )

        writer.writeheader()
        writer.writerows(best_results)

    # ========================================================
    # Wall-clock timing
    # ========================================================

    wall_time = time.perf_counter() - start_wall_time

    print("========================================")
    print(f"Complete results: {output_path}")
    print(f"Best-policy summary: {summary_path}")
    print(
        f"Total wall-clock time: "
        f"{wall_time:.1f} s "
        f"({wall_time / 60:.1f} min)"
    )


if __name__ == "__main__":
    main()