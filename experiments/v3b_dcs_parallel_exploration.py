"""Complementary V3-B exploration for the DCS machine.

This driver deliberately leaves the production scenario and environment code
unchanged.  Custom hard cuts are installed only on per-run scenario instances.

Phases are independently resumable, for example::

    .venv/bin/python experiments/v3b_dcs_parallel_exploration.py safety
    .venv/bin/python experiments/v3b_dcs_parallel_exploration.py robustness
    .venv/bin/python experiments/v3b_dcs_parallel_exploration.py trace
    .venv/bin/python experiments/v3b_dcs_parallel_exploration.py fourth-safety
    .venv/bin/python experiments/v3b_dcs_parallel_exploration.py fourth-screen \
        upper_x_10p50 upper_x_17p50

No PPO training is performed.
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
from collections import Counter
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

from shapely import Point
from shapely.geometry import box
from shapely.ops import unary_union

# Permit the documented ``python experiments/...py`` invocation from the
# repository root, matching the style of the older experiment drivers.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from envs.evacuation_env import EvacuationEnv
from scenarios.building_v2 import BuildingV2Scenario
from simulation.jupedsim_backend import JuPedSimBackend


RESULTS = Path("results")
FULL_SWEEP_PATH = RESULTS / (
    "incident_one_switch_A40_B20_C20_D20_Csurge40_t10_seed0.csv"
)
INITIAL_COUNTS = {"A": 40, "B": 20, "C": 20, "D": 20}
SURGE_ROOM = "C"
SURGE_TIME = 10.0
SURGE_COUNT = 40
CONTROL_INTERVAL = 10.0
MAX_TIME = 180.0
MAX_WORKERS = 11
PRE_POLICY = "CCA"
SEEDS = (0, 1, 2, 3, 4)
AGENT_RADIUS = 0.2
NEAR_DISTANCE = 0.4

LOWER_CUT_XS = (6.5, 7.0, 7.5, 8.0, 8.5, 9.0, 9.5, 10.0, 10.5, 11.0)

# Full-width 0.05 m hard cuts.  These are exploratory definitions, not changes
# to BuildingV2Scenario.
FOURTH_CUTS = {
    **{
        f"upper_x_{x:.2f}".replace(".", "p"): (x, 23.0, x + 0.05, 27.0)
        for x in (6.5, 7.5, 8.5, 9.5, 10.5, 13.5, 14.5, 15.5,
                  16.0, 16.5, 17.5, 20.5, 21.5, 22.5)
    },
    **{
        f"left_connector_y_{y:.2f}".replace(".", "p"): (4.0, y, 6.0, y + 0.05)
        for y in (19.5, 20.5, 21.5, 22.5)
    },
    **{
        f"right_connector_y_{y:.2f}".replace(".", "p"): (18.0, y, 20.0, y + 0.05)
        for y in (19.5, 20.5, 21.5, 22.5)
    },
    **{
        f"room_A_connector_y_{y:.2f}".replace(".", "p"): (11.0, y, 13.0, y + 0.05)
        for y in (27.5, 28.0, 28.5, 29.0, 29.5)
    },
}

# Partial capacity reductions leave a central opening and therefore preserve
# connectivity.  Each value is the collection of side slabs removed from the
# base geometry.  Names encode the remaining opening width in metres.
FOURTH_CUTS.update({
    "upper_left_narrow_0p8": (
        (6.25, 23.15, 10.75, 24.60),
        (6.25, 25.40, 10.75, 26.85),
    ),
    "upper_left_narrow_1p2": (
        (6.25, 23.15, 10.75, 24.40),
        (6.25, 25.60, 10.75, 26.85),
    ),
    "upper_right_narrow_0p8": (
        (13.25, 23.15, 17.75, 24.60),
        (13.25, 25.40, 17.75, 26.85),
    ),
    "upper_right_narrow_1p2": (
        (13.25, 23.15, 17.75, 24.40),
        (13.25, 25.60, 17.75, 26.85),
    ),
    "left_connector_narrow_0p8": (
        (4.15, 19.15, 4.60, 22.85),
        (5.40, 19.15, 5.85, 22.85),
    ),
    "right_connector_narrow_0p8": (
        (18.15, 19.15, 18.60, 22.85),
        (19.40, 19.15, 19.85, 22.85),
    ),
})

ROBUSTNESS_POLICIES = {
    "B_route_08": ("CJ3A", "AJ3B", "ACB", "CCA", "ACA"),
    "C_exit": ("AJ3B", "CJ3A", "ACB", "CCA", "ACA"),
    "bottom_loop_left": ("ACB", "ACA", "ACJ1", "CCA", "AJ3B", "CJ3A"),
}

# Eight deliberately paired policies: four different J1 choices while J2->C,
# plus A/C variants for the two J2->J3 exit routes.
FOURTH_SCREEN_POLICIES = (
    "ACB", "CCB", "J2CB", "J3CB", "CCA", "AJ3B", "CJ3B", "CJ3A",
)


def policy_label(policy):
    return "".join(policy[j] for j in ("J1", "J2", "J3"))


def policies_and_labels():
    scenario = BuildingV2Scenario(room_counts=dict(INITIAL_COUNTS))
    env = EvacuationEnv(scenario, control_interval=CONTROL_INTERVAL, max_time=MAX_TIME)
    try:
        policies = [dict(policy) for policy in env.guidance_plans]
    finally:
        env.close()
    return policies, [policy_label(policy) for policy in policies]


def write_csv(path, rows, fieldnames=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = list(rows)
    if not rows and fieldnames is None:
        raise ValueError(f"Cannot infer fields for empty CSV {path}")
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def run_parallel(function, jobs, description):
    rows = []
    context = mp.get_context("spawn")
    with ProcessPoolExecutor(max_workers=MAX_WORKERS, mp_context=context) as executor:
        futures = {executor.submit(function, *job): job for job in jobs}
        total = len(futures)
        for completed, future in enumerate(as_completed(futures), 1):
            result = future.result()
            rows.append(result)
            if completed == 1 or completed % 10 == 0 or completed == total:
                print(f"{description}: {completed}/{total}", flush=True)
    return rows


def pre_event_snapshot(policy, seed):
    scenario = BuildingV2Scenario(room_counts=dict(INITIAL_COUNTS))
    backend = JuPedSimBackend(scenario)
    try:
        backend.reset(seed=seed)
        backend.apply_guidance(policy)
        backend.advance(SURGE_TIME)
        return backend.elapsed_time, [
            {
                "agent_id": agent.id,
                "origin": backend.agent_origins[agent.id],
                "x": agent.position[0],
                "y": agent.position[1],
                "assignment": agent.assignment,
            }
            for agent in backend.get_state().agents
        ]
    finally:
        backend.close()


def removal_geometry(spec):
    if len(spec) == 4 and all(isinstance(value, (int, float)) for value in spec):
        return box(*spec)
    return unary_union([box(*bounds) for bounds in spec])


def attempt_switch(policy, seed, bounds):
    scenario = BuildingV2Scenario(room_counts=dict(INITIAL_COUNTS))
    backend = JuPedSimBackend(scenario)
    try:
        backend.reset(seed=seed)
        backend.apply_guidance(policy)
        backend.advance(SURGE_TIME)
        backend.simulation.switch_geometry(
            scenario.geometry.difference(removal_geometry(bounds))
        )
        # Exercise route recomputation and junction handling, rather than only
        # accepting a successful return from switch_geometry.
        backend.advance(0.2)
        return ""
    except Exception as error:  # retained verbatim in the raw audit
        return f"{type(error).__name__}: {error}"
    finally:
        backend.close()


def safety_case(policy_index, policy, label, seed, cuts):
    elapsed, agents = pre_event_snapshot(policy, seed)
    summaries = []
    nearby_rows = []
    for cut_name, bounds in cuts:
        cut = removal_geometry(bounds)
        point_hits = []
        body_hits = []
        nearby = []
        for agent in agents:
            distance = cut.distance(Point(agent["x"], agent["y"]))
            item = {**agent, "distance_to_cut": distance}
            if distance == 0.0:
                point_hits.append(item)
            if distance <= AGENT_RADIUS:
                body_hits.append(item)
            if distance <= NEAR_DISTANCE:
                nearby.append(item)
                nearby_rows.append({
                    "cut": cut_name,
                    "bounds": repr(bounds),
                    "policy_action": policy_index,
                    "pre_policy": label,
                    "seed": seed,
                    **item,
                    "center_inside_strip": int(distance == 0.0),
                    "body_intersects_strip": int(distance <= AGENT_RADIUS),
                })
        switch_error = attempt_switch(policy, seed, bounds)
        summaries.append({
            "cut": cut_name,
            "bounds": repr(bounds),
            "policy_action": policy_index,
            "pre_policy": label,
            "seed": seed,
            "snapshot_time": elapsed,
            "agents_present": len(agents),
            "center_hits": len(point_hits),
            "body_intersections_r0p2": len(body_hits),
            "nearby_within_0p4": len(nearby),
            "center_hit_ids": "|".join(str(row["agent_id"]) for row in point_hits),
            "body_hit_ids": "|".join(str(row["agent_id"]) for row in body_hits),
            "nearby_ids": "|".join(str(row["agent_id"]) for row in nearby),
            "switch_error": switch_error,
        })
    return {"summaries": summaries, "nearby": nearby_rows}


def position_only_safety_case(policy_index, policy, label, seed, cuts):
    """Fine-grained occupancy scan without thousands of redundant switches."""
    elapsed, agents = pre_event_snapshot(policy, seed)
    summaries = []
    for cut_name, bounds in cuts:
        cut = removal_geometry(bounds)
        distances = [cut.distance(Point(agent["x"], agent["y"])) for agent in agents]
        summaries.append({
            "cut": cut_name,
            "bounds": repr(bounds),
            "policy_action": policy_index,
            "pre_policy": label,
            "seed": seed,
            "snapshot_time": elapsed,
            "agents_present": len(agents),
            "center_hits": sum(distance == 0.0 for distance in distances),
            "body_intersections_r0p2": sum(distance <= AGENT_RADIUS for distance in distances),
            "nearby_within_0p4": sum(distance <= NEAR_DISTANCE for distance in distances),
            "minimum_center_distance": min(distances, default=float("inf")),
        })
    return summaries


def run_safety():
    policies, labels = policies_and_labels()
    cuts = [
        (f"lower_x_{x:.2f}".replace(".", "p"), (x, 17.0, x + 0.05, 19.0))
        for x in LOWER_CUT_XS
    ]
    jobs = [
        (index, policy, labels[index], seed, cuts)
        for index, policy in enumerate(policies)
        for seed in SEEDS
    ]
    bundles = run_parallel(safety_case, jobs, "bottom-loop safety cases")
    summaries = [row for bundle in bundles for row in bundle["summaries"]]
    nearby = [row for bundle in bundles for row in bundle["nearby"]]
    summaries.sort(key=lambda row: (row["cut"], row["pre_policy"], row["seed"]))
    nearby.sort(key=lambda row: (row["cut"], row["pre_policy"], row["seed"], row["agent_id"]))
    write_csv(RESULTS / "v3b_bottom_loop_left_safety_cases.csv", summaries)
    write_csv(
        RESULTS / "v3b_bottom_loop_left_safety_nearby_agents.csv",
        nearby,
        fieldnames=(
            "cut", "bounds", "policy_action", "pre_policy", "seed", "agent_id",
            "origin", "x", "y", "assignment", "distance_to_cut",
            "center_inside_strip", "body_intersects_strip",
        ),
    )
    print_safety_summary(summaries)


def run_fine_lower_scan():
    policies, labels = policies_and_labels()
    # 6.05--11.70 remains outside both junction polygons and scans at the
    # thickness of the cut.  It is an occupancy screen; shortlisted positions
    # must still pass a real switch test.
    xs = [round(6.05 + 0.05 * index, 2) for index in range(114)]
    cuts = [
        (f"lower_x_{x:.2f}".replace(".", "p"), (x, 17.0, x + 0.05, 19.0))
        for x in xs
    ]
    jobs = [
        (index, policy, labels[index], seed, cuts)
        for index, policy in enumerate(policies)
        for seed in SEEDS
    ]
    bundles = run_parallel(position_only_safety_case, jobs, "fine lower occupancy scan")
    rows = [row for bundle in bundles for row in bundle]
    rows.sort(key=lambda row: (row["cut"], row["pre_policy"], row["seed"]))
    write_csv(RESULTS / "v3b_bottom_loop_left_fine_occupancy_scan.csv", rows)
    ranked = []
    for cut in sorted({row["cut"] for row in rows}):
        selected = [row for row in rows if row["cut"] == cut]
        ranked.append((
            sum(row["body_intersections_r0p2"] for row in selected),
            sum(row["center_hits"] for row in selected),
            sum(row["nearby_within_0p4"] for row in selected),
            min(row["minimum_center_distance"] for row in selected),
            cut,
        ))
    print("Best fine-scan positions by body intersections:")
    for body, center, nearby, minimum, cut in sorted(ranked)[:15]:
        print(
            f"{cut}: body={body}, center={center}, nearby={nearby}, "
            f"minimum_distance={minimum:.4f}"
        )


def run_selected_lower_switch(xs):
    policies, labels = policies_and_labels()
    cuts = [
        (f"lower_x_{x:.2f}".replace(".", "p"), (x, 17.0, x + 0.05, 19.0))
        for x in xs
    ]
    jobs = [
        (index, policy, labels[index], seed, cuts)
        for index, policy in enumerate(policies)
        for seed in SEEDS
    ]
    bundles = run_parallel(safety_case, jobs, "selected lower switch audit")
    summaries = [row for bundle in bundles for row in bundle["summaries"]]
    nearby = [row for bundle in bundles for row in bundle["nearby"]]
    write_csv(RESULTS / "v3b_bottom_loop_left_selected_switch_cases.csv", summaries)
    write_csv(
        RESULTS / "v3b_bottom_loop_left_selected_switch_nearby_agents.csv",
        nearby,
        fieldnames=(
            "cut", "bounds", "policy_action", "pre_policy", "seed", "agent_id",
            "origin", "x", "y", "assignment", "distance_to_cut",
            "center_inside_strip", "body_intersects_strip",
        ),
    )
    print_safety_summary(summaries)


def print_safety_summary(rows):
    print("cut,cases_with_center_hits,total_center_hits,cases_with_body_hits,"
          "total_body_hits,total_nearby,switch_errors")
    for cut in sorted({row["cut"] for row in rows}):
        selected = [row for row in rows if row["cut"] == cut]
        print(
            f"{cut},{sum(row['center_hits'] > 0 for row in selected)},"
            f"{sum(row['center_hits'] for row in selected)},"
            f"{sum(row['body_intersections_r0p2'] > 0 for row in selected)},"
            f"{sum(row['body_intersections_r0p2'] for row in selected)},"
            f"{sum(row['nearby_within_0p4'] for row in selected)},"
            f"{sum(bool(row['switch_error']) for row in selected)}"
        )


def install_incident(scenario, incident, custom_bounds=None):
    if custom_bounds is not None:
        scenario.incident_geometries[incident] = scenario.geometry.difference(
            removal_geometry(custom_bounds)
        )


def episode_case(incident, post_label, seed, custom_bounds=None):
    policies, labels = policies_and_labels()
    pre_action = labels.index(PRE_POLICY)
    post_action = labels.index(post_label)
    scenario = BuildingV2Scenario(
        room_counts=dict(INITIAL_COUNTS),
        dynamic_events=[{
            "time": SURGE_TIME,
            "room": SURGE_ROOM,
            "count": SURGE_COUNT,
            "incident": incident,
        }],
    )
    install_incident(scenario, incident, custom_bounds)
    env = EvacuationEnv(scenario, control_interval=CONTROL_INTERVAL, max_time=MAX_TIME)
    try:
        _, info = env.reset(seed=seed)
        reward_total = 0.0
        step_number = 0
        terminated = truncated = False
        while not terminated and not truncated:
            action = pre_action if step_number == 0 else post_action
            _, reward, terminated, truncated, info = env.step(action)
            reward_total += reward
            step_number += 1
        return {
            "incident": incident,
            "bounds": "" if custom_bounds is None else repr(custom_bounds),
            "pre_policy": PRE_POLICY,
            "post_policy": post_label,
            "post_action": post_action,
            "seed": seed,
            "elapsed_time": info["elapsed_time"],
            "remaining_agents": info["remaining_agents"],
            "episode_reward": reward_total,
            "steps": step_number,
            "error": "",
        }
    except Exception as error:
        return {
            "incident": incident,
            "bounds": "" if custom_bounds is None else repr(custom_bounds),
            "pre_policy": PRE_POLICY,
            "post_policy": post_label,
            "post_action": post_action,
            "seed": seed,
            "elapsed_time": MAX_TIME,
            "remaining_agents": -1,
            "episode_reward": 0.0,
            "steps": 0,
            "error": f"{type(error).__name__}: {error}",
        }
    finally:
        env.close()


def run_robustness():
    jobs = [
        (incident, label, seed)
        for incident, labels in ROBUSTNESS_POLICIES.items()
        for label in labels
        for seed in SEEDS
    ]
    rows = run_parallel(episode_case, jobs, "three-class robustness")
    rows.sort(key=lambda row: (row["incident"], row["post_policy"], row["seed"]))
    write_csv(RESULTS / "v3b_three_class_multiseed.csv", rows)
    print("incident,policy,mean_time,mean_reward,errors,timeouts")
    for incident, labels in ROBUSTNESS_POLICIES.items():
        for label in labels:
            chosen = [row for row in rows if row["incident"] == incident and row["post_policy"] == label]
            print(
                f"{incident},{label},"
                f"{sum(r['elapsed_time'] for r in chosen)/len(chosen):.3f},"
                f"{sum(r['episode_reward'] for r in chosen)/len(chosen):.6f},"
                f"{sum(bool(r['error']) for r in chosen)},"
                f"{sum(r['remaining_agents'] > 0 for r in chosen)}"
            )


def run_lower_alternative():
    incident = "bottom_loop_left_x8p40"
    bounds = (8.40, 17.0, 8.45, 19.0)
    jobs = [
        (incident, label, seed, bounds)
        for label in ROBUSTNESS_POLICIES["bottom_loop_left"]
        for seed in SEEDS
    ]
    rows = run_parallel(episode_case, jobs, "x=8.40 lower-cut validation")
    rows.sort(key=lambda row: (row["post_policy"], row["seed"]))
    write_csv(RESULTS / "v3b_bottom_loop_left_x8p40_multiseed.csv", rows)
    print("policy,mean_time,mean_reward,errors,timeouts")
    for label in ROBUSTNESS_POLICIES["bottom_loop_left"]:
        chosen = [row for row in rows if row["post_policy"] == label]
        print(
            f"{label},{sum(r['elapsed_time'] for r in chosen)/len(chosen):.3f},"
            f"{sum(r['episode_reward'] for r in chosen)/len(chosen):.6f},"
            f"{sum(bool(r['error']) for r in chosen)},"
            f"{sum(r['remaining_agents'] > 0 for r in chosen)}"
        )


def nominal_c_route(policy):
    current = "J2"
    route = [current]
    while current in policy:
        current = policy[current]
        route.append(current)
    return ">".join(route)


def write_policy_classes():
    policies, labels = policies_and_labels()
    rows = []
    for action, (policy, label) in enumerate(zip(policies, labels)):
        route = nominal_c_route(policy)
        rows.append({
            "action": action,
            "policy": label,
            "J1": policy["J1"],
            "J2": policy["J2"],
            "J3": policy["J3"],
            "nominal_C_route": route,
            "nominal_C_exit": route.split(">")[-1],
        })
    write_csv(RESULTS / "v3b_policy_routing_classes.csv", rows)


def write_full35_class_summary():
    """Summarise an imported 3x35 seed-0 sweep without rerunning it."""
    if not FULL_SWEEP_PATH.exists():
        raise FileNotFoundError(
            f"The laptop full sweep has not been imported: {FULL_SWEEP_PATH}"
        )

    policies, labels = policies_and_labels()
    route_by_label = {
        label: nominal_c_route(policy)
        for policy, label in zip(policies, labels)
    }
    with FULL_SWEEP_PATH.open(newline="") as handle:
        source_rows = list(csv.DictReader(handle))

    expected_incidents = set(ROBUSTNESS_POLICIES)
    imported_incidents = {row["incident"] for row in source_rows}
    if imported_incidents != expected_incidents:
        raise ValueError(
            "Expected the three-incident laptop sweep; found "
            f"{sorted(imported_incidents)}"
        )
    counts = Counter(row["incident"] for row in source_rows)
    if any(counts[incident] != 35 for incident in expected_incidents):
        raise ValueError(f"Expected 35 actions per incident; found {dict(counts)}")
    if {row["seed"] for row in source_rows} != {"0"}:
        raise ValueError("Expected a seed-0-only imported sweep")

    grouped = {}
    for row in source_rows:
        label = row["post_policy"]
        key = (row["incident"], route_by_label[label])
        grouped.setdefault(key, []).append(row)

    summaries = []
    for (incident, route_class), rows in sorted(grouped.items()):
        best_time = min(rows, key=lambda row: float(row["elapsed_time"]))
        best_reward = max(rows, key=lambda row: float(row["episode_reward"]))
        summaries.append({
            "incident": incident,
            "nominal_C_route_class": route_class,
            "policies_in_class": len(rows),
            "successful_policies": sum(
                not row["error"] and int(row["remaining_agents"]) == 0
                for row in rows
            ),
            "best_time_policy": best_time["post_policy"],
            "best_time": best_time["elapsed_time"],
            "best_reward_policy": best_reward["post_policy"],
            "best_reward": best_reward["episode_reward"],
        })

    write_csv(RESULTS / "v3b_full35_seed0_policy_class_summary.csv", summaries)
    print(f"Summarised {len(source_rows)} imported rows into {len(summaries)} classes")


def locate_zone(scenario, position):
    point = Point(position)
    for name in ("J1", "J2", "J3"):
        if scenario.junctions[name].covers(point):
            return name
    for name, region in scenario.rooms.items():
        if region.covers(point):
            return f"room_{name}"
    named = (
        ("room_A_connector", scenario.room_A_connector),
        ("room_B_connector", scenario.room_B_connector),
        ("room_C_connector", scenario.room_C_connector),
        ("room_D_connector", scenario.room_D_connector),
        ("left_connector", scenario.left_loop_connector),
        ("right_connector", scenario.right_loop_connector),
        ("lower_west_of_cut", box(4.0, 17.0, 8.5, 19.0)),
        ("lower_east_of_cut", box(8.55, 17.0, 20.0, 19.0)),
        ("upper_west_of_J1", box(0.0, 23.0, 11.0, 27.0)),
        ("upper_east_of_J1", box(13.0, 23.0, 24.0, 27.0)),
        ("exit_A_neck", scenario.exit_A_neck),
        ("exit_C_neck", scenario.exit_C_neck),
        ("south_exit_corridor", scenario.south_exit_corridor),
    )
    for name, region in named:
        if region.covers(point):
            return name
    return "other"


def trace_case(post_label, seed=0, sample_interval=0.2):
    policies, labels = policies_and_labels()
    pre_policy = policies[labels.index(PRE_POLICY)]
    post_policy = policies[labels.index(post_label)]
    scenario = BuildingV2Scenario(
        room_counts=dict(INITIAL_COUNTS),
        dynamic_events=[{
            "time": SURGE_TIME,
            "room": SURGE_ROOM,
            "count": SURGE_COUNT,
            "incident": "bottom_loop_left",
        }],
    )
    backend = JuPedSimBackend(scenario)
    samples = []
    events = []
    try:
        backend.reset(seed=seed)
        initial_ids = {agent.id for agent in backend.get_state().agents}
        backend.apply_guidance(pre_policy)
        backend.advance(SURGE_TIME)
        surge_ids = {
            agent.id for agent in backend.get_state().agents
            if agent.id not in initial_ids
        }
        if len(surge_ids) != SURGE_COUNT:
            raise RuntimeError(f"Expected {SURGE_COUNT} surge agents; got {len(surge_ids)}")
        backend.apply_guidance(post_policy)
        previous = {}
        while backend.elapsed_time < MAX_TIME and any(
            agent.id in surge_ids for agent in backend.get_state().agents
        ):
            state = backend.get_state()
            counts = Counter()
            live_ids = set()
            for agent in state.agents:
                if agent.id not in surge_ids:
                    continue
                live_ids.add(agent.id)
                zone = locate_zone(scenario, agent.position)
                counts[f"zone:{zone}"] += 1
                counts[f"assignment:{agent.assignment}"] += 1
                key = (zone, agent.assignment)
                if previous.get(agent.id) != key:
                    events.append({
                        "post_policy": post_label,
                        "seed": seed,
                        "time": state.elapsed_time,
                        "agent_id": agent.id,
                        "origin": backend.agent_origins[agent.id],
                        "x": agent.position[0],
                        "y": agent.position[1],
                        "zone": zone,
                        "assignment": agent.assignment,
                        "previous_zone": previous.get(agent.id, ("", ""))[0],
                        "previous_assignment": previous.get(agent.id, ("", ""))[1],
                    })
                    previous[agent.id] = key
            samples.append({
                "post_policy": post_label,
                "seed": seed,
                "time": state.elapsed_time,
                "surge_remaining": len(live_ids),
                **{key: counts[key] for key in sorted(counts)},
            })
            backend.advance(sample_interval)
        return {"samples": samples, "events": events}
    finally:
        backend.close()


def run_trace():
    bundles = [trace_case(label) for label in ("ACB", "AJ3B", "CJ3A")]
    samples = [row for bundle in bundles for row in bundle["samples"]]
    events = [row for bundle in bundles for row in bundle["events"]]
    sample_fields = ["post_policy", "seed", "time", "surge_remaining"] + sorted({
        key for row in samples for key in row if key not in {"post_policy", "seed", "time", "surge_remaining"}
    })
    for row in samples:
        for field in sample_fields:
            row.setdefault(field, 0)
    write_csv(RESULTS / "v3b_bottom_loop_left_surge_trace.csv", samples, sample_fields)
    write_csv(RESULTS / "v3b_bottom_loop_left_surge_agent_events.csv", events)
    print(f"Wrote {len(samples)} aggregate samples and {len(events)} agent transition events")


def run_fourth_safety():
    policies, labels = policies_and_labels()
    cuts = sorted(FOURTH_CUTS.items())
    jobs = [
        (index, policy, labels[index], seed, cuts)
        for index, policy in enumerate(policies)
        for seed in SEEDS
    ]
    bundles = run_parallel(safety_case, jobs, "fourth-incident safety cases")
    summaries = [row for bundle in bundles for row in bundle["summaries"]]
    nearby = [row for bundle in bundles for row in bundle["nearby"]]
    summaries.sort(key=lambda row: (row["cut"], row["pre_policy"], row["seed"]))
    nearby.sort(key=lambda row: (row["cut"], row["pre_policy"], row["seed"], row["agent_id"]))
    write_csv(RESULTS / "v3b_fourth_incident_safety_cases.csv", summaries)
    write_csv(
        RESULTS / "v3b_fourth_incident_safety_nearby_agents.csv",
        nearby,
        fieldnames=(
            "cut", "bounds", "policy_action", "pre_policy", "seed", "agent_id",
            "origin", "x", "y", "assignment", "distance_to_cut",
            "center_inside_strip", "body_intersects_strip",
        ),
    )
    print_safety_summary(summaries)


def run_selected_fourth_safety(candidate_names):
    unknown = sorted(set(candidate_names) - set(FOURTH_CUTS))
    if unknown:
        raise ValueError(f"Unknown fourth-incident candidates: {unknown}")
    policies, labels = policies_and_labels()
    cuts = [(name, FOURTH_CUTS[name]) for name in candidate_names]
    jobs = [
        (index, policy, labels[index], seed, cuts)
        for index, policy in enumerate(policies)
        for seed in SEEDS
    ]
    bundles = run_parallel(safety_case, jobs, "selected fourth safety cases")
    summaries = [row for bundle in bundles for row in bundle["summaries"]]
    nearby = [row for bundle in bundles for row in bundle["nearby"]]
    write_csv(RESULTS / "v3b_fourth_partial_safety_cases.csv", summaries)
    write_csv(
        RESULTS / "v3b_fourth_partial_safety_nearby_agents.csv",
        nearby,
        fieldnames=(
            "cut", "bounds", "policy_action", "pre_policy", "seed", "agent_id",
            "origin", "x", "y", "assignment", "distance_to_cut",
            "center_inside_strip", "body_intersects_strip",
        ),
    )
    print_safety_summary(summaries)


def run_fourth_screen(candidate_names, seeds):
    unknown = sorted(set(candidate_names) - set(FOURTH_CUTS))
    if unknown:
        raise ValueError(f"Unknown fourth-incident candidates: {unknown}")
    jobs = [
        (name, label, seed, FOURTH_CUTS[name])
        for name in candidate_names
        for label in FOURTH_SCREEN_POLICIES
        for seed in seeds
    ]
    rows = run_parallel(episode_case, jobs, "fourth-incident screen")
    rows.sort(key=lambda row: (row["incident"], row["post_policy"], row["seed"]))
    write_csv(RESULTS / "v3b_fourth_incident_policy_screen.csv", rows)
    print("candidate,policy,mean_time,mean_reward,errors,timeouts")
    for name in candidate_names:
        for label in FOURTH_SCREEN_POLICIES:
            chosen = [row for row in rows if row["incident"] == name and row["post_policy"] == label]
            print(
                f"{name},{label},"
                f"{sum(r['elapsed_time'] for r in chosen)/len(chosen):.3f},"
                f"{sum(r['episode_reward'] for r in chosen)/len(chosen):.6f},"
                f"{sum(bool(r['error']) for r in chosen)},"
                f"{sum(r['remaining_agents'] > 0 for r in chosen)}"
            )


def parse_args():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest="phase", required=True)
    subparsers.add_parser("safety")
    subparsers.add_parser("fine-lower-scan")
    selected = subparsers.add_parser("selected-lower-switch")
    selected.add_argument("xs", nargs="+", type=float)
    subparsers.add_parser("robustness")
    subparsers.add_parser("lower-alternative")
    subparsers.add_parser("trace")
    subparsers.add_parser("policy-classes")
    subparsers.add_parser("full35-class-summary")
    subparsers.add_parser("fourth-safety")
    selected_fourth = subparsers.add_parser("selected-fourth-safety")
    selected_fourth.add_argument("candidates", nargs="+")
    screen = subparsers.add_parser("fourth-screen")
    screen.add_argument("candidates", nargs="+")
    screen.add_argument("--seeds", nargs="+", type=int, default=[0])
    return parser.parse_args()


def main():
    args = parse_args()
    if args.phase == "safety":
        run_safety()
    elif args.phase == "fine-lower-scan":
        run_fine_lower_scan()
    elif args.phase == "selected-lower-switch":
        run_selected_lower_switch(tuple(args.xs))
    elif args.phase == "robustness":
        run_robustness()
    elif args.phase == "lower-alternative":
        run_lower_alternative()
    elif args.phase == "trace":
        run_trace()
    elif args.phase == "policy-classes":
        write_policy_classes()
    elif args.phase == "full35-class-summary":
        write_full35_class_summary()
    elif args.phase == "fourth-safety":
        run_fourth_safety()
    elif args.phase == "selected-fourth-safety":
        run_selected_fourth_safety(args.candidates)
    elif args.phase == "fourth-screen":
        run_fourth_screen(args.candidates, tuple(args.seeds))


if __name__ == "__main__":
    main()
