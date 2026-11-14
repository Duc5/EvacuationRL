"""Record matched PPO and fixed-routing runs for the public EvacRL demo.

Run from the repository root:
    python -m experiments.record_demo_comparison

Recording requires installed simulation/RL dependencies, but no training.
Commit the generated SQLite files for instant playback by future visitors.
"""

import argparse
import json
from pathlib import Path

from stable_baselines3 import PPO

from envs.evacuation_env import EvacuationEnv
from scenarios.building_v2 import BuildingV2Scenario


ROOT = Path(__file__).resolve().parents[1]
MODEL_PATH = ROOT / "models/ppo_v3c2_random_surges.zip"
OUTPUT_DIR = ROOT / "trajectories"
MANIFEST_PATH = OUTPUT_DIR / "demo_exit_c_comparison.json"

SEED = 8
INCIDENT = "C_exit"
SURGE_ROOM = "C"
SURGE_COUNT = 40
SURGE_TIME = 10.0
CONTROL_INTERVAL = 10.0
MAX_TIME = 180.0
INITIAL_COUNTS = {"A": 40, "B": 20, "C": 20, "D": 20}
FIXED_POLICY = "ACA"


def policy_label(policy):
    return "".join(policy[junction] for junction in ("J1", "J2", "J3"))


def run_controller(controller, trajectory_path, model=None):
    scenario = BuildingV2Scenario(room_counts=dict(INITIAL_COUNTS))
    env = EvacuationEnv(
        scenario=scenario,
        control_interval=CONTROL_INTERVAL,
        max_time=MAX_TIME,
        record=True,
        trajectory_path=trajectory_path,
        dynamic_surge_rooms=[SURGE_ROOM],
        dynamic_surge_times=[SURGE_TIME],
        dynamic_surge_counts=[SURGE_COUNT],
        incidents=[INCIDENT],
        forced_incident=INCIDENT,
    )

    actions = []
    total_reward = 0.0
    try:
        labels = [policy_label(policy) for policy in env.guidance_plans]
        fixed_action = labels.index(FIXED_POLICY)
        observation, info = env.reset(seed=SEED)
        terminated = truncated = False

        while not (terminated or truncated):
            if controller == "ppo":
                action, _ = model.predict(observation, deterministic=True)
                action = int(action)
            else:
                action = fixed_action

            actions.append({"time": round(info["elapsed_time"], 2), "policy": labels[action]})
            observation, reward, terminated, truncated, info = env.step(action)
            total_reward += reward

        if not terminated or info["remaining_agents"] != 0:
            raise RuntimeError(f"{controller}: episode timed out with {info['remaining_agents']} agents remaining")

        return {
            "trajectory": str(trajectory_path.relative_to(ROOT).as_posix()),
            "elapsed_time": round(float(info["elapsed_time"]), 4),
            "remaining_agents": int(info["remaining_agents"]),
            "episode_reward": round(float(total_reward), 6),
            "actions": actions,
            "geometry_switch_delay": info["geometry_switch_delay"],
        }
    finally:
        env.close()  # Flush the buffered SQLite trajectory writer.


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--force", action="store_true", help="Replace previously generated demo recordings")
    args = parser.parse_args()

    if not MODEL_PATH.is_file():
        parser.error(f"Pretrained PPO checkpoint not found: {MODEL_PATH}")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    paths = {name: OUTPUT_DIR / f"demo_exit_c_{name}.sqlite" for name in ("ppo", "fixed")}
    existing = [p for p in (*paths.values(), MANIFEST_PATH) if p.exists()]
    if existing and not args.force:
        parser.error("Demo recordings already exist. Use --force to replace them deliberately.")

    # Never overwrite research checkpoints, evaluation CSVs, or old trajectories.
    print(f"Recording seed={SEED}, incident={INCIDENT}, surge={SURGE_ROOM}+{SURGE_COUNT} at {SURGE_TIME}s")
    model = PPO.load(str(MODEL_PATH), device="cpu")

    recordings = {}
    for name in ("ppo", "fixed"):
        print(f"Running {name} controller...")
        recordings[name] = run_controller(name, paths[name], model if name == "ppo" else None)
        print(f"  Completed in {recordings[name]['elapsed_time']:.2f} simulated seconds")

    manifest = {
        "description": "Matched pretrained PPO vs fixed ACA evacuation runs (selected illustrative scenario)",
        "seed": SEED,
        "incident": INCIDENT,
        "initial_counts": INITIAL_COUNTS,
        "surge": {"room": SURGE_ROOM, "count": SURGE_COUNT, "time": SURGE_TIME},
        "control_interval": CONTROL_INTERVAL,
        "model": str(MODEL_PATH.relative_to(ROOT).as_posix()),
        "runs": recordings,
    }
    MANIFEST_PATH.write_text(json.dumps(manifest, indent=2) + "\n", encoding="utf-8")

    fixed, ppo = recordings["fixed"]["elapsed_time"], recordings["ppo"]["elapsed_time"]
    print(f"\nFixed: {fixed:.2f}s | PPO: {ppo:.2f}s | saved: {fixed - ppo:.2f}s ({100 * (fixed - ppo) / fixed:.1f}%)")
    print(f"Wrote {MANIFEST_PATH.relative_to(ROOT)}")
    print("Next: python -m experiments.play_demo")
    print("Commit the two demo_exit_c_*.sqlite files and demo_exit_c_comparison.json for instant playback.")


if __name__ == "__main__":
    main()
