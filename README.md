# EvacRL- Adaptive Evacuation Guidance using Reinforcement Learning

EvacRL is a research project that uses RL to change evacuation guidance direction in a simulated building. Instead of a fixed predetermined route, a **Proximal Policy Optimization (PPO)** agent observes the evolving crowd and changes guidance at junctions as conditions change.

The system combines [JuPedSim](https://www.jupedsim.org/) for pedestrian simulation, [Gymnasium](https://gymnasium.farama.org/) for the RL environment, and [Stable-Baselines3](https://stable-baselines3.readthedocs.io/) for PPO training and evaluation.

Developed at the **University of Warwick** as an Undergraduate Research Support Scheme (URSS) project and was presented at the **International Conference of Undergraduate Research (ICUR), 2026**.

## Key result

In the final version, PPO reduced average simulated evacuation time by **17.8%** relative to the tested robust fixed-routing policy.

 Fixed routing | PPO | Time saved |
 **66.90 s** | **55.00 s** | **11.90 s (17.8%)** |

This result came from **72 scenario and seed combinations**, evaluated with both controllers. The tested setup starts with 100 pedestrians and adds a surge of 20, 30, or 40 people at 10 simulated seconds.
## How it works

The simulated building contains **four rooms (A–D)**, **three exits (A–C)** of differing widths, and **three controllable junctions (J1–J3)**. Pedestrians move physically through the geometry in JuPedSim. The RL agent changes the destinations displayed on junction signs, which pedestrians follow when they enter a junction.

Each control step represents **10 seconds of simulated time**:

1. **Observe:** Read room populations, assigned destinations, congestion near exits and corridors, elapsed time, and incident indicators.
2. **Decide:** PPO selects one of 35 valid joint sign configurations for J1, J2, and J3.
3. **Simulate:** JuPedSim advances pedestrians according to the displayed guidance and physical constraints.
4. **Learn and valuate:** During training, the agent receives a reward penalising on elapsed time and pedestrians still inside. During evaluation, evacuation time is compared against a fixed policy.

The incidents that occur during the simulation are **a narrowed Exit ** (`C_exit`), an **obstruction of the corridor between room B and D** (`B_route_08`), and **a fully blocked corridor** (`bottom_loop_left`). Each is paired with a sudden crowd surge.

### Policy notation

A short label such as `ACA` specifies where the signs at **Junction 1, Junction 2, and Junction 3**, respectively, point to:

- `ACA`: J1 → Exit A, J2 → Exit C, J3 → Exit A.
- `CCA`: J1 → Exit C, J2 → Exit C, J3 → Exit A.
- `AJ3B`: J1 → Exit A, J2 → Junction J3, J3 → Exit B.

The environment enumerates valid policies automatically, excluding cycles.

## Repository guide

```text
EvacuationRL/
├── envs/
│   ├── evacuation_env.py         # Gymnasium interface: actions, observations, reward
│   └── static_plan_env.py        # Fixed-policy experiment 
├── scenarios/
│   ├── building_v2.py            # Building geometry, junctions, exits, incidents
│   └── buildingDiagram.jpeg      # Building schematic
├── simulation/
│   ├── jupedsim_backend.py       # Pedestrian movement, routing and geometry changes
│   └── state.py                  # Simulator-independent state representation
├── experiments/
│   ├── evaluate_incident_ppo.py  # Final PPO-versus-fixed evaluation
│   ├── train_incident_ppo_multicore.py  # Parallel PPO training
│   ├── smoke_test_v3c2.py        # Scenario / incident smoke tests
│   └── record_incident_run.py    # Record a diagnostic trajectory
├── models/                     # Trained PPO models and checkpoints
├── results/                    # Recorded evaluation CSVs and analyses
├── trajectories/               # JuPedSim SQLite trajectory recordings
├── v0_prototype/               # Initial gridworld / tabular RL exploration
├── v1_JupedsimPrototype/        # Early JuPedSim integration
└── requirements.txt
```


## Getting started

All commands should be run from the **repository root** so Python can resolve project packages and relative file paths.

### Installation

**Prerequisites:** A compatible Python installation, `pip`, and a CPU-capable machine. A GPU is not required. Python 3.12 has been used for this project, although dependency compatibility may vary by operating system.

```bash
git clone https://github.com/Duc5/EvacuationRL.git
cd EvacuationRL

python -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
```

On Windows, activate the environment using `.venv\Scripts\activate` instead.

The requirements include JuPedSim, Gymnasium, Stable-Baselines3, PyTorch and visualisation libraries.

### 1. Interactive evacuation demo

**No training required.** Explore how reinforcement learning can improve evacuation performance through an interactive, side-by-side comparison of adaptive PPO guidance and fixed routing.

```bash
python -m experiments.play_demo
```

The script generates `demo_evacuation.html` and attempts to open it in your default browser. The animation includes:

- **Side-by-side simulations:** Compare PPO-controlled and fixed-routing evacuations under identical conditions.
- **Dynamic incidents:** Observe physical obstructions appearing during the simulation, highlighted in red.
- **Adaptive guidance:** Track how PPO changes its junction routing decisions in response to the incident.
- **Interactive playback:** Play, pause and navigate through the evacuation timeline.

The featured scenario begins with **100 pedestrians**, followed by a **40-person surge in Room C** and an **Exit C obstruction** at 10 seconds.

| Controller | Evacuation time |
|---|---:|
| Fixed routing (ACA) | 97.65 s |
| Adaptive PPO | 66.25 s |
| **Improvement** | **32.2%** |

This is a selected illustrative scenario. The broader experimental evaluation is presented below.

The viewer uses recorded JuPedSim trajectories when available. If the demo recordings are absent, the script first generates them using the pretrained model, then creates the animation. Once the recordings are generated, subsequent playback does not require rerunning the simulations.

### 2. Reproduce the experimental results

To evaluate the pretrained PPO controller against a fixed routing baseline across the full set of final test conditions:

```bash
python -m experiments.evaluate_incident_ppo
```

This experiment evaluates both controllers across:

- Three incident types: Exit C obstruction, Route B obstruction and lower-loop obstruction.
- Four crowd-surge locations (Rooms A–D).
- Three surge sizes (20, 30 and 40 pedestrians).
- Two random seeds (8 and 9).

The experiment runs **144 simulation episodes** in total and reports overall evacuation times, performance by incident, and PPO routing decisions.

The final evaluation achieved an average evacuation time of **55.00 seconds with PPO**, compared with **66.90 seconds using fixed routing**, representing a **17.8% reduction** across the tested scenarios.

The script also saves detailed results to `results/ppo_v3c2_random_surges_131072_steps_seeds8-9.csv`.

### 3. Generate new visual recordings

To regenerate the recordings used in the interactive demo:

```bash
python -m experiments.record_demo_comparison --force
```

This runs the pretrained PPO controller and fixed ACA routing under identical conditions, saving the resulting JuPedSim trajectories in `trajectories/` alongside a JSON file containing the experiment configuration and recorded results.

The `--force` flag allows existing demo recordings to be replaced.

After generating the recordings, run `python -m experiments.play_demo` to visualise them.

For more detailed trajectory diagnostics, `experiments/record_incident_run.py` provides a separate fixed-guidance recording utility.


### 4. Train or continue training PPO

```bash
python -m experiments.train_incident_ppo_multicore
```

The training script uses Stable-Baselines3 PPO with **eight parallel JuPedSim environments**, randomised crowd surges and incident conditions.

Training configurations, including the number of environments, rollout length and total timesteps, can be adjusted within the script. Model checkpoints are saved under `models/v3c2_incident_threeway_ppo_random_surges_checkpoints/`.

**Important:** If `models/ppo_v3c2_random_surges.zip` already exists, training resumes from that checkpoint and saves back to the same file. Back up the pretrained model before running additional training.

---

## Experimental design and research progression

EvacRL was developed through several stages, progressively increasing the complexity of the evacuation environment to investigate when adaptive guidance becomes beneficial.

1. **Initial prototypes (V0–V1):** Developed simplified evacuation environments and established the integration between JuPedSim, Gymnasium and reinforcement learning.
2. **Static-routing experiments (V2):** Evaluated different junction configurations and established strong fixed-routing baselines. PPO largely converged towards the best static policy, revealing limited improvement opportunities in relatively stable conditions.
3. **Dynamic incidents (V3):** Introduced crowd surges and physical route obstructions to create situations where routing decisions must adapt during evacuation.
4. **Final evaluation (V3c2):** Trained PPO across different incidents, surge locations and population sizes, then benchmarked its performance against fixed guidance.

### Technical details

| Component | Implementation |
|---|---|
| Reinforcement learning | Proximal Policy Optimization (PPO), using Stable-Baselines3 |
| Pedestrian simulation | JuPedSim 1.4.2 |
| RL environment | Custom Gymnasium environment |
| Action space | 35 routing configurations |
| Observation space | 24 normalised features covering occupancy, routing assignments, congestion, elapsed time and incident indicators |
| Control interval | 10 simulated seconds |
| Reward | `-Δt × (1 + remaining_fraction) / max_time` |
| Episode termination | All pedestrians evacuated with no pending events, or the simulation time limit is reached |

Pedestrians follow JuPedSim movement and collision constraints. Guidance changes affect the routes pedestrians select when entering controlled junctions, rather than immediately redirecting every pedestrian already in transit.

## Scope and limitations

EvacRL is a research project evaluated in simulated indoor environments, not a safety-certified evacuation guidance system.

The results are specific to the building geometry, pedestrian behaviour model, incident configurations and experimental conditions used. The final evaluation uses two random seeds. Broader validation across different building layouts, pedestrian populations and incident conditions would be necessary to establish generalisability.

## Acknowledgements

Developed through the **University of Warwick Undergraduate Research Support Scheme (URSS)** under the supervision of **Dr Long Tran-Thanh**.

The research was presented at the **International Conference of Undergraduate Research (ICUR) 2026**, hosted at the University of Warwick.
