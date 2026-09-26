# V3-B DCS parallel exploration

Date: 2026-09-26 (Europe/London)  
Machine: Warwick DCS Rocky Linux  
JuPedSim: 1.4.2  
Experiment driver: `experiments/v3b_dcs_parallel_exploration.py`

## Executive conclusions

1. The three desired response classes survive seeds 0--4 in the requested
   representative-policy screen. `B_route_08` selects the nominal A route,
   `C_exit` selects the nominal B route, and `bottom_loop_left` selects
   `J2 -> C`. The time and reward objectives agree at routing-class level.
   The subsequently imported laptop 3x35 seed-0 sweep independently selects
   the same three actions/classes: CJ3A, AJ3B, and ACB.
2. The checked-in `bottom_loop_left` at x=8.50 has no pedestrian *centres* in
   the removed strip and produces no `switch_geometry` errors in any of 175
   pre-policy/seed cases. It is completely clear (even within 0.4 m) under CCA
   for seeds 0--4. However, under arbitrary PPO pre-policies, a 0.2 m-radius
   pedestrian disk intersects the proposed wall in 36/175 cases (40 total disk
   intersections). It is therefore engine-valid but not conservatively safe.
3. A fine spatial scan found x=8.40 to be the best clearance compromise. It has
   zero centre hits, zero switch errors, 16 disk intersections, a worst-case
   centre clearance of 0.110 m, and preserves the `J2 -> C` result. No x
   coordinate between J3 and J2 was clear of all 0.2 m-radius pedestrian disks
   for all 35 pre-policies and five seeds at t=10.
4. No credible fourth response class was found. Eight switch-valid hard cuts
   reverted to an existing A/B/C response or produced sub-second ties. Six
   partial-capacity candidates failed their safety gate and were not screened.
5. Before PPO, the incident observation must be corrected: the current encoder
   still tests old names (`A_exit`, `C_exit`, `top_loop`, `bottom_loop`). Both
   `B_route_08` and `bottom_loop_left` therefore appear as all-zero incident
   indicators. Production code was intentionally not changed in this work.

## 1. Repository state

At the start of this audit, the checkout already contained user-owned modified
tracked result/experiment files and untracked exploratory outputs, so none of
those files were reset or overwritten. The DCS exploration data were generated
against `main` at:

```text
b38a0af9f4904955e2c6e2ea952258c5f478adb0
2026-09-26T03:12:17+01:00
added J2-J3 block incident
```

During the investigation, `main`/`origin/main` advanced to:

```text
2113dffbaad547880829dad737d807bca6050333
2026-09-26T20:39:03+01:00
run evaluations on 4 incidents
```

That commit adds validation scripts/results but does not change
`building_v2.py`, `jupedsim_backend.py`, or `evacuation_env.py` relative to the
source used for the DCS runs. Three representative cases were rerun twice on
the current checkout and exactly reproduced the DCS CSV values, so the DCS
results remain compatible with current simulation code.

Contrary to the anticipated laptop/DCS lag, both newest geometry definitions
already existed in `scenarios/building_v2.py`. The laptop's 3 incidents x 35
post-actions seed-0 result was absent when DCS work began and was not
duplicated. It arrived later in the working tree as
`incident_one_switch_A40_B20_C20_D20_Csurge40_t10_seed0.csv` (105 data rows),
and is analysed in Section 6 as independent, laptop-origin evidence. Historical
files with similar names include stale incident sets, and the locally modified
one-switch script contains a literal `{SEED}` in one output filename; result
contents must therefore be identified from their `incident` and `seed` columns,
not filename alone.

Only a dedicated exploratory driver and new result files were added. The
scenario, backend, environment, and PPO code were not edited.

## 2. Exact incident definitions used

The definitions below were read from the checked-in scenario and used unchanged
for the main experiments.

```python
"B_route_08": self.geometry.difference(
    box(4.5, 8.15, 5.7, 9.85)
)

"bottom_loop_left": self.geometry.difference(
    box(8.5, 17.0, 8.55, 19.0)
)
```

`B_route_08` leaves the east-side x=5.7--6.5 opening in the Room D/Room B
connector. `bottom_loop_left` is a full-width, 0.05 m cut between J3 and J2.

The alternative lower cut was exploratory only and was not added to the
scenario:

```python
g.difference(box(8.40, 17.0, 8.45, 19.0))
```

## 3. Bottom-loop switch-safety method

For every one of the repository-generated 35 acyclic pre-event actions and
seeds 0, 1, 2, 3, 4:

1. create the A40/B20/C20/D20 population;
2. apply that pre-policy from t=0;
3. advance to t=10 without an incident and record all centres, IDs, origin
   rooms, and current journey assignments;
4. measure distance to each candidate 0.05 m x full-width strip;
5. independently replay the case, call `switch_geometry`, and advance another
   0.2 simulated seconds without suppressing exceptions.

Three geometric thresholds are reported:

- **centre hits**: centre lies in/on the removed strip (distance 0);
- **disk intersections**: centre-to-strip distance <= the model's default
  pedestrian radius, 0.2 m;
- **nearby**: distance <= 0.4 m.

Counts below are summed pedestrian-case observations across 175 cases. Switch
errors count cases, not pedestrians. Every centre-hit case produced JuPedSim's
explicit "agents would be outside of the new geometry" error; no zero-centre-hit
lower cut produced a switch error.

| Cut x | Centre hits | Disk intersections (cases) | Nearby | Switch errors | Disk-overlap seeds | Disk-overlap pre-policies |
|---:|---:|---:|---:|---:|---|---|
| 6.50 | 0 | 68 (48) | 117 | 0 | 0,1,2,3,4 | ACJ2, AJ1J2, AJ3A, AJ3B, AJ3J1, CCJ2, CJ1J2, CJ3A, CJ3B, CJ3J1, J2CJ2, J2J3A, J2J3B, J3CJ2, J3J3A, J3J3B |
| 7.00 | 0 | 14 (10) | 88 | 0 | 0 | AJ3A, AJ3B, AJ3J1, CJ3A, CJ3B, CJ3J1, J2J3A, J2J3B, J3J3A, J3J3B |
| 7.50 | 8 | 70 (54) | 106 | 8 | 0,2,3,4 | same 16-policy family as x=6.50 |
| 8.00 | 0 | 40 (34) | 76 | 0 | 1,2,3,4 | same 16-policy family as x=6.50 |
| **8.50 (current)** | **0** | **40 (36)** | **80** | **0** | **0,2,3,4** | **same 16-policy family as x=6.50** |
| 9.00 | 16 | 50 (50) | 122 | 16 | 0,1,2,3,4 | same 16-policy family as x=6.50 |
| 9.50 | 0 | 38 (32) | 58 | 0 | 1,2,3,4 | same 16-policy family as x=6.50 |
| 10.00 | 10 | 36 (36) | 72 | 10 | 0,1,2,3 | same 16-policy family as x=6.50 |
| 10.50 | 18 | 80 (70) | 116 | 18 | 0,1,2,3,4 | same 16-policy family as x=6.50 |
| 11.00 | 0 | 34 (28) | 70 | 0 | 1,2,3 | same 16-policy family as x=6.50 |

At x=8.50, 34/40 disk intersections are C-origin agents assigned to J3 and
6/40 are B-origin agents assigned to J2. Ten centres pass within 0.05 m of the
new wall (minimum 0.015 m). Agent IDs, origins, positions, and assignments are
in `v3b_bottom_loop_left_safety_nearby_agents.csv`.

Importantly, CCA is not among the offending policies: for CCA and every seed
0--4, x=8.50 has 0 centre hits, 0 disk intersections, 0 agents within 0.4 m,
and 0 switch errors. The existing CCA-first result is therefore clean; the risk
appears only when generalising to arbitrary PPO pre-event actions.

## 4. Fine coordinate search and safest candidate

A 0.05 m-resolution occupancy scan covered x=6.05--11.70. No coordinate had
zero 0.2 m-radius disk intersections. Relevant Pareto candidates were:

| x | Centre hits | Disk intersections | Nearby | Global minimum centre distance |
|---:|---:|---:|---:|---:|
| 7.05 | 0 | 12 | 84 | 0.010 m |
| 7.00 | 0 | 14 | 88 | 0.060 m |
| **8.40** | **0** | **16** | **78** | **0.110 m** |
| 9.75 | 0 | 18 | 42 | 0.018 m |
| 9.90 | 0 | 20 | 60 | 0.055 m |
| 8.50 | 0 | 40 | 80 | 0.015 m |

x=8.40 is recommended over x=8.50 if a t=10 hard cut is retained: it maximises
worst-case clearance among all zero-centre-hit scan positions, has relatively
few disk intersections, and had zero switch errors in all 175 real switch
attempts. Its 16 overlaps occur only for seeds 1 and 2 and the following
pre-policies:

```text
ACJ2, AJ1J2, AJ3A, AJ3B, AJ3J1, CCJ2, CJ1J2, CJ3A,
CJ3B, CJ3J1, J2CJ2, J2J3A, J2J3B, J3CJ2, J3J3A, J3J3B
```

Even x=8.40 is not fully safe under the strict disk-intersection criterion.
It should therefore remain exploratory until a safer event time/implementation
is found, or until a documented decision is made that JuPedSim's centre-based
switch acceptance plus follow-up trajectory evidence is the operative safety
criterion.

## 5. Three-way multi-seed robustness

Configuration: CCA from t=0 to t=10, incident and +40 Room-C surge at t=10,
candidate response afterward, 10 s control interval, 180 s maximum, seeds 0--4.
There were 80 simulations, zero errors, and only the deliberately poor CCA
response to `C_exit` timed out (all five seeds).

Values are mean +/- sample standard deviation, with the seed range in brackets.

| Incident | Post-policy | Nominal C-surge route | Time (s) | Episode reward | Timeouts |
|---|---|---|---:|---:|---:|
| B_route_08 | **CJ3A** | **J2>J3>A** | **66.24 +/- 0.58 [65.52,67.09]** | **-0.5181 +/- 0.0025** | 0 |
| B_route_08 | ACB | J2>C | 70.46 +/- 2.25 [68.54,74.34] | -0.5454 +/- 0.0136 | 0 |
| B_route_08 | ACA | J2>C | 70.46 +/- 2.25 [68.54,74.34] | -0.5490 +/- 0.0132 | 0 |
| B_route_08 | CCA | J2>C | 71.94 +/- 1.82 [70.51,74.92] | -0.5640 +/- 0.0114 | 0 |
| B_route_08 | AJ3B | J2>J3>B | 91.14 +/- 1.67 [89.79,94.01] | -0.6773 +/- 0.0111 | 0 |
| C_exit | **AJ3B** | **J2>J3>B** | **66.16 +/- 0.76 [65.25,67.36]** | **-0.5237 +/- 0.0057** | 0 |
| C_exit | CJ3A | J2>J3>A | 88.91 +/- 3.55 [86.41,95.02] | -0.6899 +/- 0.0241 | 0 |
| C_exit | ACB | J2>C | 113.63 +/- 1.44 [111.45,114.99] | -0.8526 +/- 0.0106 | 0 |
| C_exit | ACA | J2>C | 113.63 +/- 1.44 [111.45,114.99] | -0.8568 +/- 0.0103 | 0 |
| C_exit | CCA | J2>C | 180.00 | -1.5789 +/- 0.0275 | 5 |
| bottom_loop_left | **ACB** | **J2>C** | **69.46 +/- 0.72 [68.35,70.23]** | **-0.5394 +/- 0.0048** | 0 |
| bottom_loop_left | ACA | J2>C | 69.46 +/- 0.72 [68.35,70.23] | -0.5431 +/- 0.0049 | 0 |
| bottom_loop_left | ACJ1 | J2>C | 69.46 +/- 0.72 [68.35,70.23] | -0.5433 +/- 0.0047 | 0 |
| bottom_loop_left | CCA | J2>C | 70.95 +/- 1.11 [70.02,72.88] | -0.5581 +/- 0.0062 | 0 |
| bottom_loop_left | AJ3B | J2>J3>B | 83.84 +/- 1.90 [82.08,86.89] | -0.6397 +/- 0.0150 | 0 |
| bottom_loop_left | CJ3A | J2>J3>A | 92.96 +/- 1.41 [91.27,94.84] | -0.7088 +/- 0.0077 | 0 |

The x=8.40 exploratory alternative also preserved the result over seeds 0--4:

| Post-policy | Mean time (s) | Mean reward |
|---|---:|---:|
| ACB | **70.15** | **-0.5429** |
| ACA | **70.15** | -0.5466 |
| ACJ1 | **70.15** | -0.5469 |
| CCA | 71.34 | -0.5603 |
| AJ3B | 83.16 | -0.6352 |
| CJ3A | 92.14 | -0.7043 |

Thus the meaningful `J2 -> C` margin at x=8.40 is 13.01 s over the best
non-`J2 -> C` class. The within-class time tie is expected; J3 is irrelevant
to the dominant cohort when J2 sends it directly to C. Reward selects ACB.

## 6. Policy-class analysis

Following the acyclic sign graph from J2 gives nine nominal C-surge classes:

| Nominal route | Number of actions | Policies |
|---|---:|---|
| J2>C | 15 | ACB, ACJ1, ACJ2, ACA, CCB, CCJ1, CCJ2, CCA, J2CB, J2CJ1, J2CJ2, J2CA, J3CB, J3CJ2, J3CA |
| J2>J1>A | 4 | AJ1B, AJ1J1, AJ1J2, AJ1A |
| J2>J1>C | 4 | CJ1B, CJ1J1, CJ1J2, CJ1A |
| J2>J1>J3>A | 1 | J3J1A |
| J2>J1>J3>B | 1 | J3J1B |
| J2>J3>A | 4 | AJ3A, CJ3A, J2J3A, J3J3A |
| J2>J3>B | 4 | AJ3B, CJ3B, J2J3B, J3J3B |
| J2>J3>J1>A | 1 | AJ3J1 |
| J2>J3>J1>C | 1 | CJ3J1 |

The representative screen gives:

| Incident | Best action by time | Best action by reward | Best nominal class by time/reward | Margin to second class |
|---|---|---|---|---:|
| B_route_08 | CJ3A | CJ3A | J2>J3>A | 4.22 s |
| C_exit | AJ3B | AJ3B | J2>J3>B | 22.74 s |
| bottom_loop_left | ACB/ACA/ACJ1 tie | ACB | J2>C | 14.38 s |

These are the multi-seed results among the requested representative actions.

Across the five actions common to all three representative sets, the best
event-blind timed response is AJ3B at 80.38 s mean. The incident-aware mean is
67.29 s. The restricted-set incident-information value is therefore:

```text
80.379 - 67.289 = 13.091 s = 16.29%
```

The corresponding event-blind mean reward is -0.6136. Choosing the best-reward
action per incident gives -0.5271, an improvement of 0.0865 reward units. This
restricted-set comparison is valuable for robustness but is not the all-action
event-blind comparator.

### Imported all-35 seed-0 confirmation

The laptop full sweep arrived after the DCS screen completed. It contains
exactly 35 actions for each of the same three incidents and was **read, not
rerun**, on DCS. It independently confirms both the action winners and the
class-level reward preference:

| Incident | Best action by time | Best action by reward | Best class | Second-best class by time | Class margin |
|---|---|---|---|---|---:|
| B_route_08 | CJ3A, 65.52 s | CJ3A, -0.5152 | J2>J3>A | J2>C, 69.57 s | 4.05 s |
| C_exit | AJ3B, 67.36 s | AJ3B, -0.5333 | J2>J3>B | J2>J3>A, 76.72 s | 9.36 s |
| bottom_loop_left | ACB, 69.96 s | ACB, -0.5430 | J2>C | J2>J1>A, 77.85 s | 7.89 s |

Reward also separates the winning class from its nearest reward competitor:
0.0257 for `B_route_08`, 0.0755 for `C_exit`, and 0.0601 for
`bottom_loop_left`. These are not noise-sized policy-string distinctions.

Across all 35 post-event actions at seed 0, the best event-blind timed response
is CCA -> AJ1B at 78.02 s mean. The incident-aware all-action mean is 67.61 s,
giving:

```text
78.023 - 67.613 = 10.410 s = 13.34%
```

AJ1B is also best event-blind by reward (-0.6070). The incident-aware mean best
reward is -0.5305, an improvement of 0.0765. This is a one-seed all-action
estimate; the five-seed representative result above is stronger evidence for
robust class identity, while the full sweep is stronger evidence that an
untested nominal action did not steal the seed-0 optimum.

The imported laptop seeds-1--4 CSV has small platform-dependent numerical
differences from DCS (for example `B_route_08`, CJ3A, seed 1 is 68.14 s on the
laptop CSV and 66.41 s when rerun on DCS), but it preserves all three routing
class winners. Accordingly, cross-platform values are not pooled into one mean.

## 7. Why `bottom_loop_left` works

The seed-0 trace followed the 40 newly spawned Room-C IDs at 0.2 s resolution.

### CCA -> ACB

All 40 enter J2 and change assignment from J2 to C. They proceed east on the
lower corridor, climb the right connector, remain east of J1, and reach Exit C.
All 40 use the right connector and Exit-C neck; none needs a J1 sign decision.
The last recorded Exit-C-neck entry is at 65.32 s.

### CCA -> AJ3B

All 40 enter J2 and are assigned to J3. The direct westward J2--J3 path is cut,
so JuPedSim routes them east, up the right connector, and west along the upper
corridor. All 40 then encounter J1. AJ3B has `J1 -> A`, so their assignment
changes from J3 to A and all 40 continue west to Exit A. The last Exit-A-neck
entry is at 81.78 s. The nominal `J2 -> J3 -> B` plan therefore does **not**
send the surge to B in this geometry: the forced J1 encounter intercepts it.

### CCA -> CJ3A

The initial detour is the same: all 40 are assigned J3 at J2 and route east/up
toward the upper corridor. On crossing J1, all 40 change assignment from J3 to
C because CJ3A has `J1 -> C`. They then reverse toward Exit C. All 40 ultimately
enter the Exit-C neck, but the unnecessary trip to J1 and reversal gives a last
neck-entry time of 89.09 s.

Therefore `J2 -> C` wins for a causal, interpretable reason: it chooses the
reachable east-side route immediately. `J2 -> J3` first requests an unreachable
west-side waypoint, triggers physical rerouting around the loop, and creates an
otherwise unexpected second controlled decision at J1. The result is not an
arbitrary policy-string effect.

## 8. Fourth-incident search

### Stage 0: hard-cut safety map

The map included 14 upper-corridor x cuts, four left-connector y cuts, four
right-connector y cuts, and five Room-A-connector y cuts. All were full-width
and 0.05 m thick. Each was audited over all 175 pre-policy/seed cases.

Key rejection findings:

- every left-connector candidate had centre hits and matching switch errors;
- all five Room-A-connector cuts failed all 175 switches with `accessible area
  not connected`;
- upper x=20.50, 21.50, and 22.50 also failed all 175 switches because the
  accessible area became disconnected;
- upper x=13.50, 14.50, 16.00, and 16.50 and right-connector y=20.50 had
  centre-hit switch failures;
- eight candidates had zero centre hits and zero switch errors and advanced to
  Stage 1: right-connector y=19.50/21.50/22.50 and upper-corridor
  x=6.50/8.50/10.50/15.50/17.50.

None of the eight was disk-clear. The least-overlapped was right-connector
y=21.50 with 15 disk intersections; upper x=6.50 had 38.

### Stage 1: representative-policy screen

Each surviving candidate was tested at seed 0 with ACB, CCB, J2CB, J3CB, CCA,
AJ3B, CJ3B, and CJ3A. There were no exceptions; `upper x=8.50` timed out with
3 agents remaining under ACB and 1 remaining under J3CB, providing an
additional rejection reason for that candidate.

| Candidate | Best time action | Time (s) | Best reward action | Class margin | Decision |
|---|---|---:|---|---:|---|
| right connector y=19.50 | CJ3A | 70.99 | CJ3A | 0.70 s | existing A class; weak |
| right connector y=21.50 | AJ3B/CJ3B tie | 69.22 | AJ3B | 0.00 s | existing B class; J1 irrelevant |
| right connector y=22.50 | AJ3B/CJ3B tie | 70.58 | AJ3B | 0.00 s | existing B class; J1 irrelevant |
| upper x=6.50 | CJ3A | 65.35 | CJ3A | 2.01 s | existing A class |
| upper x=8.50 | CJ3A | 66.04 | CJ3A | 1.32 s | existing A class |
| upper x=10.50 | CJ3A | 65.52 | CJ3A | 1.84 s | existing A class |
| upper x=15.50 | ACB/CCB/J2CB/J3CB tie | 67.32 | ACB | 0.00 s | existing C class; J1 tie |
| upper x=17.50 | AJ3B | 67.80 | AJ3B | 0.44 s | existing B class; noise-sized |

The strongest of these by class margin is upper x=6.50:

```python
g.difference(box(6.50, 23.0, 6.55, 27.0))
```

It is not a qualifying fourth incident. It reproduces the already-used
`J2 -> J3 -> A` response (CJ3A), clears the second class by only 2.01 s at one
seed, and has 38 disk intersections in 31/175 safety cases.

### Partial-capacity follow-up

Six connected alternatives were audited: 0.8/1.2 m central openings along the
left and right upper branches, and 0.8 m openings in the left and right loop
connectors. They were rejected before policy screening:

| Candidate | Cases with centre hits / switch errors | Total centre hits |
|---|---:|---:|
| upper-left narrow 0.8 m | 110/175 | 566 |
| upper-left narrow 1.2 m | 110/175 | 513 |
| upper-right narrow 0.8 m | 85/175 | 488 |
| upper-right narrow 1.2 m | 85/175 | 419 |
| left connector narrow 0.8 m | 110/175 | 396 |
| right connector narrow 0.8 m | 135/175 | 255 |

No fourth candidate merits a full 35-action or multi-seed expansion.

## 9. Pre-PPO observation issue

`EvacuationEnv._make_observation` currently appends these four flags:

```python
float(state.active_incident == "A_exit")
float(state.active_incident == "C_exit")
float(state.active_incident == "top_loop")
float(state.active_incident == "bottom_loop")
```

The new names `B_route_08` and `bottom_loop_left` never activate a flag. A PPO
policy trained on the present three incidents could observe `C_exit` explicitly
but would see identical all-zero incident flags for the other two. Physical
state may still differ, but this defeats the intended explicit incident-aware
observation and must be resolved before training. The user-owned working change
to `train_incident_env_ppo.py` now lists the correct three incident names, but
that does not repair the observation encoding itself.

## 10. Exact recommended next experiment

Do **not** train PPO yet. First run a reusable spatiotemporal safety audit:

1. For all 35 pre-policies and seeds 0--9, simulate once to t=12 and snapshot
   positions every 0.25 s from t=8 to t=12 (350 simulations total; evaluate all
   coordinates offline from those trajectories).
2. At each snapshot, scan 0.05 m full-width cuts from x=6.05 to x=11.70 in
   0.05 m steps. Require zero centre hits and search first for a pair with
   global minimum centre distance >0.20 m.
3. For shortlisted `(time, x)` pairs, replay all 350 cases with the actual
   `switch_geometry` call and advance at least 1 simulated second, retaining
   every exception and the minimum post-switch wall clearance.
4. If a strict-safe pair exists, run only ACB, CCA, AJ3B, and CJ3A with CCA
   before the event and seeds 0--9, moving the C surge to the same shortlisted
   event time. This is 40 performance runs per shortlisted pair and tests
   whether the three-class structure survives the safety adjustment.
5. If a strict-safe pair survives, use the now-imported full-sweep winners
   (CJ3A/AJ3B/ACB) plus the closest class competitors for seeds 5--9. Do not
   rerun the 3x35 seed-0 sweep on DCS.

If no strict-safe `(time, x)` pair exists, either constrain the pre-event plan
to the demonstrably clear CCA warm start or replace instantaneous hard removal
with a different incident mechanism. Do not silently accept the arbitrary-plan
disk overlaps.

Separately, before any PPO run, replace the stale hard-coded incident indicators
with an encoding that distinguishes `B_route_08`, `C_exit`, and
`bottom_loop_left`, and add an observation test asserting that the three event
states are distinct. That code change is intentionally outside this exploratory
geometry/results task.

## 11. Raw outputs

- `v3b_bottom_loop_left_safety_cases.csv`: all 10 coarse x positions x 175 cases
- `v3b_bottom_loop_left_safety_nearby_agents.csv`: IDs/origins/positions within 0.4 m
- `v3b_bottom_loop_left_fine_occupancy_scan.csv`: 0.05 m coordinate scan
- `v3b_bottom_loop_left_selected_switch_cases.csv`: x=8.40 real-switch audit
- `v3b_bottom_loop_left_selected_switch_nearby_agents.csv`: x=8.40 agent detail
- `v3b_three_class_multiseed.csv`: requested 80-run robustness screen
- `v3b_bottom_loop_left_x8p40_multiseed.csv`: 30-run safer-coordinate validation
- `v3b_policy_routing_classes.csv`: all 35 nominal route classes
- `v3b_full35_seed0_policy_class_summary.csv`: derived class summary of the
  imported laptop 3x35 seed-0 sweep
- `v3b_bottom_loop_left_surge_trace.csv`: aggregate C-surge trajectory samples
- `v3b_bottom_loop_left_surge_agent_events.csv`: per-agent zone/assignment changes
- `v3b_fourth_incident_safety_cases.csv`: hard-cut Stage-0 matrix
- `v3b_fourth_incident_safety_nearby_agents.csv`: hard-cut proximity detail
- `v3b_fourth_incident_policy_screen.csv`: eight hard cuts x eight policies
- `v3b_fourth_partial_safety_cases.csv`: partial-narrowing Stage-0 matrix
- `v3b_fourth_partial_safety_nearby_agents.csv`: partial-narrowing agent detail
- `incident_one_switch_A40_B20_C20_D20_Csurge40_t10_seed0.csv`: imported raw
  laptop 3x35 seed-0 sweep (pre-existing user-owned working result)
