"""Replay the recorded EvacRL PPO/fixed-routing comparison in a web browser.

    python -m experiments.play_demo
    python -m experiments.play_demo --no-open

On first use, if recordings have not been committed, the script records the
two runs once (requires full EvacRL dependencies and pretrained checkpoint).
"""
import argparse
from bisect import bisect_right
from collections import defaultdict
import json
from pathlib import Path
import sqlite3
import subprocess
import sys
import webbrowser

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from shapely import wkt
from shapely.geometry import Polygon

ROOT = Path(__file__).resolve().parents[1]
MANIFEST = ROOT / "trajectories/demo_exit_c_comparison.json"
DEFAULT_OUTPUT = ROOT / "demo_evacuation.html"


def polygons(geometry):
    if geometry.is_empty:
        return
    if isinstance(geometry, Polygon):
        yield geometry
    elif hasattr(geometry, "geoms"):
        for piece in geometry.geoms:
            yield from polygons(piece)


def coordinates(geometry):
    x, y = [], []
    for polygon in polygons(geometry):
        for ring in (polygon.exterior, *polygon.interiors):
            a, b = ring.xy
            x.extend((*a, None))
            y.extend((*b, None))
    return x, y


def read_recording(path):
    if not path.is_file():
        raise FileNotFoundError(f"Missing recording: {path}")
    with sqlite3.connect(path) as db:
        row = db.execute("SELECT value FROM metadata WHERE key='fps'").fetchone()
        if row is None or float(row[0]) <= 0:
            raise ValueError(f"Missing/invalid trajectory frame rate: {path}")
        fps = float(row[0])
        geometry = {int(h): wkt.loads(s) for h, s in db.execute("SELECT hash,wkt FROM geometry")}
        frame_geometry = {int(f): int(h) for f, h in db.execute("SELECT frame,geometry_hash FROM frame_data")}
        positions = defaultdict(list)
        for frame, x, y in db.execute("SELECT frame,pos_x,pos_y FROM trajectory_data ORDER BY frame,id"):
            positions[int(frame)].append((float(x), float(y)))

    frames = sorted(set(positions) | set(frame_geometry))
    if not frames or not geometry:
        raise ValueError(f"Empty/unsupported trajectory: {path}")
    first_hash = frame_geometry.get(frames[0], next(iter(geometry)))
    return {"times": [f / fps for f in frames], "frames": frames, "positions": positions,
            "frame_geometry": frame_geometry, "geometry": geometry, "original": geometry[first_hash]}


def snapshot(recording, time, finish):
    index = max(0, bisect_right(recording["times"], time) - 1)
    frame = recording["frames"][index]
    geom_hash = recording["frame_geometry"].get(frame)
    geom = recording["geometry"].get(geom_hash, recording["original"])
    return geom, [] if time >= finish else recording["positions"].get(frame, [])


def policy_at(actions, time):
    policy = actions[0]["policy"]
    for action in actions:
        if action["time"] > time + 0.0001:
            break
        policy = action["policy"]
    return policy


def geometry_trace(geometry):
    x, y = coordinates(geometry)
    return go.Scatter(x=x, y=y, mode="lines", line=dict(color="#64748b", width=2),
                      hoverinfo="skip", showlegend=False)


def obstruction_trace(original, current):
    x, y = coordinates(original.difference(current))
    return go.Scatter(x=x, y=y, mode="lines", fill="toself",
                      fillcolor="rgba(213,55,69,0.5)", line=dict(color="#d53745", width=1),
                      hoverinfo="skip", showlegend=False)


def crowd_trace(points, color):
    return go.Scatter(x=[p[0] for p in points], y=[p[1] for p in points],
                      mode="markers", marker=dict(size=6, color=color),
                      hoverinfo="skip", showlegend=False)


def create_figure(manifest, recordings, frame_count):
    runs = manifest["runs"]
    names = ("ppo", "fixed")
    colours = {"ppo": "#1686cd", "fixed": "#e99432"}
    end = max(float(run["elapsed_time"]) for run in runs.values())
    samples = sorted({0.0, 10.0, end} | {round(end * i / (frame_count - 1), 3)
                                        for i in range(frame_count)})

    fig = make_subplots(rows=1, cols=2, subplot_titles=("Adaptive PPO", "Fixed ACA"),
                        horizontal_spacing=0.06)
    dynamic_indices = []
    for column, name in enumerate(names, 1):
        rec = recordings[name]
        current, positions = snapshot(rec, 0, float(runs[name]["elapsed_time"]))
        start = len(fig.data)
        fig.add_trace(geometry_trace(current), row=1, col=column)
        fig.add_trace(obstruction_trace(rec["original"], current), row=1, col=column)
        fig.add_trace(crowd_trace(positions, colours[name]), row=1, col=column)
        dynamic_indices.extend((start, start + 1, start + 2))
        fig.add_trace(go.Scatter(
            x=[-1.8, 5.5, 29.8, 12, 13, 5], y=[25, 1, 25, 25, 18, 18],
            text=["Exit A", "Exit B", "Exit C", "J1", "J2", "J3"], mode="text",
            textfont=dict(size=10, color="#36536c"), hoverinfo="skip", showlegend=False
        ), row=1, col=column)

    def title(t):
        return (f"EvacRL — {t:.1f}s | PPO {policy_at(runs['ppo']['actions'], t)}"
                f" · Fixed {policy_at(runs['fixed']['actions'], t)}")

    frames = []
    for i, time in enumerate(samples):
        traces = []
        for name in names:
            rec = recordings[name]
            current, positions = snapshot(rec, time, float(runs[name]["elapsed_time"]))
            traces += [geometry_trace(current), obstruction_trace(rec["original"], current),
                       crowd_trace(positions, colours[name])]
        frames.append(go.Frame(name=str(i), data=traces, traces=dynamic_indices,
                               layout=go.Layout(title=dict(text=title(time)))))

    fig.frames = tuple(frames)
    slider_steps = [
        dict(method="animate", label=f"{time:.0f}s" if i % 16 == 0 or i == len(samples)-1 else "",
             args=[[str(i)], {"mode": "immediate", "frame": {"duration": 0, "redraw": True},
                               "transition": {"duration": 0}}])
        for i, time in enumerate(samples)
    ]
    fig.update_layout(
        title=dict(text=title(0), font=dict(size=19)), template="plotly_white",
        width=1250, height=790, margin=dict(l=35, r=35, t=100, b=105),
        updatemenus=[dict(type="buttons", direction="left", x=0.02, y=-0.09,
                          buttons=[
            dict(label="▶ Play", method="animate",
                 args=[None, {"frame": {"duration": 85, "redraw": True},
                               "transition": {"duration": 0}, "fromcurrent": True}]),
            dict(label="❚❚ Pause", method="animate",
                 args=[[None], {"mode": "immediate", "frame": {"duration": 0, "redraw": True}}])
        ])],
        sliders=[dict(x=0.18, y=-0.07, len=0.8, steps=slider_steps)],
    )
    fig.add_annotation(
        text=f"Selected C-exit incident · Room C +40 at 10s · seed {manifest['seed']} · "
             f"PPO {runs['ppo']['elapsed_time']:.2f}s vs fixed {runs['fixed']['elapsed_time']:.2f}s",
        xref="paper", yref="paper", x=0.5, y=-0.16, showarrow=False,
        font=dict(size=13, color="#425466")
    )
    for col in (1, 2):
        xaxis = "xaxis" if col == 1 else "xaxis2"
        yaxis = "yaxis" if col == 1 else "yaxis2"
        fig.layout[xaxis].update(range=[-3.5, 31.5], showgrid=False, zeroline=False,
                                 showticklabels=False)
        fig.layout[yaxis].update(range=[0, 39], showgrid=False, zeroline=False,
                                 showticklabels=False, scaleanchor="x" if col == 1 else "x2",
                                 scaleratio=1)
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--no-open", action="store_true", help="Only write the HTML")
    parser.add_argument("--frames", type=int, default=220, help="Playback frames (default: 220)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    args = parser.parse_args()
    if args.frames < 2:
        parser.error("--frames must be at least 2")
    if not MANIFEST.is_file():
        print("No bundled recordings found: recording both runs once with the pretrained model.")
        subprocess.run([sys.executable, "-m", "experiments.record_demo_comparison"],
                       cwd=ROOT, check=True)

    manifest = json.loads(MANIFEST.read_text(encoding="utf-8"))
    recordings = {name: read_recording(ROOT / run["trajectory"])
                  for name, run in manifest["runs"].items()}
    fig = create_figure(manifest, recordings, args.frames)
    output = args.output.expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.write_html(str(output), include_plotlyjs=True, auto_play=False, full_html=True)
    print(f"Saved evacuation comparison to: {output}")
    if not args.no_open and not webbrowser.open(output.as_uri()):
        print(f"Open in your browser: {output}")


if __name__ == "__main__":
    main()
