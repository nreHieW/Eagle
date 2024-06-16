# Example of Messi's Goal Vs Athletic Bilbao

import json
import pandas as pd
import numpy as np
from mplsoccer import Pitch
import matplotlib.pyplot as plt
import sys

sys.path.append("../")
from eagle.utils.io import read_video

frames, fps = read_video("messi.mp4")
with open("output/messi/raw_coordinates.json", "r") as f:
    coords = json.load(f)

df = pd.read_json("output/messi/processed_data.json").fillna(np.nan)
pitch = Pitch(pitch_type="uefa", pitch_color="None", goal_type="box", linewidth=0.8)
fig, ax = pitch.draw()
fig.set_facecolor("black")

ball_coords = []
for i in range(75, len(df) - 26, 10):
    item = df["Coordinates"][i]
    for x in item:
        if x["ID"] == "Ball":
            ball_coords.append(x["Coordinates"])

ax.plot([x[0] for x in ball_coords], [x[1] for x in ball_coords], color="white", zorder=5, linewidth=1)

ax.scatter(ball_coords[0][0], ball_coords[0][1], color="blue", zorder=5, s=50)
ax.scatter(ball_coords[-1][0], ball_coords[-1][1], color="blue", zorder=5, s=50)
