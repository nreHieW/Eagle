# Example: Lamine Yamal's Assist vs Northern Ireland
from mplsoccer import Pitch
import pandas as pd
import json
import numpy as np
import matplotlib.pyplot as plt
import sys

sys.path.append("../")
from eagle.utils.io import read_video


frames, fps = read_video("lamine_yamal.mp4")
with open("output/lamine_yamal/raw_coordinates.json", "r") as f:
    coords = json.load(f)

df = pd.read_json("output/lamine_yamal/processed_data.json").fillna(np.nan)
with open("output/lamine_yamal/metadata.json", "r") as f:
    team_mapping = json.load(f)["team_mapping"]

pitch = Pitch(pitch_type="uefa", pitch_color="None", goal_type="box", linewidth=0.8)
fig, ax = pitch.draw()
fig.set_facecolor("black")

coords = df["Coordinates"][38]
for item in coords:
    id = item["ID"]
    x, y = item["Coordinates"]
    player_type = item.get("Type", None)
    if id == "Ball":
        ax.scatter(x, y, color="white", zorder=5, facecolors="none", edgecolors="white", s=50)
        start = (x, y)
    else:
        if player_type == "Goalkeeper":
            color = "green"
        else:
            team = team_mapping[str(id)]
            if team == 0:
                color = "red"
            else:
                color = "white"
            if int(id) == 21:
                alpha = 1
            elif int(id) == 19:
                alpha = 1
            else:
                alpha = 0.25
        ax.scatter(x, y, color=color, zorder=5, s=100, alpha=alpha, edgecolors=color)

# end of ball trajectory
coords = df["Coordinates"][55]
for item in coords:
    id = item["ID"]
    x, y = item["Coordinates"]
    player_type = item.get("Type", None)
    if id == "Ball":
        end = (x, y)

ax.arrow(start[0], start[1], end[0] - start[0], end[1] - start[1], head_width=1, head_length=1, fc="white", ec="white", zorder=5)

plt.show()
