import json
import numpy as np
import pandas as pd
from mplsoccer import Pitch
import matplotlib.pyplot as plt
import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True)  # "../output/mancity"
args = parser.parse_args()

df = pd.read_json(f"{args.input_dir}/processed_data.json").fillna(np.nan)
with open(f"{args.input_dir}/metadata.json", "r") as f:
    team_mapping = json.load(f)["team_mapping"]
pitch = Pitch(pitch_type="uefa", pitch_color="None", goal_type="box", linewidth=0.8)
fig, ax = pitch.draw()
fig.set_facecolor("black")

coords = df["Coordinates"][0]
player_locs = []
teams = []
opp_locs = []
all_x = []
all_y = []
for item in coords:
    id = item["ID"]
    x, y = item["Coordinates"]
    if id == "23":  # Incorrect detection of assistant referee
        continue
    player_type = item.get("Type", None)
    if id == "Ball":
        ax.scatter(x, y, color="white", zorder=15, facecolors="none", edgecolors="white", s=50)
        start = (x, y)
    else:
        if player_type == "Goalkeeper":
            color = "green"
        else:
            team = team_mapping[str(id)]
            if team == 0:
                color = "#add8e6"
                opp_locs.append((x, y, color))
                teams.append(1)
            else:
                color = "red"
                player_locs.append((x, y, color))
                teams.append(0)

            all_x.append(x)
            all_y.append(y)

t1, t2 = pitch.voronoi(all_x, all_y, teams=teams)
t1 = pitch.polygon(t1, facecolor="#add8e6", edgecolor="#add8e6", alpha=0.2, zorder=1, ax=ax)
t2 = pitch.polygon(t2, facecolor="red", edgecolor="red", alpha=0.2, zorder=1, ax=ax)
sc1 = pitch.scatter([x[0] for x in player_locs], [x[1] for x in player_locs], color=[x[2] for x in player_locs], zorder=5, s=100, edgecolors=[x[2] for x in player_locs], ax=ax)
sc2 = pitch.scatter([x[0] for x in opp_locs], [x[1] for x in opp_locs], color=[x[2] for x in opp_locs], zorder=5, s=100, edgecolors=[x[2] for x in opp_locs], ax=ax)


plt.savefig("voronoi.png")
print("Saved voronoi.png")
