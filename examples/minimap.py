# pip install mplsoccer
import pandas as pd
import cv2
import numpy as np
from mplsoccer import Pitch
import matplotlib.pyplot as plt
from io import BytesIO
import sys
import argparse

sys.path.append("../")
from eagle.utils.io import write_video
import json

parser = argparse.ArgumentParser()
parser.add_argument("--input_dir", type=str, required=True)  # "../output/spurs"
args = parser.parse_args()

df = pd.read_json(f"{args.input_dir}/raw_data.json").fillna(value=np.nan)
with open(f"{args.input_dir}/metadata.json") as f:
    metadata = json.load(f)
fps = metadata["fps"]
team_mapping = metadata["team_mapping"]
team_mapping["19"] = 0

to_draw = [x for x in df.columns if "video" not in x and x not in ["Bottom_Left", "Top_Left", "Top_Right", "Bottom_Right"]]
pitch = Pitch(pitch_type="uefa", pitch_color="None", goal_type="box")

out = []
for i, row in df.iterrows():
    buffer = BytesIO()
    fig, ax = plt.subplots(figsize=(8, 12))
    pitch.draw(ax)
    fig.set_facecolor("black")

    boundaries = row[["Bottom_Left", "Top_Left", "Top_Right", "Bottom_Right", "Bottom_Left"]].values.tolist()
    polygon = plt.Polygon(boundaries, facecolor="white", zorder=1, closed=True, alpha=0.3)
    ax.add_patch(polygon)

    for col in to_draw:
        if type(row[col]) == float:
            continue
        x, y = row[col]

        if "Ball" in col:
            ax.scatter(x, y, color="white", zorder=5, facecolors="none", edgecolors="white", s=50)
        else:
            id = int(col.split("_")[1])
            if "Goalkeeper" in col:
                color = "green"
            else:
                if str(id) not in team_mapping:  # use string because json keys are always strings
                    continue
                team = team_mapping[str(id)]
                if team == 1:
                    color = "#43A1D5"
                else:
                    color = "#F36C21"

            ax.scatter(x, y, color=color, zorder=5, s=100)

    plt.savefig(buffer, format="png", bbox_inches="tight", pad_inches=0)
    plt.close()
    buffer.seek(0)
    img = cv2.imdecode(np.frombuffer(buffer.read(), np.uint8), 1)
    out.append(img)
print("Saving video to output_test.mp4")
write_video(out, "output_test.mp4", fps=fps)
