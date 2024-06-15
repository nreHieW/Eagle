import os

os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"  # only for macs
from eagle.models import CoordinateModel
from eagle.processor import Processor
from eagle.utils.io import read_video, write_video
import json
from argparse import ArgumentParser
import pandas as pd
import cv2


def main():
    parser = ArgumentParser()
    parser.add_argument("--video_path", type=str, required=True)
    args = parser.parse_args()

    os.makedirs("output", exist_ok=True)
    video_name = args.video_path.split("/")[-1].split(".")[0]
    os.makedirs(f"output/{video_name}", exist_ok=True)
    root = f"output/{video_name}"

    frames, fps = read_video(args.video_path)
    model = CoordinateModel()
    coordinates = model.get_coordinates(frames, fps, num_homography=1, num_keypoint_detection=3)

    with open(f"{root}/raw_coordinates.json", "w") as f:
        json.dump(coordinates, f, default=str)

    print("Processing Data")

    processor = Processor(coordinates, frames, fps)
    df, team_mapping = processor.process_data()  # Smoothing
    df.to_json(f"{root}/raw_data.json", orient="records")
    with open(f"{root}/metadata.json", "w") as f:
        json.dump({"fps": fps, "team_mapping": team_mapping}, f, default=str)

    processed_df = processor.format_data(df)
    processed_df.to_json(f"{root}/processed_data.json", orient="records")

    out = []
    cols = [x for x in df.columns if "video" in x and x not in ["Bottom_Left", "Top_Left", "Top_Right", "Bottom_Right"]]
    for i, row in df.iterrows():
        curr_frame = frames[int(i)].copy()
        for col in cols:
            if pd.isna(row[col]):
                continue
            x, y = row[col]

            if "Ball" in col:
                color = (0, 255, 0)
                cv2.circle(curr_frame, (int(x), int(y)), 10, color, 1)
            else:
                id = int(col.split("_")[1])
                if "Goalkeeper" in col:
                    color = (0, 255, 0)
                else:

                    if str(id) not in team_mapping:
                        continue
                    team = team_mapping[str(id)]
                    if team == 0:
                        color = (0, 0, 255)
                    else:
                        color = (255, 0, 0)
                cv2.circle(curr_frame, (int(x), int(y)), 10, color, 1)
                cv2.putText(curr_frame, str(id), (int(x), int(y)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        out.append(curr_frame)

    write_video(out, f"{root}/annotated.mp4", fps)


if __name__ == "__main__":
    main()
