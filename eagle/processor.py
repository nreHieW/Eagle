from sklearn.cluster import KMeans
import pandas as pd
import cv2
import numpy as np
import math
from collections import Counter

PITCH_WIDTH = 105
PITCH_HEIGHT = 68
color_ranges = {
    "red": [(0, 100, 100), (10, 255, 255)],
    "red2": [(160, 100, 100), (179, 255, 255)],
    "orange": [(11, 100, 100), (25, 255, 255)],
    "yellow": [(26, 100, 100), (35, 255, 255)],
    "green": [(36, 100, 100), (85, 255, 255)],
    "cyan": [(86, 100, 100), (95, 255, 255)],
    "blue": [(96, 100, 100), (125, 255, 255)],
    "purple": [(126, 100, 100), (145, 255, 255)],
    "magenta": [(146, 100, 100), (159, 255, 255)],
    "white": [(0, 0, 200), (180, 30, 255)],
    "gray": [(0, 0, 50), (180, 30, 200)],
    "black": [(0, 0, 0), (180, 255, 50)],
}


def calculate_distance(pt1, pt2):
    return np.sqrt((pt1[0] - pt2[0]) ** 2 + (pt1[1] - pt2[1]) ** 2)


def interpolate_df(df, col_name: str, fill: bool = False):
    col_x = f"{col_name}_x"
    col_y = f"{col_name}_y"
    if fill:
        df[col_x] = df[col_name].apply(lambda x: x[0] if x is not np.NaN else np.NaN).interpolate(method="linear").bfill().ffill()
        df[col_y] = df[col_name].apply(lambda x: x[1] if x is not np.NaN else np.NaN).interpolate(method="linear").bfill().ffill()
    else:
        df[col_x] = df[col_name].apply(lambda x: x[0] if x is not np.NaN else np.NaN).interpolate(method="linear", limit_area="inside")
        df[col_y] = df[col_name].apply(lambda x: x[1] if x is not np.NaN else np.NaN).interpolate(method="linear", limit_area="inside")
    df[col_name] = df[[col_x, col_y]].apply(lambda x: (x.iloc[0], x.iloc[1]) if not math.isnan(x.iloc[0]) or not math.isnan(x.iloc[1]) else np.NaN, axis=1)

    df = df.drop(columns=[col_x, col_y])
    return df


def smooth_df(df, col_name: str):
    col_x = f"{col_name}_x"
    col_y = f"{col_name}_y"
    df[col_x] = df[col_name].apply(lambda x: x[0] if x is not np.NaN else np.NaN)
    df[col_y] = df[col_name].apply(lambda x: x[1] if x is not np.NaN else np.NaN)
    df.loc[::2, col_x] = np.NaN
    df.loc[::2, col_y] = np.NaN
    df[col_x] = df[col_x].interpolate(method="linear", limit_area="inside")
    df[col_y] = df[col_y].interpolate(method="linear", limit_area="inside")
    df[col_name] = df[[col_x, col_y]].apply(lambda x: (x.iloc[0], x.iloc[1]) if not math.isnan(x.iloc[0]) or not math.isnan(x.iloc[1]) else np.NaN, axis=1)
    df = df.drop(columns=[col_x, col_y])
    return df


class Processor:
    def __init__(self, coords, frames: list, fps: int):
        assert len(coords) == len(frames), f"Length of coords ({len(coords)}) and frames ({len(frames)}) should be the same"
        self.coords = coords  # Data should be same format as CoordinateModel output
        self.frames = frames
        self.fps = fps

    def process_data(self, smooth: bool = False):
        df = self.create_dataframe()
        df = interpolate_df(df, "Ball", fill=True)
        df = interpolate_df(df, "Ball_video", fill=True)
        team_mapping = self.get_team_mapping()
        df.index = df.index.astype(int)
        df = self.merge_data(df, team_mapping)

        for col in df.columns:
            df = interpolate_df(df, col, fill=False)
            if smooth:
                df = smooth_df(df, col)
        return df, team_mapping

    def format_data(self, df):
        out = []
        for frame_number in df.index:
            indiv = {}
            indiv["Boundaries"] = [df.loc[frame_number, "Bottom_Left"], df.loc[frame_number, "Top_Left"], df.loc[frame_number, "Top_Right"], df.loc[frame_number, "Bottom_Right"]]
            row = df.loc[frame_number]
            indiv_data = []
            indiv_data_video = []
            # select non na values
            for col in df.columns:
                if col in ["Bottom_Left", "Top_Left", "Top_Right", "Bottom_Right"]:
                    continue
                val = row[col]
                if pd.isna(val):
                    continue
                if "ball" in col.lower():  # handle separately
                    continue
                id = col.split("_")[1]
                id = int(id)
                col_type = col.split("_")[0]
                item = {"ID": id, "Coordinates": val, "Type": col_type}
                if "video" in col:
                    indiv_data_video.append(item)
                else:
                    indiv_data.append(item)

            # handle ball
            ball = row["Ball"]
            indiv_data.append({"ID": "Ball", "Coordinates": ball})
            ball_video = row["Ball_video"]
            indiv_data_video.append({"ID": "Ball", "Coordinates": ball_video})

            indiv["Coordinates"] = indiv_data
            indiv["Coordinates_video"] = indiv_data_video

            out.append(indiv)
        return pd.DataFrame(out)

    def create_dataframe(self):
        ball_coords_image = []
        ball_coords = []
        out = {}
        for frame_number in self.coords.keys():
            indiv = {}
            boundaries = self.coords[frame_number]["Boundaries"]
            indiv["Bottom_Left"] = boundaries[0]
            indiv["Top_Left"] = boundaries[1]
            indiv["Top_Right"] = boundaries[2]
            indiv["Bottom_Right"] = boundaries[3]
            for name in ["Player", "Goalkeeper"]:
                if name not in self.coords[frame_number]["Coordinates"]:
                    continue
                curr_coords = self.coords[frame_number]["Coordinates"][name]
                for id, item in curr_coords.items():
                    x1, y1, x2, y2 = item["BBox"]
                    indiv[f"{name}_{id}"] = item["Transformed_Coordinates"]
                    indiv[f"{name}_{id}_video"] = ((x1 + x2) / 2, y2)
            out[frame_number] = indiv
            if "Ball" not in self.coords[frame_number]["Coordinates"]:
                ball_coords.append(None)
                ball_coords_image.append(None)
                continue
            curr_coords = self.coords[frame_number]["Coordinates"]["Ball"]
            indiv_img = []
            indiv_real = []
            for id, item in curr_coords.items():
                confidence = float(item["Confidence"])
                transformed_coords = item["Transformed_Coordinates"]
                x1, y1, x2, y2 = item["BBox"]
                center = ((x1 + x2) / 2, y2)
                if transformed_coords[0] < 0 or transformed_coords[0] > PITCH_WIDTH or transformed_coords[1] < 0 or transformed_coords[1] > PITCH_HEIGHT:
                    continue
                else:
                    indiv_real.append((transformed_coords, confidence))
                    indiv_img.append((center, confidence))

            if len(indiv) == 0:
                ball_coords.append(None)
                ball_coords_image.append(None)
                continue
            indiv_img = sorted(indiv_img, key=lambda x: x[1], reverse=True)
            indiv_real = sorted(indiv_real, key=lambda x: x[1], reverse=True)
            ball_coords.append([x[0] for x in indiv_real])
            ball_coords_image.append([x[0] for x in indiv_img])
        final_ball_coords_img = self.parse_ball_detections_with_kalman(ball_coords_image)
        final_ball_coords = self.parse_ball_detections_with_kalman(ball_coords)
        df = pd.DataFrame(out).T
        df["Ball"] = [x if x is not None else np.nan for x in final_ball_coords]
        df["Ball_video"] = [x if x is not None else np.nan for x in final_ball_coords_img]
        df = df.loc[:, df.notna().sum() >= 0.01 * len(df)]  # Remove columns with less than 1% non-None values
        return df

    def merge_data(self, df, team_mapping):
        goal_keeper_cols = [x for x in df.columns if "Goalkeeper" in x and "video" in x]
        goal_keeper_ids = [x.split("_")[1] for x in goal_keeper_cols]
        for id in goal_keeper_ids:
            player_col = f"Player_{id}"
            player_col_video = f"Player_{id}_video"
            goal_keeper_col = f"Goalkeeper_{id}"
            goal_keeper_col_video = f"Goalkeeper_{id}_video"
            if player_col in df.columns and player_col_video in df.columns:
                df[goal_keeper_col] = df[player_col].combine_first(df[goal_keeper_col])
                df[goal_keeper_col_video] = df[player_col_video].combine_first(df[goal_keeper_col_video])
                df.drop(columns=[player_col, player_col_video], inplace=True)

        cols = [x for x in df.columns if "Ball" not in x and "video" in x]
        TEMPORAL_THRESHOLD = int(self.fps * 1.1)

        player_video_cols = [x for x in cols if "Player" in x and "video" in x]
        player_cols = [x for x in cols if "Player" in x and "video" not in x]
        goalkeeper_video_cols = [x for x in cols if "Goalkeeper" in x and "video" in x]
        goalkeeper_cols = [x for x in cols if "Goalkeeper" in x and "video" not in x]

        to_merge = []

        # Should only merge based on video coordinates
        for col in cols:
            if "Player" in col:
                candidate_cols = player_video_cols
            elif "Goalkeeper" in col:
                candidate_cols = goalkeeper_video_cols
            else:
                print("(Should not see this): Error in column name:", col)
                continue

            last_valid_index_col = df[col].last_valid_index()
            first_valid_index_col = df[col].first_valid_index()
            for candidate in candidate_cols:
                if col == candidate:
                    continue
                first_valid_index_candidate = df[candidate].first_valid_index()
                last_valid_index_candidate = df[candidate].last_valid_index()

                # If there is an overlap, ignore
                if last_valid_index_col is not None and first_valid_index_candidate is not None and last_valid_index_col >= first_valid_index_candidate:
                    continue

                # Check which appears first
                if first_valid_index_candidate < first_valid_index_col:  # candidate appears first so we want the last
                    first_valid_index = first_valid_index_col
                    first_valid_val = df[col].loc[first_valid_index]
                    last_valid_index = last_valid_index_candidate
                    last_valid_val = df[candidate].loc[last_valid_index]
                else:
                    first_valid_index = first_valid_index_candidate
                    first_valid_val = df[candidate].loc[first_valid_index]
                    last_valid_index = last_valid_index_col
                    last_valid_val = df[col].loc[last_valid_index]

                # Essentially, we want the first valid index of the second id and the last valid index of the first appearing id
                # Condition 1 - Temporal
                if last_valid_index is None or first_valid_index is None:
                    continue
                if abs(last_valid_index - first_valid_index) > TEMPORAL_THRESHOLD:
                    continue

                # Condition 2 - Spatial
                threshold = abs(last_valid_index - first_valid_index) * 10

                dist = calculate_distance(last_valid_val, first_valid_val)
                if dist > threshold:
                    continue

                # Condition 3 - Team
                id = col.split("_")[1]
                id = int(id)
                candidate_id = candidate.split("_")[1]
                candidate_id = int(candidate_id)

                # Edge case: If we could not determine the team previously, it will not appear in the team mapping.
                # then, we just assume that this unidentified player can belong to any team
                if id in team_mapping and candidate_id in team_mapping:
                    if team_mapping[id] != team_mapping[candidate_id]:
                        continue

                # Merge
                to_merge.append((col, candidate))
            # Add the real world coordinates
        merge_real = []
        for a, b in to_merge:
            merge_real.append((a.replace("_video", ""), b.replace("_video", "")))

        to_merge.extend(merge_real)
        merged_cols = {}

        def find_root(col):
            # Find the root column to merge into
            while col in merged_cols:
                col = merged_cols[col]
            return col

        # Merge
        for col, candidate in to_merge:
            root_col = find_root(col)
            root_candidate = find_root(candidate)

            if root_col != root_candidate:
                df[root_col] = df[root_col].combine_first(df[root_candidate])
                df.drop(columns=[root_candidate], inplace=True)
                merged_cols[root_candidate] = root_col

        return df

    def parse_ball_detections_with_kalman(self, detections: list, num_to_init: int = 5, threshold: int = 20):
        init_vals = []
        non_none_init_vals = 0
        i = 0
        while True:
            if (non_none_init_vals >= 2) and (len(init_vals) >= num_to_init):
                break
            curr = detections[i]
            if curr is not None:
                init_vals.append(curr[0])
                non_none_init_vals += 1
            else:
                init_vals.append(None)
            i += 1
            if i == len(detections):
                break

        if non_none_init_vals < 2:
            print("Not enough non-None coordinates to initialize Kalman Filter")
            return detections

        # Interpolate the initial values and backfill

        init_x = [x[0] if x is not None else None for x in init_vals]
        init_y = [x[1] if x is not None else None for x in init_vals]
        init_x = pd.Series(init_x).interpolate(method="linear").bfill().ffill().tolist()
        init_y = pd.Series(init_y).interpolate(method="linear").bfill().ffill().tolist()
        init_vals = [(x, y) for x, y in zip(init_x, init_y)]
        velocities = [(init_vals[i][0] - init_vals[i - 1][0], init_vals[i][1] - init_vals[i - 1][1]) for i in range(1, len(init_vals))]
        avg_velocity = (np.mean([x[0] for x in velocities]), np.mean([x[1] for x in velocities]))
        kf = KalmanFilter(initial_state=init_vals[0], initial_velocity=avg_velocity)
        ball_positions = []
        prev_pos = None
        prev_idx = None
        for i, candidates in enumerate(detections):
            if candidates is None or len(candidates) == 0:
                ball_positions.append(None)
                continue
            if len(candidates) == 1:
                measurement = np.array([[np.float32(candidates[0][0])], [np.float32(candidates[0][1])]])
            else:
                # Select the candidate closest to the prediction
                prediction = kf.predict()
                predicted_pos = (prediction[0, 0], prediction[1, 0])
                distances_from_pred = [np.linalg.norm(np.array(candidate) - np.array(predicted_pos)) for candidate in candidates]
                if prev_pos is not None:
                    distances_from_prev = [np.linalg.norm(np.array(candidate) - np.array(prev_pos)) for candidate in candidates]
                    distances = [0.5 * dist_pred + 0.5 * dist_prev for dist_pred, dist_prev in zip(distances_from_pred, distances_from_prev)]
                else:
                    distances = distances_from_pred
                best_candidate = candidates[np.argmin(distances)]
                measurement = np.array([[np.float32(best_candidate[0])], [np.float32(best_candidate[1])]])

            if prev_pos is not None:
                dist = calculate_distance((measurement[0, 0], measurement[1, 0]), prev_pos)[0]
                if dist > threshold * (i - prev_idx):
                    # If the distance is too large, we assume that the detection is incorrect
                    ball_positions.append(None)
                else:
                    # If the distance is reasonable, we correct the Kalman filter
                    kf.correct(measurement)
                    prediction = kf.predict()
                    ball_positions.append((measurement[0, 0], measurement[1, 0]))
                    prev_pos = measurement
                    prev_idx = i
            else:
                # If this is the first detection, we just correct the Kalman filter
                kf.correct(measurement)
                ball_positions.append((measurement[0, 0], measurement[1, 0]))
                prev_pos = measurement
                prev_idx = i

        return ball_positions

    def get_team_mapping(self):  # This is pretty slow
        counts = {}
        # done = set()
        # First pass: Get the frequency of colors detected for each player
        for frame, coord in zip(self.frames, self.coords):
            curr_crops = [item["BBox"] for item in self.coords[coord]["Coordinates"]["Player"].values()]
            for player_id, item in self.coords[coord]["Coordinates"]["Player"].items():
                player_id = int(player_id)
                # if player_id in done:
                #     continue
                bbox = item["BBox"]
                x1, y1, x2, y2 = bbox
                curr_size = (x2 - x1) * (y2 - y1)
                # determine amount of overlap with other crops
                num_overlaps = 0
                max_overlap = 0
                for crop in curr_crops:
                    if crop == bbox:
                        continue
                    x1_, y1_, x2_, y2_ = crop
                    x_overlap = max(0, min(x2, x2_) - max(x1, x1_))
                    y_overlap = max(0, min(y2, y2_) - max(y1, y1_))
                    overlap = x_overlap * y_overlap
                    max_overlap = max(max_overlap, overlap)
                    if overlap > 0:
                        num_overlaps += 1
                prop_overlap = max_overlap / curr_size
                if prop_overlap > 0.5:
                    continue
                crop = frame[y1:y2, x1:x2]
                indiv_counts = self.detect_color(crop)
                if player_id not in counts:
                    counts[player_id] = {}
                for color, count in indiv_counts:
                    if color not in counts[player_id]:
                        counts[player_id][color] = 0
                    counts[player_id][color] += 1 - prop_overlap

                # if num_overlaps == 0 or prop_overlap < 0.05:
                #     done.add(player_id)

        out = {player_id: max(color_count, key=color_count.get) for player_id, color_count in counts.items()}

        # Second pass to fix outliers
        most_common = Counter(out.values()).most_common(2)
        id_map = {color: i for i, (color, _) in enumerate(most_common)}
        team_mapping = {}
        for player_id, color in out.items():
            if color in id_map:
                team_mapping[player_id] = id_map[color]
            else:  # This is an outlier -> not in the 2 most common colors
                # Go back to the original counts and pick the most common color out of the 2 most common colors
                color_count = counts[player_id]
                color_count = [(color, count) for color, count in color_count.items() if color in id_map]
                if len(color_count) == 0:
                    print(f"Unable to determine team for player {player_id}")
                    continue
                color_count = sorted(color_count, key=lambda x: x[1], reverse=True)

                team_mapping[player_id] = id_map[color_count[0][0]]

        return team_mapping

    def detect_color(self, image):  # Get counts based on HSV after using KMeans to segment
        # Ref: https://github.com/abdullahtarek/football_analysis/blob/main/team_assigner/team_assigner.py

        hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Use the RGB image to cluster (RGB works better than HSV for some reason)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(rgb_image.reshape(-1, 3))
        labels = kmeans.labels_
        labels = labels.reshape(image.shape[:2])
        corners = [labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]
        non_player_cluster = max(set(corners), key=corners.count)
        player_cluster = 1 if non_player_cluster == 0 else 0
        mask = labels == player_cluster
        player_mask = mask.astype(np.uint8) * 255
        hsv_image = cv2.bitwise_and(hsv_image, hsv_image, mask=player_mask)
        color_count = {color: 0 for color in color_ranges.keys()}
        masks = []
        for color, (lower, upper) in color_ranges.items():
            lower = np.array(lower, dtype="uint8")
            upper = np.array(upper, dtype="uint8")
            color_mask = cv2.inRange(hsv_image, lower, upper)
            color_mask = cv2.bitwise_and(color_mask, color_mask, mask=player_mask)
            masks.append(color_mask)

            color_count[color] += cv2.countNonZero(color_mask)

        color_count["red"] += color_count.pop("red2")

        color_count = [(color, count) for color, count in color_count.items() if count > 0]
        color_count = sorted(color_count, key=lambda x: x[1], reverse=True)

        return color_count


class KalmanFilter:
    def __init__(self, initial_state, initial_velocity):
        self.kalman = cv2.KalmanFilter(4, 2)
        self.kalman.statePre = np.array([initial_state[0], initial_state[1], initial_velocity[0], initial_velocity[1]], dtype=np.float32).reshape(-1, 1)
        self.kalman.transitionMatrix = np.array([[1, 0, 1, 0], [0, 1, 0, 1], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32)
        self.kalman.measurementMatrix = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], np.float32)
        self.kalman.processNoiseCov = np.array([[1, 0, 0, 0], [0, 1, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]], np.float32) * 1e-5
        self.kalman.measurementNoiseCov = np.array([[1, 0], [0, 1]], np.float32) * 1e-1
        self.kalman.errorCovPost = np.eye(4, dtype=np.float32)

    def predict(self):
        return self.kalman.predict()

    def correct(self, measurement):
        self.kalman.correct(measurement)
