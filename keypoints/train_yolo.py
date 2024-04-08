from datasets import load_dataset
import os
import yaml
from tqdm import tqdm
from ultralytics import YOLO

dataset = load_dataset("nreHieW/SoccerNet_Field_Segmentation")

INTERSECTION_TO_PITCH_POINTS = {
    0: "L_GOAL_TL_POST",
    1: "L_GOAL_TR_POST",
    2: "L_GOAL_BL_POST",
    3: "L_GOAL_BR_POST",
    4: "L_GOAL_AREA_BR_CORNER",
    5: "L_GOAL_AREA_TR_CORNER",
    6: "L_GOAL_AREA_BL_CORNER",
    7: "L_GOAL_AREA_TL_CORNER",
    8: "L_PENALTY_AREA_BR_CORNER",
    9: "L_PENALTY_AREA_TR_CORNER",
    10: "L_PENALTY_AREA_BL_CORNER",
    11: "L_PENALTY_AREA_TL_CORNER",
    12: "BL_PITCH_CORNER",
    13: "TL_PITCH_CORNER",
    14: "B_TOUCH_AND_HALFWAY_LINES_INTERSECTION",
    15: "T_TOUCH_AND_HALFWAY_LINES_INTERSECTION",
    16: "R_PENALTY_AREA_BL_CORNER",
    17: "R_PENALTY_AREA_TL_CORNER",
    18: "R_PENALTY_AREA_BR_CORNER",
    19: "R_PENALTY_AREA_TR_CORNER",
    20: "R_GOAL_AREA_BL_CORNER",
    21: "R_GOAL_AREA_TL_CORNER",
    22: "R_GOAL_AREA_BR_CORNER",
    23: "R_GOAL_AREA_TR_CORNER",
    24: "R_GOAL_TL_POST",
    25: "R_GOAL_TR_POST",
    26: "R_GOAL_BL_POST",
    27: "R_GOAL_BR_POST",
    28: "BR_PITCH_CORNER",
    29: "TR_PITCH_CORNER",
    30: "CENTER_CIRCLE_TANGENT_TR",
    31: "CENTER_CIRCLE_TANGENT_TL",
    32: "CENTER_CIRCLE_TANGENT_BR",
    33: "CENTER_CIRCLE_TANGENT_BL",
    34: "CENTER_CIRCLE_TR",
    35: "CENTER_CIRCLE_TL",
    36: "CENTER_CIRCLE_BR",
    37: "CENTER_CIRCLE_BL",
    38: "CENTER_CIRCLE_R",
    39: "CENTER_CIRCLE_L",
    40: "T_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION",
    41: "B_HALFWAY_LINE_AND_CENTER_CIRCLE_INTERSECTION",
    42: "CENTER_MARK",
    43: "LEFT_CIRCLE_R",
    44: "BL_16M_LINE_AND_PENALTY_ARC_INTERSECTION",
    45: "TL_16M_LINE_AND_PENALTY_ARC_INTERSECTION",
    46: "LEFT_CIRCLE_TANGENT_T",
    47: "LEFT_CIRCLE_TANGENT_B",
    48: "L_PENALTY_MARK",
    49: "L_MIDDLE_PENALTY",
    50: "RIGHT_CIRCLE_L",
    51: "BR_16M_LINE_AND_PENALTY_ARC_INTERSECTION",
    52: "TR_16M_LINE_AND_PENALTY_ARC_INTERSECTION",
    53: "RIGHT_CIRCLE_TANGENT_T",
    54: "RIGHT_CIRCLE_TANGENT_B",
    55: "R_PENALTY_MARK",
    56: "R_MIDDLE_PENALTY",
}


dataset = load_dataset("nreHieW/SoccerNet_Field_Keypoints")

os.makedirs("images", exist_ok=True)
os.makedirs("labels", exist_ok=True)
os.makedirs("images/train", exist_ok=True)
os.makedirs("images/val", exist_ok=True)
os.makedirs("images/test", exist_ok=True)
os.makedirs("labels/train", exist_ok=True)
os.makedirs("labels/val", exist_ok=True)
os.makedirs("labels/test", exist_ok=True)

for split in dataset.keys():
    ds = dataset[split]
    n = len(ds)
    for idx in tqdm(range(n)):
        item = ds[idx]
        img = item["image"]
        width, height = img.size
        box_width = 1 / width
        box_height = 1 / height
        img.save(f"images/{split}/{idx}.jpg")
        keypoints = item["keypoints"]

        label_str = ""
        for class_idx in INTERSECTION_TO_PITCH_POINTS:
            curr = keypoints[class_idx]
            if (curr is not None) and not any(math.isnan(x) for x in curr):
                x = curr[0] / width
                y = curr[1] / height
                box_width = (
                    min(box_width, 1 - x) if x + box_width / 2 > 1 else box_width
                )
                box_height = (
                    min(box_height, 1 - y) if y + box_height / 2 > 1 else box_height
                )
                label_str += f"{class_idx} {x} {y} {box_width} {box_height}\n"

        label_str = label_str.strip("\n")
        with open(f"labels/{split}/{idx}.txt", "w") as f:
            f.write(label_str)

cfg = {
    "path": "/content",
    "train": "images/train",
    "val": "images/val",
    "test": "images/test",
    "names": INTERSECTION_TO_PITCH_POINTS,
}

with open("config.yml", "w") as f:
    yaml.dump(cfg, f, default_flow_style=False)


# Load a model
model = YOLO("yolov8n.pt")  # load a pretrained model (recommended for training)

# Train the model
results = model.train(data="config.yml", epochs=1, imgsz=640)
