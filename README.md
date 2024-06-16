# Eagle

<div align="center">
  <div>
<a target="_blank" href="https://colab.research.google.com/drive/1oGiZA0uj9MIarkhg2ty21WC4A0KXuhZX?authuser=3#scrollTo=h1KXqSjicSJU">
  <img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"/>
</a>

  </div>
  <p>
  <img src="demo.gif" width="640" height="400"/>
  </p>
  <br>
</div>

## Introduction 
Eagle converts football broadcast data from television feeds to tracking data useful for analysis and visualisation. It uses a collection of custom trained models and a variety of computer vision techniques to identify, track and obtain player and ball coordinates from each frame of broadcast data.

## Installation
Eagle works best with Python > 3.8 environments. 
```bash
git clone https://github.com/nreHieW/Eagle.git
cd Eagle
pip install -r requirements.txt
```

Next, you will need to download the weights of the models. 
```bash
cd eagle/models
sh get_weights.sh
cd ../../
```

## Basic Usage 
An inference script is provided. First, obtain a clip of the broadcast data that you want to use as an `.mp4` file. Eagle works best in clips where the camera position is relatively stable (ie no changes in angle). You can use [FFmpeg](https://ffmpeg.org/) to trim as necessary using the following example.

```bash
ffmpeg -ss 00:00:07 -to 00:00:15 -i video.mp4 -c copy input_video.mp4
```
Then run. You can also change the FPS depending on the granularity required using the `--fps` argument.
```bash
python main.py --video_path input_video.mp4 # Replace with your video name
```
The output data can be found in `output/(input_video)/`. For a detailed description of the output data format refer to [the section below](#output-explanation).

## Advanced Usage 
Eagle works best in CUDA enabled GPU environments or at the very least with [Apple Metal](https://developer.apple.com/metal/pytorch/). If you do not have access to such resources, feel free to use the [Google Colab provided](https://colab.research.google.com/drive/1oGiZA0uj9MIarkhg2ty21WC4A0KXuhZX?authuser=3#scrollTo=h1KXqSjicSJU). There are different variants of models provided - both [PyTorch](https://pytorch.org/) and [ONNX](https://onnx.ai/) formats as well as different sizes of the detector model. Feel free to choose the relevant format/sizes for your hardware requirements. Feel free to change the Tracker model used as well.

The Homography Calculation and Keypoint detection are pretty computationally expensive operations. If you have the compute requirements, feel free to invoke them more often as it might lead to more accurate results. They are controlled by the `num_homography` and `num_keypoint_detection` parameter which determine the number of times these operations are carried out per second respectively. 

### Capabilities
Given that Eagle was trained on consumer hardware, it is not 100% accurate especially when dealing with irregular camera angles or heavy occlusion of players and frames. While wider camera angles such as those used in scouting feeds are preferred, Eagle is trained on standard broadcast data so it would work just fine. While many attempts and heuristics are in place to handle the inaccuracies, it is still highly recommended to use `annotated.mp4` to determine if there are any errors in the output before using the data provided. 

Some common debugging strategies:
- The most common issue is incorrect team assignment. Team assignment is currently done purely based on heuristics. The solution is simply to edit the team mapping dictionary in the metadata.
- The second most common issue is when balls are not detected (given their size). This could cause erratic ball coordinates since Eagle interpolates the coordinates. One solution is to reduce the confidence threshold required for the detector (`detector_conf`).
- Lastly, a homography requires 4 points at minimum. Some camera angles makes this difficult. One solution is to reduce the confidence threshold for the keypoint detector (`keypoint_conf`)


## Output Explanation 
Outputs are stored in `output/(your video name)/`. All transformed coordinates use the UEFA pitch specifications (105 x 68). For detailed breakdown of the coordinate system, see `eagle/utils/pitch.py`.

1. **Metadata:** `metadata.json` contains the frames per second that Eagle used when processing the data and the team mapping of all the player ids detected in the video. 
2. **Debug Info:** Eagle automatically creates an annotated copy of the video provided to allow for quick mapping of player id to names and quick detection of any inaccuracies.
3. **Raw Data**: There are 2 raw data files provided. `raw_coordinates.json` is the detections and coordinates determined by the various models. `raw_data.json` is a pandas dataframe (read with `pd.read_json()`) which contains the `x,y` coordinates of the visible areas in that frame, players and ball for each frame in the video. `None` values indicate that nothing was detected for that particular id at that particular frame 
4. **Cleaned Data**: It is recommended to use `processed_data.json` for most use cases, similarly read as a pandas dataframe. It is modelled after [Statsbomb 360](https://github.com/statsbomb/open-data) and contains 3 columns: the visible areas in that frame, the coordinates from the video, and the transformed coordinates for each frame. Each coordinate value is a list of json objects with the type of object (Ball, Player or Goalkeeper), the coordinates and the team (for Player) identified in that particular freeze frame.


## Future Improvements
The quality of the model as well as additional features (such as ReID) can make Eagle stronger and more robust. Any contribution to Eagle is deeply appreciated! 

Feel free to use the trained models for your own custom usecases.

## Acknowledgements
Huge acknowledgements goes to the following projects that have helped the development of Eagle tremendously:
- Code in a Jiffy's [code](https://github.com/abdullahtarek/football_analysis) and [YouTube](https://www.youtube.com/watch?v=neBZ6huolkg&)
- Roboflow's [sports repo](https://github.com/roboflow/sports)
- Training code for keypoint detection in PyTorch by [tlpss](https://github.com/tlpss/keypoint-detection)
- The [Soccernet](https://github.com/SoccerNet) team for the data they have provided
- The [winning team](https://github.com/NikolasEnt/soccernet-calibration-sportlight) from the Soccernet Camera Calibration Challenge 2023
- An easy way to use Trackers by [mikel-brostrom](https://github.com/mikel-brostrom/yolo_tracking)