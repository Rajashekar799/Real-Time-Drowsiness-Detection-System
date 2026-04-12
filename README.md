# Real-Time-Drowsiness-Detection-System

Drowsiness detection is a safety technology that can prevent accidents that are caused by drivers who fell asleep while driving. The objective of this project is to build a drowsiness detection system that will detect drowsiness through the implementation of computer vision system that automatically detects drowsiness in real-time from a live video stream and then alert the user with an alarm notification.

## Motivation 
According to the National Highway Traffic Safety Administration, every year about 100,000 police-reported crashes involve drowsy driving. These crashes result in more than 1,550 fatalities and 71,000 injuries. The real number may be much higher, however, as it is difficult to determine whether a driver was drowsy at the time of a crash. So, we tried to build a system, that detects whether a person is drowsy and alert him.

## Built With

* [OpenCV Library](https://opencv.org/) - Most used computer vision library. Highly efficient. Facilitates real-time image processing.
* [imutils library](https://github.com/jrosebr1/imutils) -  A collection of helper functions and utilities to make working with OpenCV easier.
* [Dlib library](http://dlib.net/) - Implementations of state-of-the-art CV and ML algorithms (including face recognition).
* [scikit-learn library](https://scikit-learn.org/stable/) - Machine learning in Python. Simple. Efficient. Beautiful, easy to use API.
* [Numpy](http://www.numpy.org/) - NumPy is the fundamental package for scientific computing with Python. 


## Getting Started

These steps let you clone and run this project on another system (Windows, Linux, or macOS).

### Prerequisites

1. Python 3.8+ installed and available in PATH.
1. Git installed.
1. CMake installed (required by some dlib builds).
1. A webcam.
1. The dlib landmark model file: `shape_predictor_68_face_landmarks.dat` in the project root.

### Clone and setup

1. Clone the repository.

    ```bash
    git clone https://github.com/Rajashekar799/Real-Time-Drowsiness-Detection-System.git
    ```

1. Move into the project directory.

    ```bash
    cd Real-Time-Drowsiness-Detection-System
    ```

1. Create a virtual environment.

    Windows (PowerShell/cmd):
    ```bash
    python -m venv .venv
    .venv\Scripts\activate
    ```

    Linux/macOS:
    ```bash
    python3 -m venv .venv
    source .venv/bin/activate
    ```

1. Install dependencies.

    ```bash
    pip install -r requirements.txt
    ```

### Run

Use default webcam (index 0):

```bash
python drowsiness_yawn.py --webcam 0
```

Use a custom alarm file:

```bash
python drowsiness_yawn.py --webcam 0 --alarm Alert.wav
```

### Notes for other systems

1. If `dlib` fails to install on Linux, follow this guide: [dlib Linux install help](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf).
1. If using Windows and `dlib` build fails, install CMake, restart terminal, then retry `pip install -r requirements.txt`.
1. Ensure these files exist in project root before running:
    1. `drowsiness_yawn.py`
    1. `haarcascade_frontalface_default.xml`
    1. `shape_predictor_68_face_landmarks.dat`

## Algorithm

1. Capture frames from the webcam.
2. Detect the face using the Haar cascade classifier.
3. Extract facial landmarks using dlib's 68-point shape predictor.
4. Compute the eye aspect ratio (EAR) from both eyes.
5. If the EAR stays below `0.3` for `30` consecutive frames, trigger the eyes-closed drowsiness alert.
6. Compute the lip distance from the mouth landmarks.
7. Smooth yawn distance over recent frames and trigger yawn alert when it stays above `20` for `8` consecutive frames.
8. If face is not detected for `75` consecutive frames (3 seconds), trigger face out of frame alert.
9. If EAR values are invalid (< 0.05 or > 1.0), trigger poor lighting alert.
10. Display compact on-screen metrics: `EAR`, `ETHR`, `YAWN`, and `YTHR`.
11. Prevent overlapping sirens by allowing only one alarm playback at a time.
12. Resize the frame for display and draw eye/mouth contours on the video feed.

For a more detailed explanation of the implementation, see [drowsiness_yawn.py](drowsiness_yawn.py).

## Test Cases

| TC ID | Module | Test Case Description | Expected Output | Status |
|-------|--------|----------------------|-----------------|--------|
| 1 | System | Start driver drowsiness monitoring | Click "Start Behaviour Monitoring Using Webcam" | ✅ Pass |
| 2 | Camera | Capture driver face | Webcam activated and monitoring started successfully | ✅ Pass |
| 3 | Eye Detection | Detect open eyes | Face detected and landmarks displayed | ✅ Pass |
| 4 | Drowsiness Detection | Detect prolonged eye closure | Status displayed: "Eyes Open" with EAR value, then "DROWSINESS DETECTED - Eyes Closed!" with alarm | ✅ Pass |
| 5 | Yawn Detection | Driver yawning | Message displayed: "DROWSINESS DETECTED - Yawning!" with alarm sound | ✅ Pass |
| 6 | Camera | Driver moves out of frame | Message displayed: "Face Out of Frame - Monitoring Paused" with alarm | ✅ Pass |
| 7 | Camera | Poor lighting conditions | Face not detected — monitoring paused | ✅ Pass |
| 8 | Drowsiness Detection | Driver moves out of frame | Face not detected — monitoring paused | ✅ Pass |

## Detection Thresholds

- **Eye Aspect Ratio (EAR)**: `0.3` - Eyes are considered closed when EAR drops below this value
- **Drowsiness Frames**: `30` consecutive frames with closed eyes before alerting (~1 second at 30 FPS)
- **Yawn Distance**: `20` - Smoothed mouth-distance threshold to detect yawning
- **Yawn Consecutive Frames**: `8` frames - Yawn must persist before alerting
- **Face Detection Threshold**: `75` frames (3 seconds) - Time face must be missing before alerting
- **Alert Display Duration**: `2.5` seconds - How long alert message persists
- **Alert Cooldown**: `3.0` seconds - Minimum time between consecutive alerts

## Features

✅ **Real-time Face Detection** - Uses Haar Cascade for fast face detection
✅ **Eyes Closed Detection** - Monitors eye aspect ratio (EAR) to detect drowsiness
✅ **Yawn Detection** - Detects mouth opening to identify yawning
✅ **Face Out of Frame Alert** - Alerts when driver looks away for extended period
✅ **Poor Lighting Detection** - Handles low-light conditions gracefully
✅ **Audio Alerts** - Single beep alarm for each detection
✅ **Visual Messages** - Clear on-screen messages explaining each alert
✅ **Smart Cooldown** - Prevents alert spam with 3-second cooldown between alerts
✅ **Normal Behavior Support** - 3-second threshold for face detection allows normal driving behaviors (mirror checks, etc.)
✅ **Compact Metrics HUD** - Small `EAR/ETHR/YAWN/YTHR` panel that stays readable without clutter
✅ **No Siren Overlap** - Global alarm gate ensures only one siren plays at a time

## Testing and Results in Real-World Scenario:

The tests were conducted by running the webcam feed through the actual detection pipeline in the code. The observed checks are based on face detection, eye aspect ratio, and lip distance.

Result: When a face is visible to the camera, the script detects facial landmarks and displays the EAR and YAWN values on the frame.

Test case 2: Eyes open

Result: When the eyes remain open, the EAR stays above the threshold and the drowsiness alarm does not trigger.

Test case 3: Eyes closed for multiple frames

Result: When the EAR stays below 0.3 for 30 consecutive frames, the script shows `DROWSINESS DETECTED - Eyes Closed!` and starts the alarm sound.

Test case 4: Yawning

Result: When smoothed yawn distance stays above 20 for multiple frames, the script shows `DROWSINESS DETECTED - Yawning!` and plays the alarm sound.

Test case 5: Face not clearly visible

Result: If the face is not detected clearly enough for the landmark predictor, EAR and yawn checks are not updated for that frame.

The system was tested in real-world webcam usage, and the alerts were triggered only when the face landmarks were detected and the threshold conditions were met.

## Troubleshooting

### 1. `shape_predictor_68_face_landmarks.dat` not found

Symptom:
- App exits at startup with a file-not-found error.

Fix:
- Put `shape_predictor_68_face_landmarks.dat` in the project root folder (same folder as `drowsiness_yawn.py`).

### 2. `dlib` installation fails

Symptom:
- `pip install -r requirements.txt` fails while building/installing `dlib`.

Fix:
- Windows: install CMake, restart terminal, then run install again.
- Linux: install build tools and CMake, then retry.
- Use the guide linked in this README if needed.

### 3. Camera does not open

Symptom:
- Black frame, no video, or immediate crash.

Fix:
- Try different webcam index values:

```bash
python drowsiness_yawn.py --webcam 0
python drowsiness_yawn.py --webcam 1
python drowsiness_yawn.py --webcam 2
```

- Close other apps that may be using the webcam.

### 4. Alarm sound not heard

Symptom:
- Visual alert appears, but no sound.

Fix:
- Ensure `Alert.wav` exists in project root, or pass a valid file path using `--alarm`.
- Check OS volume/output device.

### 5. Yawn alert not triggering

Symptom:
- Mouth opens but no yawn alert appears.

Fix:
- Confirm on-screen `YAWN` rises above `YTHR` (`20.00`) for several frames.
- Ensure face and mouth landmarks are clearly visible (good lighting, camera at eye level).
- Move closer to camera if facial landmarks are unstable.

### 6. Face out-of-frame alert triggers too often

Symptom:
- Frequent "Face Out of Frame" warnings during normal movement.

Fix:
- Keep face centered and improve lighting.
- Ensure webcam is stable and not shaking.
- If needed, increase `face_detection_threshold` in code.

## Future Scope

Smart phone application: It can be implemented as a smart phone application that can be installed on smartphones. The driver can start the application after placing the camera where it is focused on the driver.

## References

IEEE standard Journal Paper,

[1]	Facial Features Monitoring for Real Time Drowsiness Detection by Manu B.N, 2016 12th International Conference on Innovations in Information Technology (IIT) [Pg. 78-81] (https://ieeexplore.ieee.org/document/7880030)

[2]	Real Time Drowsiness Detection using Eye Blink Monitoring by Amna Rahman Department of Software Engineering Fatima Jinnah Women University 2015 National Software Engineering Conference (NSEC 2015) (https://ieeexplore.ieee.org/document/7396336)

Websites referred:

1.	https://www.codeproject.com/Articles/26897/TrackEye-Real-Time-Tracking-Of-Human-Eyes-
2.	https://realpython.com/face-recognition-with-python/
3.	https://www.pyimagesearch.com/2017/04/03/facial-landmarks-dlib-opencv
4.	https://www.pyimagesearch.com/2017/04/10/detect-eyes-nose-lips-jaw-dlib-opencv
5.	https://www.codeproject.com/Articles/26897/TrackEye-Real-Time-Tracking-Of-HumanEyesUsing-a
6.	https://docs.opencv.org/3.4/d7/d8b/tutorial_py_face_detection.html
7.	https://www.learnopencv.com/training-better-haar-lbp-cascade-eye-detector-opencv/


## Author

**Rajashekar Rikkula**




