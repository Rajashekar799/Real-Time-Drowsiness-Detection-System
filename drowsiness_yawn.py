# python drowsiness_yawn.py --webcam 0 --alarm Alert.wav

from scipy.spatial import distance as dist
from imutils.video import VideoStream
from imutils import face_utils
from threading import Thread
from collections import deque
import numpy as np
import argparse
import imutils
import time
import dlib
import cv2
import playsound
import os
import ctypes


alarm_playing = False


def sound_alarm(path):
    global alarm_playing
    # Play sound only once, not in loop
    if not os.path.exists(path):
        print(f"Alarm file not found: {path}")
        alarm_playing = False
        return
    
    try:
        playsound.playsound(path)
    except Exception as e:
        print(f"Error playing sound: {e}")
    finally:
        alarm_playing = False


def trigger_alarm(path, alarm_sound_time, alarm_cooldown, block_if_yawn=False, yawn_alert_until=0):
    global alarm_playing
    current_time = time.time()

    # While yawn alert is active, suppress all non-yawn sirens.
    if block_if_yawn and current_time < yawn_alert_until:
        return False, alarm_sound_time

    if alarm_playing or (current_time - alarm_sound_time) < alarm_cooldown:
        return False, alarm_sound_time

    alarm_playing = True
    alarm_sound_time = current_time
    if path != "":
        t = Thread(target=sound_alarm, args=(path,))
        t.daemon = True
        t.start()
    else:
        alarm_playing = False

    return True, alarm_sound_time

def eye_aspect_ratio(eye):
    A = dist.euclidean(eye[1], eye[5])
    B = dist.euclidean(eye[2], eye[4])

    C = dist.euclidean(eye[0], eye[3])

    ear = (A + B) / (2.0 * C)

    return ear

def final_ear(shape):
    (lStart, lEnd) = face_utils.FACIAL_LANDMARKS_IDXS["left_eye"]
    (rStart, rEnd) = face_utils.FACIAL_LANDMARKS_IDXS["right_eye"]

    leftEye = shape[lStart:lEnd]
    rightEye = shape[rStart:rEnd]

    leftEAR = eye_aspect_ratio(leftEye)
    rightEAR = eye_aspect_ratio(rightEye)

    ear = (leftEAR + rightEAR) / 2.0
    return (ear, leftEye, rightEye)

def lip_distance(shape):
    top_lip = shape[50:53]
    top_lip = np.concatenate((top_lip, shape[61:64]))

    low_lip = shape[56:59]
    low_lip = np.concatenate((low_lip, shape[65:68]))

    top_mean = np.mean(top_lip, axis=0)
    low_mean = np.mean(low_lip, axis=0)

    distance = abs(top_mean[1] - low_mean[1])
    return distance


def fit_frame_to_screen(frame, screen_width, screen_height, header_height=70):
    frame_height, frame_width = frame.shape[:2]
    available_height = screen_height - header_height
    scale = min(screen_width / frame_width, available_height / frame_height)
    resized_width = int(frame_width * scale)
    resized_height = int(frame_height * scale)

    resized_frame = cv2.resize(frame, (resized_width, resized_height))
    canvas = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    canvas[:header_height, :] = (0, 191, 255)

    x_offset = (screen_width - resized_width) // 2
    y_offset = header_height + (available_height - resized_height) // 2
    canvas[y_offset:y_offset + resized_height, x_offset:x_offset + resized_width] = resized_frame

    title = "Driver Drowsiness Detector"
    text_size = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)[0]
    text_x = (screen_width - text_size[0]) // 2
    text_y = (header_height + text_size[1]) // 2
    cv2.putText(canvas, title, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
    return canvas


def draw_compact_metrics(frame, ear_value, yawn_value):
    panel_width = 160
    panel_height = 92
    x1 = frame.shape[1] - panel_width - 8
    y1 = 8
    x2 = x1 + panel_width
    y2 = y1 + panel_height

    # Small, clean metrics panel to avoid large distracting text on the video.
    overlay = frame.copy()
    cv2.rectangle(overlay, (x1, y1), (x2, y2), (25, 25, 25), -1)
    cv2.addWeighted(overlay, 0.55, frame, 0.45, 0, frame)

    font = cv2.FONT_HERSHEY_SIMPLEX
    scale = 0.45
    thickness = 1
    color = (200, 230, 255)

    cv2.putText(frame, f"EAR: {ear_value:.2f}", (x1 + 8, y1 + 20), font, scale, color, thickness)
    cv2.putText(frame, f"ETHR: {EYE_AR_THRESH:.2f}", (x1 + 8, y1 + 40), font, scale, color, thickness)
    cv2.putText(frame, f"YAWN: {yawn_value:.2f}", (x1 + 8, y1 + 60), font, scale, color, thickness)
    cv2.putText(frame, f"YTHR: {YAWN_THRESH:.2f}", (x1 + 8, y1 + 80), font, scale, color, thickness)


ap = argparse.ArgumentParser()
ap.add_argument("-w", "--webcam", type=int, default=0,
                help="index of webcam on system")
ap.add_argument("-a", "--alarm", type=str, default=os.path.join(os.path.dirname(__file__), "Alert.wav"), help="path alarm .WAV file")
args = vars(ap.parse_args())

EYE_AR_THRESH = 0.3
EYE_AR_CONSEC_FRAMES = 30
YAWN_THRESH = 20
YAWN_SMOOTHING_WINDOW = 5
YAWN_CONSEC_FRAMES = 8
alarm_status = False
alarm_status2 = False
COUNTER = 0
yawn_counter = 0
yawn_distances = deque(maxlen=YAWN_SMOOTHING_WINDOW)
yawn_alert_until = 0
drowsiness_alert_until = 0  # Persist drowsiness message for duration
face_detection_alert_until = 0  # Persist face detection message for duration
poor_lighting_alert_until = 0  # Persist poor lighting message for duration
face_not_detected_frames = 0
face_detection_threshold = 75  # Alert only after 75 frames (3 sec) without face detection - allows normal glances
alarm_sound_time = 0  # Track when last alarm sound played
alarm_cooldown = 3.0  # Minimum seconds between alarm sounds
alert_display_duration = 2.5  # How long to show alert message (seconds)

print("-> Loading the predictor and detector...")
#detector = dlib.get_frontal_face_detector()
detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")    #Faster but less accurate
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')


print("-> Starting Video Stream")
vs = VideoStream(src=args["webcam"]).start()
#vs= VideoStream(usePiCamera=True).start()       //For Raspberry Pi
time.sleep(1.0)

cv2.namedWindow("Frame", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Frame", 1280, 720)

screen_width = ctypes.windll.user32.GetSystemMetrics(0)
screen_height = ctypes.windll.user32.GetSystemMetrics(1)

while True:

    frame = vs.read()
    frame = imutils.resize(frame, width=450)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    #rects = detector(gray, 0)
    rects = detector.detectMultiScale(gray, scaleFactor=1.1,
		minNeighbors=5, minSize=(30, 30),
		flags=cv2.CASCADE_SCALE_IMAGE)

    # Handle face detection failure (only alert if face missing for sustained time)
    if len(rects) == 0:
        face_not_detected_frames += 1
        if face_not_detected_frames >= face_detection_threshold:
            current_time = time.time()
            if alarm_status == False:
                alarm_status = True
                played, alarm_sound_time = trigger_alarm(
                    args["alarm"],
                    alarm_sound_time,
                    alarm_cooldown,
                    block_if_yawn=True,
                    yawn_alert_until=yawn_alert_until,
                )
                if played:
                    face_detection_alert_until = current_time + alert_display_duration
    else:
        face_not_detected_frames = 0
        alarm_status = False

    #for rect in rects:
    for (x, y, w, h) in rects:
        rect = dlib.rectangle(int(x), int(y), int(x + w),int(y + h))

        shape = predictor(gray, rect)
        shape = face_utils.shape_to_np(shape)

        eye = final_ear(shape)
        ear = eye[0]
        leftEye = eye [1]
        rightEye = eye[2]

        distance = lip_distance(shape)
        yawn_distances.append(distance)
        smoothed_yawn = float(np.mean(yawn_distances))

        leftEyeHull = cv2.convexHull(leftEye)
        rightEyeHull = cv2.convexHull(rightEye)
        cv2.drawContours(frame, [leftEyeHull], -1, (0, 255, 0), 1)
        cv2.drawContours(frame, [rightEyeHull], -1, (0, 255, 0), 1)

        lip = shape[48:60]
        cv2.drawContours(frame, [lip], -1, (0, 255, 0), 1)

        # Check for poor lighting (unable to detect eyes accurately)
        if ear < 0.05 or ear > 1.0:  # Invalid EAR range indicates poor lighting
            current_time = time.time()
            poor_lighting_alert_until = current_time + alert_display_duration
        else:
            if ear < EYE_AR_THRESH:
                COUNTER += 1

                if COUNTER >= EYE_AR_CONSEC_FRAMES:
                    current_time = time.time()
                    if alarm_status == False:
                        alarm_status = True
                        played, alarm_sound_time = trigger_alarm(
                            args["alarm"],
                            alarm_sound_time,
                            alarm_cooldown,
                            block_if_yawn=True,
                            yawn_alert_until=yawn_alert_until,
                        )
                        if played:
                            drowsiness_alert_until = current_time + alert_display_duration

            else:
                COUNTER = 0
                alarm_status = False

            if smoothed_yawn > YAWN_THRESH:
                yawn_counter += 1
            else:
                yawn_counter = 0
                alarm_status2 = False

            if yawn_counter >= YAWN_CONSEC_FRAMES and time.time() >= yawn_alert_until:
                    yawn_alert_until = time.time() + alert_display_duration
                    yawn_counter = 0
                    COUNTER = 0  # Reset eye-closure counter when yawning
                    alarm_status = False  # Cancel drowsiness alarm status
                    drowsiness_alert_until = 0  # Prevent stale eyes-closed message during yawns
                    face_detection_alert_until = 0  # Prevent overlap with face detection siren/message
                    poor_lighting_alert_until = 0  # Prevent overlap with poor lighting siren/message
                    played, alarm_sound_time = trigger_alarm(
                        args["alarm"],
                        alarm_sound_time,
                        alarm_cooldown,
                        block_if_yawn=False,
                    )
                    if played:
                        alarm_status2 = True


        draw_compact_metrics(frame, ear, smoothed_yawn)

    display_frame = fit_frame_to_screen(frame, screen_width, screen_height)
    
    # Display drowsiness alert message while siren rings
    if time.time() < drowsiness_alert_until and time.time() >= yawn_alert_until:
        cv2.putText(display_frame, "DROWSINESS DETECTED - Eyes Closed!", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display yawn alert message while siren rings
    if time.time() < yawn_alert_until:
        cv2.putText(display_frame, "DROWSINESS DETECTED - Yawning!", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display face not detected alert message while siren rings
    if time.time() < face_detection_alert_until and time.time() >= yawn_alert_until:
        cv2.putText(display_frame, "Face Out of Frame - Monitoring Paused", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    
    # Display poor lighting alert message while siren rings
    if time.time() < poor_lighting_alert_until and time.time() >= yawn_alert_until:
        cv2.putText(display_frame, "Poor Lighting - Unable to detect eyes", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
    cv2.imshow("Frame", display_frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

cv2.destroyAllWindows()
vs.stop()
