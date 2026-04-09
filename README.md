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

These instructions will get you a copy of the project up and running on your local machine for development and testing purposes.

1. Install and set up Python 3.
1. Install [cmake](https://github.com/Kitware/CMake/releases/download/v3.13.3/cmake-3.13.3-win64-x64.zip) in your system

## Running the application

1. Clone the repository. 

    ```
    git clone https://github.com/Rajashekar799/Real-Time-Drowsiness-Detection-System.git
    ```
    
1. Move into the project directory. 

    ```
    cd Real-Time-Drowsiness-Detection-System
    ```
 
1. (Optional) Running it in a virtual environment. 

   1. Downloading and installing _virtualenv_. 
   ```
   pip install virtualenv
   ```
   
   2. Create the virtual environment in Python 3.
   
   ```
    virtualenv -p C:\Python37\python.exe test_env
   ```    
   
   3. Activate the test environment.     
   
        1. For Windows:
        ```
        test_env\Scripts\Activate
        ```        
        
        2. For Unix:
        ```
        source test_env/bin/activate
        ```    

1. Install all the required libraries, by installing the requirements.txt file.

    ```
    pip install -r requirements.txt
    ```
    
1. Installing the dlib library.
     
    1. If you are using a Unix machine, and are facing some issues while trying to install the dlib library, follow [this guide](https://gist.github.com/ageitgey/629d75c1baac34dfa5ca2a1928a7aeaf).  
    
    1. If you are using a Windows machine, install cmake and restart your terminal. 
    
1. Run the application.

    ```
    python drowsiness_yawn.py --webcam 0 --alarm Alert.wav
    ```

## Alogorithm

1. Capture frames from the webcam.
2. Detect the face using the Haar cascade classifier.
3. Extract facial landmarks using dlib's 68-point shape predictor.
4. Compute the eye aspect ratio (EAR) from both eyes.
5. If the EAR stays below `0.3` for `30` consecutive frames, trigger the drowsiness alert.
6. Compute the lip distance from the mouth landmarks.
7. If the lip distance goes above `20`, trigger the yawn alert and play the alarm.
8. Resize the frame for display and draw the eye and mouth contours on the video feed.
9. Show the `Yawn Alert` message briefly for about `2.5` seconds when yawning is detected.

For a more detailed explanation of the implementation, see [drowsiness_yawn.py](drowsiness_yawn.py).

## Testing and Results in Real-World Scenario:

The tests were conducted by running the webcam feed through the actual detection pipeline in the code. The observed checks are based on face detection, eye aspect ratio, and lip distance.

Test case 1: Face detected under normal lighting

Result: When a face is visible to the camera, the script detects facial landmarks and displays the EAR and YAWN values on the frame.

Test case 2: Eyes open

Result: When the eyes remain open, the EAR stays above the threshold and the drowsiness alarm does not trigger.

Test case 3: Eyes closed for multiple frames

Result: When the EAR stays below 0.3 for 30 consecutive frames, the script shows `DROWSINESS ALERT!` and starts the alarm sound.

Test case 4: Yawning

Result: When the lip distance goes above 20, the script shows `Yawn Alert` for a short period and plays the alarm sound.

Test case 5: Face not clearly visible

Result: If the face is not detected clearly enough for the landmark predictor, EAR and yawn checks are not updated for that frame.

The system was tested in real-world webcam usage, and the alerts were triggered only when the face landmarks were detected and the threshold conditions were met.

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




