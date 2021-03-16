# On-road-Obstacle-Detection
Detecting objects on road in the context of Autonomous Vehicle

The central concept of this application is to detect objects that comes in reference to the vehicle using a camera. This application can detect almost all the object that are present on road. Here, objects are detected by bounding a box around it and mentioning probability of that detection. Bounding boxes also moves as the object moves anywhere that means this application can detect objects even in motion.

As of now, this program contains two features for detection. One is live detection using any camera and another is detection through input video.

<img align="" alt="Workflow" src="https://github.com/ankit-kaushal/On-road-Obstacle-Detection/blob/main/Screenshots/workflow.png" width="750" height="420"/>

## Technology stack:
* Python (Programming Language)
* YOLOv3 (Algorithm)
* OpenCV (Framework)
* COCO Dataset

#### To start the application, first install required libs:

<ol>
<li>pip install opencv-python</li>
<li>pip install numpy</li>
<li>pip install tk</li>
</ol>


#### Also, Download weights flile from <a href="https://pjreddie.com/media/files/yolov3.weights">Here</a>

#### To run the program Run " python3 final_code.py.py " in terminal in linux or " python final_code.py.py " in command prompt in windows. A GUI window will open up click on the required button and the application will start.
