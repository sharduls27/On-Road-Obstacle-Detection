# importing all the required modules
import cv2
import numpy as np
from tkinter import *
from PIL import ImageTk


def viddetect():
    # To open Video
    cap = cv2.VideoCapture("test3.mp4")
    whT = 288  # we declare width and height and T is for Target
    confThreshold = 0.5  # We will use it as if it is above 50% the it is good detection
    nmsThreshold = 0.4  # we will use it to reduce no of bounding boxes, just go on reducing its values

    classesFile = 'coco.names'  # Path for Object File
    classNames = []  # a list for storing all the object

    # Opening and Extaracing Object Class File

    with open(classesFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
    # print(classNames)           #printing names of Objects
    # print(len(classNames))      #printing the number of Objects

    colors = np.random.uniform(0, 150, size=(len(classNames), 3))

    # To create a network we use weight file and cfg file

    modelConfiguration = 'yolov3.cfg'  # import cfg file
    modelWeights = 'yolov3.weights'  # import weight file

    net = cv2.dnn.readNet(modelConfiguration, modelWeights)  # Creating a network, here we ae reading from darknet

    # Now we declare that we are going to use opencv as Backend and we want to use CPU
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def findObjects(outputs, img):
        hT, wT, cT = img.shape  # by this we find height, width and channels of our image
        # We declare 3 lists in which we will store values
        bbox = []  # this is the first list which will contain x,y,width and height
        classIds = []  # this list will contain all the class id's
        confs = []  # this list will contain confidence values

        for output in outputs:
            for det in output:
                # first remove first 5 elements and to find heighest probabilities
                scores = det[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                # filtering objects
                if confidence > confThreshold:
                    w, h = int(det[2] * wT), int(det[3] * hT)  # we multiply by wT or hT to get the pixel value
                    x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                    bbox.append([x, y, w, h])
                    classIds.append(classId)
                    confs.append(float(confidence))

        indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

        for i in indices:
            i = i[0]
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            color = colors[classIds[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # creating a loop for detection
    while True:
        success, img = cap.read()  # this will give the image

        # now input image to the network, but network accepts/understands only particular type of format and
        # this format is basically blob, So we convert our image to blob
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1,
                                     crop=False)  # this will convert our image to blob
        net.setInput(blob)

        layerNames = net.getLayerNames()  # This will give all the names of our layers
        # print(layerNames)
        outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        # print(outputNames)
        # print(net.getUnconnectedOutLayers())

        outputs = net.forward(outputNames)

        findObjects(outputs, img)

        cv2.imshow('Detect', img)  # to display the image
        cv2.waitKey(1)  # delay it for 1 millisecond


def camdetect():
    # To open web cam
    cap = cv2.VideoCapture(1)
    whT = 288  # we declare width and height and T is for Target
    confThreshold = 0.5  # We will use it as if it is above 50% the it is good detection
    nmsThreshold = 0.4  # we will use it to reduce no of bounding boxes, just go on reducing its values

    classesFile = 'coco.names'  # Path for Object File
    classNames = []  # a list for storing all the object

    # Opening and Extaracing Object Class File

    with open(classesFile, 'rt') as f:
        classNames = f.read().rstrip('\n').split('\n')
    # print(classNames)           #printing names of Objects
    # print(len(classNames))      #printing the number of Objects

    colors = np.random.uniform(0, 255, size=(len(classNames), 3))

    # To create a network we use weight file and cfg file

    modelConfiguration = 'yolov3.cfg'  # import cfg file
    modelWeights = 'yolov3.weights'  # import weight file

    net = cv2.dnn.readNet(modelConfiguration, modelWeights)  # Creating a network, here we ae reading from darknet

    # Now we declare that we are going to use opencv as Backend and we want to use CPU
    net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
    net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

    def findObjects(outputs, img):
        hT, wT, cT = img.shape  # by this we find height, width and channels of our image
        # We declare 3 lists in which we will store values
        bbox = []  # this is the first list which will contain x,y,width and height
        classIds = []  # this list will contain all the class id's
        confs = []  # this list will contain confidence values

        for output in outputs:
            for det in output:
                # first remove first 5 elements and to find heighest probabilities
                scores = det[5:]
                classId = np.argmax(scores)
                confidence = scores[classId]
                # filtering objects
                if confidence > confThreshold:
                    w, h = int(det[2] * wT), int(det[3] * hT)  # we multiply by wT or hT to get the pixel value
                    x, y = int((det[0] * wT) - w / 2), int((det[1] * hT) - h / 2)
                    bbox.append([x, y, w, h])
                    classIds.append(classId)
                    confs.append(float(confidence))

        indices = cv2.dnn.NMSBoxes(bbox, confs, confThreshold, nmsThreshold)

        for i in indices:
            i = i[0]
            box = bbox[i]
            x, y, w, h = box[0], box[1], box[2], box[3]
            color = colors[classIds[i]]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, 2)
            cv2.putText(img, f'{classNames[classIds[i]].upper()} {int(confs[i] * 100)}%', (x, y - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

    # creating a loop for detection
    while True:
        success, img = cap.read()  # this will give the image

        # now input image to the network, but network accepts/understands only particular type of format and
        # this format is basically blob, So we convert our image to blob
        blob = cv2.dnn.blobFromImage(img, 1 / 255, (whT, whT), [0, 0, 0], 1,
                                     crop=False)  # this will convert our image to blob
        net.setInput(blob)

        layerNames = net.getLayerNames()  # This will give all the names of our layers
        # print(layerNames)
        outputNames = [layerNames[i[0] - 1] for i in net.getUnconnectedOutLayers()]
        # print(outputNames)
        # print(net.getUnconnectedOutLayers())

        outputs = net.forward(outputNames)

        findObjects(outputs, img)

        cv2.imshow('Detect', img)  # to display the image
        cv2.waitKey(1)  # delay it for 1 millisecond


class Show:
    def __init__(self,root):
        self.root = root
        self.root.title("On Road Obstacle Detection")
        self.root.geometry("1199x600+100+50")
        self.root.resizable(False, False)
        # ============================#
        self.bg = ImageTk.PhotoImage(file="front.jpg")
        self.bg_image = Label(self.root, image=self.bg).place(x=0, y=0, relwidth=1, relheight=1)
        # ==============================#
        Frame_show = Frame(self.root, bg="white")
        Frame_show.place(x=150, y=150, height=300, width=700)

        title = Label(Frame_show, text="On Road Obstacle Detection", font=("Impact", 40), fg="#FF5733",
                      bg="white").place(x=40, y=10)
        subtitle = Label(Frame_show, text="Click any of the options below",
                         font=("times new roman", 20, "bold"), fg="#FF5733", bg="white").place(x=40, y=120)
        Show_btn = Button(Frame_show, text="Detect through Video", fg="white", bg="#FF5733", font=("times new roman", 20, "bold"),
                           command=viddetect).place(x=60, y=180)
        Show_btn2 = Button(Frame_show, text="Detect through Camera", fg="white", bg="#FF5733", font=("times new roman", 20, "bold"),
                           command=camdetect).place(x=360, y=180)

root=Tk()
obj=Show(root)
root.mainloop()
