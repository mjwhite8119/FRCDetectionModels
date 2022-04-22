#!/usr/bin/env python3

"""
Get a tiny yolo v4 model and displays it to localhost:8091
"""

import json
import threading
import time
from time import sleep
from pathlib import Path
from pathlib import Path
import cv2
import depthai as dai
import numpy as np
from networktables import NetworkTablesInstance

from wpi_helpers import ConfigParser, ThreadedHTTPServer, VideoStreamHandler, NetworkConfigParser

HTTP_SERVER = '10.0.0.2'
HTTP_SERVER_PORT = 8091
           
# -------------------------------------------------------------------------
# Main Program Start
# -------------------------------------------------------------------------
config_file = "/boot/frc.json"
config_parser = ConfigParser(config_file)
hardware_type = "OAK-D Camera"
frame_width = 416
frame_height = 416

custom_blob_file = '../custom.blob'
custom_config_file = '../custom_config.json'
default_blob_file = 'yolo-v3-tiny-tf_openvino_2021.4_6shave.blob'
default_config_file = 'yolo-v3-tiny-tf.json'
nnPath = str((Path(__file__).parent / Path(custom_blob_file)).resolve().absolute())
configPath = str((Path(__file__).parent / Path(custom_config_file)).resolve().absolute())

# start MJPEG HTTP Server
server_HTTP = ThreadedHTTPServer((HTTP_SERVER, HTTP_SERVER_PORT), VideoStreamHandler)
th2 = threading.Thread(target=server_HTTP.serve_forever)
th2.daemon = True
th2.start()

# Load the model
print("Running tiny_yolo_wpi.py")
print("Loading the model")
if not Path(nnPath).exists():
    print("No custom model found at path " + nnPath)
    nnPath = str((Path(__file__).parent / Path(default_blob_file)).resolve().absolute())
    configPath = str((Path(__file__).parent / Path(default_config_file)).resolve().absolute())
    print("Using:" + nnPath)
    print("with config file:", configPath)

# Read the model configuration file
print("Loading network settings")
network_config_parser = NetworkConfigParser(configPath)
print(network_config_parser.labelMap)
print("Classes:", network_config_parser.classes)
print("Confidence Threshold:", network_config_parser.confidence_threshold)

# Connect to WPILib Network Tables
print("Connecting to Network Tables")
ntinst = NetworkTablesInstance.getDefault()
ntinst.startClientTeam(config_parser.team)
ntinst.startDSClient()
entry = ntinst.getTable("ML").getEntry("detections")

hardware_entry = ntinst.getTable("ML").getEntry("device")
fps_entry = ntinst.getTable("ML").getEntry("fps")
resolution_entry = ntinst.getTable("ML").getEntry("resolution")
hardware_entry.setString(hardware_type)
resolution_entry.setString(str(frame_width) + ", " + str(frame_height))

syncNN = True

# Configure and load the camera pipeline
print("Loading camera and model")
pipeline = dai.Pipeline()

# Define sources and outputs
camRgb = pipeline.create(dai.node.ColorCamera)
detectionNetwork = pipeline.create(dai.node.YoloDetectionNetwork)
xoutRgb = pipeline.create(dai.node.XLinkOut)
nnOut = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
nnOut.setStreamName("nn")

# Camera Properties
camRgb.setPreviewSize(frame_width, frame_height)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)
camRgb.setFps(40)

# Network model specific settings
detectionNetwork.setConfidenceThreshold(network_config_parser.confidence_threshold)
detectionNetwork.setNumClasses(network_config_parser.classes)
detectionNetwork.setCoordinateSize(4)
detectionNetwork.setAnchors(np.array([10, 14, 23, 27, 37, 58, 81, 82, 135, 169, 344, 319]))
detectionNetwork.setAnchorMasks({"side26": np.array([1, 2, 3]), "side13": np.array([3, 4, 5])})
detectionNetwork.setIouThreshold(0.5)
detectionNetwork.setBlobPath(nnPath)
detectionNetwork.setNumInferenceThreads(2)
detectionNetwork.input.setBlocking(False)

# Linking
camRgb.preview.link(detectionNetwork.input)
if syncNN:
    detectionNetwork.passthrough.link(xoutRgb.input)
else:
    camRgb.preview.link(xoutRgb.input)

detectionNetwork.out.link(nnOut.input)

# Connect to device and start pipeline
print("Connecting to device and starting pipeline")
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    detectionNNQueue = device.getOutputQueue(name="nn", maxSize=4, blocking=False)

    frame = None
    detections = []
    startTime = time.monotonic()
    counter = 0
    color2 = (255, 255, 255)

    # nn data, being the bounding box locations, are in <0..1> range - they need to be normalized with frame width/height
    def frameNorm(frame, bbox):
        normVals = np.full(len(bbox), frame.shape[0])
        normVals[::2] = frame.shape[1]
        return (np.clip(np.array(bbox), 0, 1) * normVals).astype(int)

    def displayFrame(name, frame):
        color = (255, 0, 0)
        for detection in detections:
            bbox = frameNorm(frame, (detection.xmin, detection.ymin, detection.xmax, detection.ymax))
            cv2.putText(frame, network_config_parser.labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            # Put to Network Tables
            temp_entry = []
            temp_entry.append({"label": network_config_parser.labelMap[detection.label], "box": {"ymin": detection.ymin, "xmin": detection.xmin, 
                                "ymax": detection.ymax, "xmax": detection.xmax}, "confidence": int(detection.confidence * 100)})
            entry.setString(json.dumps(temp_entry))
    
        # Show the frame
        server_HTTP.frametosend = frame

    # Run detection loop
    while True:
        if syncNN:
            inPreview = previewQueue.get()
            inNN = detectionNNQueue.get()
        else:
            inPreview = previewQueue.tryGet()
            inNN = detectionNNQueue.tryGet()

        detections = inNN.detections

        if inPreview is not None:
            frame = inPreview.getCvFrame()
            cv2.putText(frame, "NN fps: {:.2f}".format(counter / (time.monotonic() - startTime)),
                        (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color2)
            fps_entry.setNumber((counter / (time.monotonic() - startTime)))            
            
        if inNN is not None:
            detections = inNN.detections
            counter += 1

        if frame is not None:
            displayFrame("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            break
