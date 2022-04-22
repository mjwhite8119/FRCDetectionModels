#!/usr/bin/env python3

import numpy as np
import json
import threading
import time
from time import sleep
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from pathlib import Path
from socketserver import ThreadingMixIn
from PIL import Image
from pathlib import Path
import sys
import cv2
import depthai as dai
from networktables import NetworkTablesInstance

from wpi_helpers import ConfigParser, ThreadedHTTPServer, VideoStreamHandler, NetworkConfigParser

'''
Spatial Tiny-yolo example
  Performs inference on RGB camera and retrieves spatial location coordinates: x,y,z relative to the center of depth map.
  Can be used for tiny-yolo-v3 or tiny-yolo-v4 networks
'''
HTTP_SERVER = '10.0.0.2'
HTTP_SERVER_PORT = 8091

# class ConfigParser:
#     def __init__(self, config_path):
#         self.team = -1

#         # parse file
#         try:
#             with open(config_path, "rt", encoding="utf-8") as f:
#                 j = json.load(f)
#         except OSError as err:
#             print("could not open '{}': {}".format(config_path, err), file=sys.stderr)

#         # top level must be an object
#         if not isinstance(j, dict):
#             self.parseError("must be JSON object", config_path)

#         # team number
#         try:
#             self.team = j["team"]
#         except KeyError:
#             self.parseError("could not read team number", config_path)

#         # cameras
#         try:
#             self.cameras = j["cameras"]
#         except KeyError:
#             self.parseError("could not read cameras", config_path)

#     def parseError(self, str, config_file):
#         """Report parse error."""
#         print("config error in '" + config_file + "': " + str, file=sys.stderr)     

# # HTTPServer MJPEG
# class VideoStreamHandler(BaseHTTPRequestHandler):
#     def do_GET(self):
#         self.send_response(200)
#         self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
#         self.end_headers()
#         while True:
#             sleep(0.1)
#             if hasattr(self.server, 'frametosend'):
#                 image = Image.fromarray(cv2.cvtColor(self.server.frametosend, cv2.COLOR_BGR2RGB))
#                 stream_file = BytesIO()
#                 image.save(stream_file, 'JPEG')
#                 self.wfile.write("--jpgboundary".encode())

#                 self.send_header('Content-type', 'image/jpeg')
#                 self.send_header('Content-length', str(stream_file.getbuffer().nbytes))
#                 self.end_headers()
#                 image.save(self.wfile, 'JPEG')

# class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
#     """Handle requests in a separate thread."""
#     pass

# class NetworkConfigParser:
#     def __init__(self, path):
#         """
#         Parses the model config file and adjusts NNetManager values accordingly. 
#         It's advised to create a config file for every new network, as it allows to 
#         use dedicated NN nodes (for `MobilenetSSD <https://github.com/luxonis/depthai/blob/main/resources/nn/mobilenet-ssd/mobilenet-ssd.json>`__ 
#         and `YOLO <https://github.com/luxonis/depthai/blob/main/resources/nn/tiny-yolo-v3/tiny-yolo-v3.json>`__)
#         or use `custom handler <https://github.com/luxonis/depthai/blob/main/resources/nn/openpose2/openpose2.json>`__ 
#         to process and display custom network results

#         Args:
#             path (pathlib.Path): Path to model config file (.json)

#         Raises:
#             ValueError: If path to config file does not exist
#             RuntimeError: If custom handler does not contain :code:`draw` or :code:`show` methods
#         """
#         configPath = Path(path)
#         if not configPath.exists():
#             raise ValueError("Path {} does not exist!".format(path))

#         with configPath.open() as f:
#             configJson = json.load(f)
#             nnConfig = configJson.get("nn_config", {})
#             self.labelMap = configJson.get("mappings", {}).get("labels", None)
#             self.nnFamily = nnConfig.get("NN_family", None)
#             self.outputFormat = nnConfig.get("output_format", "raw")
#             metadata = nnConfig.get("NN_specific_metadata", {})
#             if "input_size" in nnConfig:
#                 self.inputSize = tuple(map(int, nnConfig.get("input_size").split('x')))

#             self.confidence_threshold = metadata.get("confidence_threshold", nnConfig.get("confidence_threshold", None))
#             self.classes = metadata.get("classes", None)
           
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
print("Running spacial_tiny_yolo_wpi.py")
print("Loading the model")
if not Path(nnPath).exists():
    print("No custom model found at path " + nnPath)
    nnPath = str((Path(__file__).parent / Path(default_blob_file)).resolve().absolute())
    configPath = str((Path(__file__).parent / Path(default_config_file)).resolve().absolute())
    print("Using:" + nnPath)
    print("with config file:", configPath)

## Read the model configuration file
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
spatialDetectionNetwork = pipeline.create(dai.node.YoloSpatialDetectionNetwork)
monoLeft = pipeline.create(dai.node.MonoCamera)
monoRight = pipeline.create(dai.node.MonoCamera)
stereo = pipeline.create(dai.node.StereoDepth)

xoutRgb = pipeline.create(dai.node.XLinkOut)
xoutNN = pipeline.create(dai.node.XLinkOut)
xoutBoundingBoxDepthMapping = pipeline.create(dai.node.XLinkOut)
xoutDepth = pipeline.create(dai.node.XLinkOut)

xoutRgb.setStreamName("rgb")
xoutNN.setStreamName("detections")
xoutBoundingBoxDepthMapping.setStreamName("boundingBoxDepthMapping")
xoutDepth.setStreamName("depth")

# Properties
camRgb.setPreviewSize(frame_width, frame_height)
camRgb.setResolution(dai.ColorCameraProperties.SensorResolution.THE_1080_P)
camRgb.setInterleaved(False)
camRgb.setColorOrder(dai.ColorCameraProperties.ColorOrder.BGR)

monoLeft.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoLeft.setBoardSocket(dai.CameraBoardSocket.LEFT)
monoRight.setResolution(dai.MonoCameraProperties.SensorResolution.THE_400_P)
monoRight.setBoardSocket(dai.CameraBoardSocket.RIGHT)

# setting node configs
stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

spatialDetectionNetwork.setBlobPath(nnPath)
spatialDetectionNetwork.setConfidenceThreshold(network_config_parser.confidence_threshold)
spatialDetectionNetwork.input.setBlocking(False)
spatialDetectionNetwork.setBoundingBoxScaleFactor(0.5)
spatialDetectionNetwork.setDepthLowerThreshold(100)
spatialDetectionNetwork.setDepthUpperThreshold(5000)

# Yolo specific parameters
spatialDetectionNetwork.setNumClasses(network_config_parser.classes)
spatialDetectionNetwork.setCoordinateSize(4)
spatialDetectionNetwork.setAnchors(np.array([10,14, 23,27, 37,58, 81,82, 135,169, 344,319]))
spatialDetectionNetwork.setAnchorMasks({ "side26": np.array([1,2,3]), "side13": np.array([3,4,5]) })
spatialDetectionNetwork.setIouThreshold(0.5)

# Linking
monoLeft.out.link(stereo.left)
monoRight.out.link(stereo.right)

camRgb.preview.link(spatialDetectionNetwork.input)
if syncNN:
    spatialDetectionNetwork.passthrough.link(xoutRgb.input)
else:
    camRgb.preview.link(xoutRgb.input)

spatialDetectionNetwork.out.link(xoutNN.input)
spatialDetectionNetwork.boundingBoxMapping.link(xoutBoundingBoxDepthMapping.input)

stereo.depth.link(spatialDetectionNetwork.inputDepth)
spatialDetectionNetwork.passthroughDepth.link(xoutDepth.input)

# Connect to device and start pipeline
print("Connecting to device and starting pipeline")
with dai.Device(pipeline) as device:

    # Output queues will be used to get the rgb frames and nn data from the outputs defined above
    previewQueue = device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
    detectionNNQueue = device.getOutputQueue(name="detections", maxSize=4, blocking=False)
    xoutBoundingBoxDepthMappingQueue = device.getOutputQueue(name="boundingBoxDepthMapping", maxSize=4, blocking=False)
    depthQueue = device.getOutputQueue(name="depth", maxSize=4, blocking=False)

    startTime = time.monotonic()
    counter = 0
    fps = 0
    color = (255, 255, 255)

    # Run detection loop
    while True:
        inPreview = previewQueue.get()
        inDet = detectionNNQueue.get()
        depth = depthQueue.get()

        frame = inPreview.getCvFrame()
        depthFrame = depth.getFrame() # depthFrame values are in millimeters

        depthFrameColor = cv2.normalize(depthFrame, None, 255, 0, cv2.NORM_INF, cv2.CV_8UC1)
        depthFrameColor = cv2.equalizeHist(depthFrameColor)
        depthFrameColor = cv2.applyColorMap(depthFrameColor, cv2.COLORMAP_HOT)

        counter+=1
        current_time = time.monotonic()
        if (current_time - startTime) > 1 :
            fps = counter / (current_time - startTime)
            counter = 0
            startTime = current_time

        detections = inDet.detections

        if len(detections) != 0:
            boundingBoxMapping = xoutBoundingBoxDepthMappingQueue.get()
            roiDatas = boundingBoxMapping.getConfigData()

            for roiData in roiDatas:
                roi = roiData.roi
                roi = roi.denormalize(depthFrameColor.shape[1], depthFrameColor.shape[0])
                topLeft = roi.topLeft()
                bottomRight = roi.bottomRight()
                xmin = int(topLeft.x)
                ymin = int(topLeft.y)
                xmax = int(bottomRight.x)
                ymax = int(bottomRight.y)

                cv2.rectangle(depthFrameColor, (xmin, ymin), (xmax, ymax), color, cv2.FONT_HERSHEY_SCRIPT_SIMPLEX)


        # If the frame is available, draw bounding boxes on it and show the frame
        height = frame.shape[0]
        width  = frame.shape[1]
        for detection in detections:
            # Denormalize bounding box
            x1 = int(detection.xmin * width)
            x2 = int(detection.xmax * width)
            y1 = int(detection.ymin * height)
            y2 = int(detection.ymax * height)
            try:
                label = network_config_parser.labelMap[detection.label]
            except:
                label = detection.label
            cv2.putText(frame, str(label), (x1 + 10, y1 + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, "{:.2f}".format(detection.confidence*100), (x1 + 10, y1 + 35), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"X: {int(detection.spatialCoordinates.x)} mm", (x1 + 10, y1 + 50), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"Y: {int(detection.spatialCoordinates.y)} mm", (x1 + 10, y1 + 65), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"Z: {int(detection.spatialCoordinates.z)} mm", (x1 + 10, y1 + 80), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)

            cv2.rectangle(frame, (x1, y1), (x2, y2), color, cv2.FONT_HERSHEY_SIMPLEX)

        cv2.putText(frame, "NN fps: {:.2f}".format(fps), (2, frame.shape[0] - 4), cv2.FONT_HERSHEY_TRIPLEX, 0.4, color)
        
        # Show the frame
        server_HTTP.frametosend = frame

        # cv2.imshow("depth", depthFrameColor)
        # cv2.imshow("rgb", frame)

        if cv2.waitKey(1) == ord('q'):
            break
