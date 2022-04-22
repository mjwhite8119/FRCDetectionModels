#!/usr/bin/env python3

"""
Get a tiny yolo v4 model and displays it to localhost:8091
"""

import json
import socketserver
import threading
import time
from time import sleep
from http.server import BaseHTTPRequestHandler, HTTPServer
from io import BytesIO
from pathlib import Path
import sys
from socketserver import ThreadingMixIn
from PIL import Image
from pathlib import Path
import sys
import cv2
import depthai as dai
import numpy as np
import csv
from networktables import NetworkTablesInstance

HTTP_SERVER = '10.0.0.2'
HTTP_SERVER_PORT = 8091
classes = None

class ConfigParser:
    def __init__(self, config_path):
        self.team = -1

        # parse file
        try:
            with open(config_path, "rt", encoding="utf-8") as f:
                j = json.load(f)
        except OSError as err:
            print("could not open '{}': {}".format(config_path, err), file=sys.stderr)

        # top level must be an object
        if not isinstance(j, dict):
            self.parseError("must be JSON object", config_path)

        # team number
        try:
            self.team = j["team"]
        except KeyError:
            self.parseError("could not read team number", config_path)

        # cameras
        try:
            self.cameras = j["cameras"]
        except KeyError:
            self.parseError("could not read cameras", config_path)

    def parseError(self, str, config_file):
        """Report parse error."""
        print("config error in '" + config_file + "': " + str, file=sys.stderr)


class TCPServerRequest(socketserver.BaseRequestHandler):
    def handle(self):
        # Handle is called each time a client is connected
        # When OpenDataCam connects, do not return - instead keep the connection open and keep streaming data
        # First send HTTP header
        header = 'HTTP/1.0 200 OK\r\nServer: Mozarella/2.2\r\nAccept-Range: bytes\r\nConnection: close\r\nMax-Age: 0\r\nExpires: 0\r\nCache-Control: no-cache, private\r\nPragma: no-cache\r\nContent-Type: application/json\r\n\r\n'
        self.request.send(header.encode())
        while True:
            sleep(0.1)
            if hasattr(self.server, 'datatosend'):
                self.request.send(self.server.datatosend.encode() + "\r\n".encode())

# HTTPServer MJPEG
class VideoStreamHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-type', 'multipart/x-mixed-replace; boundary=--jpgboundary')
        self.end_headers()
        while True:
            sleep(0.1)
            if hasattr(self.server, 'frametosend'):
                image = Image.fromarray(cv2.cvtColor(self.server.frametosend, cv2.COLOR_BGR2RGB))
                stream_file = BytesIO()
                image.save(stream_file, 'JPEG')
                self.wfile.write("--jpgboundary".encode())

                self.send_header('Content-type', 'image/jpeg')
                self.send_header('Content-length', str(stream_file.getbuffer().nbytes))
                self.end_headers()
                image.save(self.wfile, 'JPEG')

class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    pass

def readConfig(path):
        """
        Parses the model config file and adjusts NNetManager values accordingly. 
        It's advised to create a config file for every new network, as it allows to 
        use dedicated NN nodes (for `MobilenetSSD <https://github.com/luxonis/depthai/blob/main/resources/nn/mobilenet-ssd/mobilenet-ssd.json>`__ 
        and `YOLO <https://github.com/luxonis/depthai/blob/main/resources/nn/tiny-yolo-v3/tiny-yolo-v3.json>`__)
        or use `custom handler <https://github.com/luxonis/depthai/blob/main/resources/nn/openpose2/openpose2.json>`__ 
        to process and display custom network results

        Args:
            path (pathlib.Path): Path to model config file (.json)

        Raises:
            ValueError: If path to config file does not exist
            RuntimeError: If custom handler does not contain :code:`draw` or :code:`show` methods
        """
        global labelMap, classes, confidence_threshold
        configPath = Path(path)
        if not configPath.exists():
            raise ValueError("Path {} does not exist!".format(path))

        with configPath.open() as f:
            configJson = json.load(f)
            nnConfig = configJson.get("nn_config", {})
            labelMap = configJson.get("mappings", {}).get("labels", None)
            # nnFamily = nnConfig.get("NN_family", None)
            outputFormat = nnConfig.get("output_format", "raw")
            metadata = nnConfig.get("NN_specific_metadata", {})
            if "input_size" in nnConfig:
                inputSize = tuple(map(int, nnConfig.get("input_size").split('x')))

            confidence_threshold = metadata.get("confidence_threshold", nnConfig.get("confidence_threshold", None))
            classes = metadata.get("classes", None)
            # if 'handler' in configJson:
            #     handler = loadModule(configPath.parent / configJson["handler"])

            #     if not callable(getattr(self._handler, "draw", None)) or not callable(getattr(self._handler, "decode", None)):
            #         raise RuntimeError("Custom model handler does not contain 'draw' or 'decode' methods!")


# -------------------------------------------------------------------------
# Main Program Start
# -------------------------------------------------------------------------
config_file = "/boot/frc.json"
config_parser = ConfigParser(config_file)
hardware_type = "OAK-D Camera"
frame_width = 416
frame_height = 416
labelMap = None
classes = None
confidence_threshold = None

custom_blob_file = '../custom.blob'
custom_config_file = '../custom_config.json'
default_blob_file = 'yolo-v3-tiny-tf_openvino_2021.4_6shave.blob'
default_config_file = 'yolo-v3-tiny-tf.json'
nnPath = str((Path(__file__).parent / Path(custom_blob_file)).resolve().absolute())
configPath = str((Path(__file__).parent / Path(custom_config_file)).resolve().absolute())

# start TCP data server
server_TCP = socketserver.TCPServer(('localhost', 8070), TCPServerRequest)
th = threading.Thread(target=server_TCP.serve_forever)
th.daemon = True
th.start()


# start MJPEG HTTP Server
server_HTTP = ThreadedHTTPServer((HTTP_SERVER, HTTP_SERVER_PORT), VideoStreamHandler)
th2 = threading.Thread(target=server_HTTP.serve_forever)
th2.daemon = True
th2.start()

print("Loading the model")
if not Path(nnPath).exists():
    print("No custom model found at path " + nnPath)
    nnPath = str((Path(__file__).parent / Path(default_blob_file)).resolve().absolute())
    configPath = str((Path(__file__).parent / Path(default_config_file)).resolve().absolute())
    print("Using:" + nnPath)
    print("with config file:", configPath)
    # import sys
    # raise FileNotFoundError(f'No custom model found using "{nnPath}"')

# Network specific settings
print("Loading network settings")
readConfig(configPath)
print(labelMap)
print("Classes:", classes)
print("Confidence Threshold:", confidence_threshold)

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

# Network specific settings
detectionNetwork.setConfidenceThreshold(confidence_threshold)
detectionNetwork.setNumClasses(classes)
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
            cv2.putText(frame, labelMap[detection.label], (bbox[0] + 10, bbox[1] + 20), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.putText(frame, f"{int(detection.confidence * 100)}%", (bbox[0] + 10, bbox[1] + 40), cv2.FONT_HERSHEY_TRIPLEX, 0.5, 255)
            cv2.rectangle(frame, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)

            # Put to Network Tables
            temp_entry = []
            temp_entry.append({"label": labelMap[detection.label], "box": {"ymin": detection.ymin, "xmin": detection.xmin, 
                                "ymax": detection.ymax, "xmax": detection.xmax}, "confidence": int(detection.confidence * 100)})
            entry.setString(json.dumps(temp_entry))

            
        # Show the frame
        server_HTTP.frametosend = frame
        # cv2.imshow(name, frame)

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
